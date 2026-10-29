"""
fisica.py  —  CAPA FÍSICA
Ecuaciones de Euler del sólido rígido libre, su Jacobiano, las integrales
primeras y la geometría de Poinsot (semiejes y separatriz).

No importa nada de la capa numérica ni de la GUI: solo física.
Convención: omega = [ω1, ω2, ω3] en ejes cuerpo, I = [I1, I2, I3].
"""
import numpy as np
from typing import List, Optional, Tuple

import esquemas_temporales as et


def euler_equations(omega: np.ndarray, I: List[float], t: float) -> np.ndarray:
    """dω/dt = F(ω, t) para el sólido rígido libre (ecuaciones de Euler)."""
    domega_dt = np.zeros(3)
    domega_dt[0] = (I[1] - I[2]) * omega[1] * omega[2] / I[0]
    domega_dt[1] = (I[2] - I[0]) * omega[2] * omega[0] / I[1]
    domega_dt[2] = (I[0] - I[1]) * omega[0] * omega[1] / I[2]
    return domega_dt


def jacobian(omega: np.ndarray, I: List[float], eps: float = 1e-8) -> np.ndarray:
    """
    Calcula numéricamente la matriz Jacobiana de las ecuaciones de Euler.
    """
    n = len(omega)
    J = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        # Creamos copias del vector de estado para perturbar solo la componente j
        w_plus = list(omega)
        w_minus = list(omega)
        
        w_plus[j] += eps
        w_minus[j] -= eps
        
        # Evaluamos la EDO en los puntos perturbados
        f_plus = euler_equations(np.asarray(w_plus), I, 0.0)
        f_minus = euler_equations(np.asarray(w_minus), I, 0.0)
        
        # Columna j del Jacobiano (derivada respecto a w_j)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
        
    return J




def invariant_energy(omega: np.ndarray, I: List[float]) -> float:
    """2T = I1 ω1² + I2 ω2² + I3 ω3²   (devuelve 2T, no T)."""
    return I[0] * omega[0]**2 + I[1] * omega[1]**2 + I[2] * omega[2]**2


def invariant_angular_momentum(omega: np.ndarray, I: List[float]) -> float:
    """L² = I1² ω1² + I2² ω2² + I3² ω3²   (devuelve L², no L)."""
    return I[0]**2 * omega[0]**2 + I[1]**2 * omega[1]**2 + I[2]**2 * omega[2]**2


def semiejes_poinsot(I: List[float], dos_T0: float,
                     L0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Semiejes de las dos superficies invariantes en el espacio (ω1, ω2, ω3):
        elipsoide de energía   2T = Σ Ii ωi²    ->  semiejes sqrt(2T / Ii)
        elipsoide de momento   L² = Σ Ii² ωi²   ->  semiejes L / Ii
    OJO: L0 es el MÓDULO del momento angular (no L²).
    """
    I = np.asarray(I, dtype=float)
    return np.sqrt(dos_T0 / I), L0 / I


def separatriz(I: List[float], Omega_norm: float,
               num_pasos: int = 300) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Calcula los dos lazos de la separatriz para un ``Omega_norm`` dado."""
    if Omega_norm <= 0:
        raise ValueError("Omega_norm debe ser positivo.")
    if num_pasos < 2:
        raise ValueError("num_pasos debe ser al menos 2.")

    w_eq = np.array([0.0, Omega_norm, 0.0])
    J = jacobian(w_eq, I)
    autovalores, autovectores = np.linalg.eig(J)

    idx_positivo = np.argmax(np.real(autovalores))
    if np.real(autovalores[idx_positivo]) <= 0:
        return None
    autovector_inestable = np.real(autovectores[:, idx_positivo])

    epsilon = 1e-4
    t_sep = et.partition(0, 25.0, num_pasos)

    def f_sistema(w: np.ndarray, t_val: float) -> np.ndarray:
        return euler_equations(w, I, t_val)

    w_ini = w_eq + epsilon * autovector_inestable
    sep1 = np.asarray(
        et.cauchy_problem(F=f_sistema, U0=w_ini, t=t_sep, scheme=et.rk4)
    )

    rama1 = sep1
    rama2 = sep1 * np.array([-1, 1, -1])
    rama3 = sep1 * np.array([1, 1, -1])
    rama4 = sep1 * np.array([-1, 1, 1])

    lazo_A = np.vstack((rama1, rama2[::-1]))
    lazo_B = np.vstack((rama3, rama4[::-1]))
    return lazo_A, lazo_B

def orientacion(t, U):
    """
    Calcula la evolución de la actitud del satélite (matrices de rotación 3x3).
    Transfiere la cinemática física a la capa matemática para obtener las 
    matrices de rotación incrementales.
    """
    R = [np.eye(3)]
    
    for k in range(len(t) - 1):
        # 1. Calculamos la velocidad angular media en el intervalo
        omega_medio = 0.5 * (U[k] + U[k + 1])
        
        # 2. Calculamos el vector de desplazamiento angular instantáneo (v = w * dt)
        dt = t[k + 1] - t[k]
        vector_giro = omega_medio * dt
        
        # 3. matriz exacta de Rodrigues
        R_incremental = et.rotacion(vector_giro)
        
        # 4. Componemos la actitud actualizando la orientación anterior
        R.append(R[-1] @ R_incremental)
        
    return np.array(R)


def ejecutar_simulacion(I, w0, dt, esquema_func, TF=20.0):
    """
    Motor principal de la simulación. 
    Totalmente independiente de interfaces gráficas.
    """
    t = et.partition(0, TF, int(TF / dt))
    
    def F(omega, t_val):
        return euler_equations(omega, I, t_val)
        
    with np.errstate(all='ignore'):
        U = np.asarray(et.cauchy_problem(F=F, U0=w0, t=t, scheme=esquema_func))
        dT = np.asarray(et.calcular_deriva(U, invariant_energy, I))
        dL = np.asarray(et.calcular_deriva(U, invariant_angular_momentum, I))
        
    # Recorte (truncamiento) de la simulación si el método diverge
    valido = np.isfinite(U).all(axis=1)
    if not valido.all():
        n = max(int(valido.argmin()), 2)
        t, U, dT, dL = t[:n], U[:n], dT[:n], dL[:n]
        
    # Calculamos la actitud con la función que acabas de mover a física
    R = orientacion(t, U) 
    
    return t, U, dT, dL, R
