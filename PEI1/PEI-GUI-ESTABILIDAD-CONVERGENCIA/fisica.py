"""
fisica.py  —  CAPA FÍSICA
Ecuaciones de Euler del sólido rígido libre, su Jacobiano, las integrales
primeras y la geometría de Poinsot (semiejes y separatriz).

No importa nada de la capa numérica ni de la GUI: solo física.
Convención: omega = [ω1, ω2, ω3] en ejes cuerpo, I = [I1, I2, I3].
"""
import numpy as np


def euler_equations(omega: np.ndarray, I: list[float], t: float) -> np.ndarray:
    """dω/dt = F(ω, t) para el sólido rígido libre (ecuaciones de Euler)."""
    domega_dt = np.zeros(3)
    domega_dt[0] = (I[1] - I[2]) * omega[1] * omega[2] / I[0]
    domega_dt[1] = (I[2] - I[0]) * omega[2] * omega[0] / I[1]
    domega_dt[2] = (I[0] - I[1]) * omega[0] * omega[1] / I[2]
    return domega_dt


def jacobian(omega: np.ndarray, I: list[float]) -> np.ndarray:
    """Jacobiano exacto J = ∂F/∂ω (matriz 3x3)."""
    J = np.zeros((3, 3))
    J[0, 1] = (I[1] - I[2]) * omega[2] / I[0]
    J[0, 2] = (I[1] - I[2]) * omega[1] / I[0]
    J[1, 0] = (I[2] - I[0]) * omega[2] / I[1]
    J[1, 2] = (I[2] - I[0]) * omega[0] / I[1]
    J[2, 0] = (I[0] - I[1]) * omega[1] / I[2]
    J[2, 1] = (I[0] - I[1]) * omega[0] / I[2]
    return J


def invariant_energy(omega: np.ndarray, I: list[float]) -> float:
    """2T = I1 ω1² + I2 ω2² + I3 ω3²   (devuelve 2T, no T)."""
    return I[0] * omega[0]**2 + I[1] * omega[1]**2 + I[2] * omega[2]**2


def invariant_angular_momentum(omega: np.ndarray, I: list[float]) -> float:
    """L² = I1² ω1² + I2² ω2² + I3² ω3²   (devuelve L², no L)."""
    return I[0]**2 * omega[0]**2 + I[1]**2 * omega[1]**2 + I[2]**2 * omega[2]**2


def semiejes_poinsot(I: list[float], dos_T0: float,
                     L0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Semiejes de las dos superficies invariantes en el espacio (ω1, ω2, ω3):
        elipsoide de energía   2T = Σ Ii ωi²    ->  semiejes sqrt(2T / Ii)
        elipsoide de momento   L² = Σ Ii² ωi²   ->  semiejes L / Ii
    OJO: L0 es el MÓDULO del momento angular (no L²).
    """
    I = np.asarray(I, dtype=float)
    return np.sqrt(dos_T0 / I), L0 / I


def separatriz(I: list[float], dos_T0: float,
               num_puntos: int = 300) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Separatriz crítica sobre el elipsoide de energía 2T0, calculada de forma
    analítica (sin integrar nada).

    Es la polhodia con L² = 2 I_int T, la que pasa por los dos puntos de silla
    (rotación pura sobre el eje intermedio, ±Ω_s con Ω_s = sqrt(2T0 / I_int)).
    Está formada por dos elipses planas contenidas en los planos
        ω_max = ± k ω_min,     k = sqrt( I_min (I_int - I_min) / (I_max (I_max - I_int)) )
    y parametrizadas por θ ∈ [0, 2π]:
        ω_min = A cos θ,   ω_int = Ω_s sin θ,   ω_max = ± C cos θ
        A = Ω_s sqrt( I_int (I_max - I_int) / (I_min (I_max - I_min)) )
        C = Ω_s sqrt( I_int (I_int - I_min) / (I_max (I_max - I_min)) )

    Funciona aunque los momentos no estén ordenados (I1 < I2 < I3 no hace falta):
    se busca qué eje es el intermedio. Si dos momentos son iguales no hay eje
    intermedio y devuelve None.

    Devuelve (lazo_1, lazo_2), cada uno un array (num_puntos x 3).
    """
    I = np.asarray(I, dtype=float)
    i_min, i_int, i_max = np.argsort(I)
    Ia, Ib, Ic = I[i_min], I[i_int], I[i_max]
    if Ib - Ia < 1e-9 * Ic or Ic - Ib < 1e-9 * Ic:
        return None

    Omega_s = np.sqrt(dos_T0 / Ib)
    A = Omega_s * np.sqrt(Ib * (Ic - Ib) / (Ia * (Ic - Ia)))
    C = Omega_s * np.sqrt(Ib * (Ib - Ia) / (Ic * (Ic - Ia)))
    theta = np.linspace(0, 2 * np.pi, num_puntos)

    lazos = []
    for signo in (1, -1):
        lazo = np.zeros((num_puntos, 3))
        lazo[:, i_min] = A * np.cos(theta)
        lazo[:, i_int] = Omega_s * np.sin(theta)
        lazo[:, i_max] = signo * C * np.cos(theta)
        lazos.append(lazo)
    return lazos[0], lazos[1]
