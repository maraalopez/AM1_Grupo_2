"""
calculo.py  —  ORQUESTADOR (puente entre capas)
Junta la física (fisica.py) con los esquemas numéricos (esquemas_temporales.py).
No dibuja nada. Es el único módulo que conoce a las dos capas.
"""
from typing import Callable

import numpy as np

import fisica as f
import esquemas_temporales as et
from ajustes import ESQUEMAS, TF, SIM_POR_FRAME, w, estado


def orientacion(t: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Actitud del satélite: matrices de rotación R_k (cuerpo -> inercial).
    Cada paso compone un giro incremental de vector ω_medio·Δt (Rodrigues).
    Devuelve un array (N+1 x 3 x 3).
    """
    R = [np.eye(3)]
    for k in range(len(t) - 1):
        omega_medio = 0.5 * (U[k] + U[k + 1])
        R.append(R[-1] @ et.rotacion(omega_medio * (t[k + 1] - t[k])))
    return np.array(R)


def superficies_poinsot(I: list[float], dos_T0: float, L0: float,
                        num_puntos: int = 50) -> tuple[tuple, tuple]:
    """Mallas de los elipsoides de energía y de momento (L0 = módulo de L)."""
    ejes_T, ejes_L = f.semiejes_poinsot(I, dos_T0, L0)
    return (et.generar_malla_elipsoide(*ejes_T, num_puntos),
            et.generar_malla_elipsoide(*ejes_L, num_puntos))


def ejecutar_simulacion(I: list[float], w0: np.ndarray, dt: float,
                        esquema_func: Callable, TF: float = 20.0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integra ω(t), calcula la deriva de 2T y L² y la actitud.
    Independiente de la GUI. Devuelve (t, U, dT, dL, R).
    """
    t = et.partition(0, TF, int(TF / dt))

    def F(omega: np.ndarray, t_val: float) -> np.ndarray:
        return f.euler_equations(omega, I, t_val)

    with np.errstate(all='ignore'):          # silencia avisos si un esquema diverge
        U = et.cauchy_problem(F, w0, t, esquema_func)
        dT = et.calcular_deriva(U, f.invariant_energy, I)
        dL = et.calcular_deriva(U, f.invariant_angular_momentum, I)

    # Si el esquema "explota" (p. ej. Euler con Δt grande) nos quedamos con la parte válida
    valido = np.isfinite(U).all(axis=1)
    if not valido.all():
        n = max(int(valido.argmin()), 2)
        t, U, dT, dL = t[:n], U[:n], dT[:n], dL[:n]

    return t, U, dT, dL, orientacion(t, U)


def resolver() -> None:
    """Lee los controles de la GUI, lanza la simulación y guarda todo en `estado`."""
    I = [w['I1'].val, w['I2'].val, w['I3'].val]
    w0 = np.array([w['w1'].val, w['w2'].val, w['w3'].val])
    dt = w['dt'].val
    esquema = ESQUEMAS[w['esquema'].value_selected]

    t, U, dT, dL, R = ejecutar_simulacion(I, w0, dt, esquema, TF)
    estado.update(I=I, w0=w0, t=t, U=U, dT=dT, dL=dL, R=R,
                  k=0, paso=max(1, round(SIM_POR_FRAME / dt)))
