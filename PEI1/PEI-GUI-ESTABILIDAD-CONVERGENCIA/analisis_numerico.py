"""
analisis_numerico.py  —  CAPA MATEMÁTICA
Estabilidad absoluta (factores de amplificación R(z)), error por
extrapolación de Richardson y ratio de convergencia.
No sabe nada de la física: todo recibe F o z como argumento.
"""
from typing import Callable

import numpy as np

from esquemas_temporales import cauchy_problem, FuncionF

# ---------------------------------------------------------------------------
# Factores de amplificación:  para dU/dt = λU, un paso da U_{n+1} = R(z) U_n,
# con z = λ·Δt.  El esquema es estable si |R(z)| <= 1.
# ---------------------------------------------------------------------------
FactorR = Callable[[np.ndarray], np.ndarray]


def R_euler(z: np.ndarray) -> np.ndarray:
    return 1 + z


def R_euler_inverso(z: np.ndarray) -> np.ndarray:
    return 1 / (1 - z)


def R_crank_nicolson(z: np.ndarray) -> np.ndarray:
    return (1 + z / 2) / (1 - z / 2)


def R_rk4(z: np.ndarray) -> np.ndarray:
    return 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24


def malla_region(R: FactorR, x_lim: tuple[float, float], y_lim: tuple[float, float],
                 n: int = 300) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """|R(z)| en una malla del plano complejo. Para dibujar: contourf(X, Y, absR, [0, 1])."""
    X, Y = np.meshgrid(np.linspace(*x_lim, n), np.linspace(*y_lim, n))
    with np.errstate(all='ignore'):                 # 1/(1-z) en z = 1
        absR = np.abs(R(X + 1j * Y))
    return X, Y, absR


def amplificacion_maxima(R: FactorR, J: np.ndarray, dt: float) -> tuple[float, np.ndarray]:
    """
    Autovalores λ_i de J -> z_i = λ_i Δt -> max |R(z_i)|.
    Si es <= 1 el paso es (linealmente) estable.  Devuelve (max|R|, z).

    Los autovalores nulos se ignoran en el máximo: R(0) = 1 en todos los
    esquemas consistentes, así que no dicen nada (son direcciones neutras,
    p. ej. cambiar Ω lleva a otro equilibrio de la misma familia).
    """
    lam = np.linalg.eigvals(J)
    z = lam * dt
    no_nulos = np.abs(lam) > 1e-10 * max(np.abs(lam).max(), 1e-300)
    if not no_nulos.any():
        return 1.0, z
    return float(np.max(np.abs(R(z[no_nulos])))), z


# ---------------------------------------------------------------------------
# Error de Richardson y ratio de convergencia
# ---------------------------------------------------------------------------
def richardson_error_temporal(F: FuncionF, U0: np.ndarray, t: np.ndarray,
                              scheme: Callable, q: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Error de la solución con malla t, estimado con una malla el doble de fina:
        E(t_n) = (U_2N(t_n) - U_N(t_n)) / (1 - 1/2^q)
    Devuelve (U_N, E), ambos (N+1 x Nv).
    """
    t2 = np.zeros(2 * len(t) - 1)
    t2[0::2] = t
    t2[1::2] = (t[:-1] + t[1:]) / 2
    U1 = cauchy_problem(F, U0, t, scheme)
    U2 = cauchy_problem(F, U0, t2, scheme)
    return U1, (U2[0::2] - U1) / (1 - 1 / 2**q)


def ratio_convergencia(F: FuncionF, U0: np.ndarray, T: float, N_values: list[int],
                       scheme: Callable) -> tuple[float, np.ndarray]:
    """
    d(N) = |U_2N(T) - U_N(T)| ~ C / N^q  ->  q = -pendiente de log10 d frente a log10 N.
    Descarta los puntos en el error de redondeo (d < 1e-11) y, si sobran,
    el primero (malla aún no asintótica).  Devuelve (q, d).
    """
    d = []
    for N in N_values:
        U_N = cauchy_problem(F, U0, np.linspace(0, T, N + 1), scheme)
        U_2N = cauchy_problem(F, U0, np.linspace(0, T, 2 * N + 1), scheme)
        d.append(np.linalg.norm(U_2N[-1] - U_N[-1]))
    d = np.array(d)

    with np.errstate(all='ignore'):
        logN, logd = np.log10(np.array(N_values, dtype=float)), np.log10(d)
    validos = np.isfinite(logd) & (logd > -11)
    if validos.sum() >= 3:
        validos[np.argmax(validos)] = False
    if validos.sum() < 2:
        return float('nan'), d
    pendiente, _ = np.polyfit(logN[validos], logd[validos], 1)
    return -pendiente, d
