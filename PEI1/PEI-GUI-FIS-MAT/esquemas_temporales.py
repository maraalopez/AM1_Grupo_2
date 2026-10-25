"""
esquemas_temporales.py  —  CAPA MATEMÁTICA
Esquemas temporales, problema de Cauchy y utilidades numéricas/geométricas.
No sabe nada de la física concreta: todo recibe F o la función a evaluar.
"""
from typing import Callable

import numpy as np
from scipy.optimize import fsolve

# Tipo de la función del problema: F(U, t) -> dU/dt
FuncionF = Callable[[np.ndarray, float], np.ndarray]


def euler(F: FuncionF, U: np.ndarray, At: float, t: float) -> np.ndarray:
    return U + At * F(U, t)


def crank_nicholson(F: FuncionF, U: np.ndarray, At: float, t: float) -> np.ndarray:
    """U_{n+1} = U_n + At/2 [F(U_n) + F(U_{n+1})]   (implícito, se resuelve con fsolve)."""
    a = U + At / 2 * F(U, t)                 # parte explícita, se calcula una vez

    def G(X: np.ndarray) -> np.ndarray:
        return X - a - At / 2 * F(X, t + At)

    return fsolve(G, euler(F, U, At, t))     # Euler como estimación inicial


def rk4(F: FuncionF, U: np.ndarray, At: float, t: float) -> np.ndarray:
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def cauchy_problem(F: FuncionF, U0: np.ndarray, t: np.ndarray,
                   scheme: Callable) -> np.ndarray:
    """Integra dU/dt = F(U, t) en la malla t. Devuelve U (N+1 x Nv)."""
    N = len(t) - 1
    U = np.zeros((N + 1, len(U0)))
    U[0, :] = U0
    for n in range(N):
        U[n + 1, :] = scheme(F, U[n, :], t[n + 1] - t[n], t[n])
    return U


def partition(a: float, b: float, N: int) -> np.ndarray:
    """N+1 puntos equiespaciados entre a y b (N segmentos)."""
    return np.array([a + i * (b - a) / N for i in range(N + 1)])


def calcular_deriva(U_simulado: np.ndarray,
                    funcion_invariante: Callable[[np.ndarray, list[float]], float],
                    parametros_fisicos: list[float]) -> np.ndarray:
    """Deriva absoluta |I(t) - I(0)| de cualquier invariante a lo largo de la solución."""
    valor_referencia = funcion_invariante(U_simulado[0, :], parametros_fisicos)
    deriva = np.zeros(len(U_simulado))
    for i in range(len(U_simulado)):
        deriva[i] = abs(funcion_invariante(U_simulado[i, :], parametros_fisicos)
                        - valor_referencia)
    return deriva


def generar_malla_elipsoide(a: float, b: float, c: float,
                            num_puntos: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Malla (X, Y, Z) de un elipsoide de semiejes (a, b, c)."""
    u = np.linspace(0, 2 * np.pi, num_puntos)
    v = np.linspace(0, np.pi, num_puntos)
    U, V = np.meshgrid(u, v)
    return a * np.cos(U) * np.sin(V), b * np.sin(U) * np.sin(V), c * np.cos(V)


def rotacion(v: np.ndarray) -> np.ndarray:
    """Matriz de rotación de ángulo |v| alrededor del eje v (fórmula de Rodrigues)."""
    ang = np.linalg.norm(v)
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
