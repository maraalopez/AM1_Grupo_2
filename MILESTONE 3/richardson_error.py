import numpy as np
from esquemas_matematicos import  cauchy

def richardson_error(F, U0, T, N, scheme, p):
    """
    Estima el error global mediante extrapolación de Richardson.

    F: función del problema de Cauchy
    U0: condición inicial
    T: tiempo final
    N: número de pasos de la malla más gruesa
    scheme: método numérico
    p: orden teórico del método
    """

    # Malla gruesa: paso h
    t_h = np.linspace(0, T, N + 1)
    U_h = cauchy(F, U0, t_h, scheme)

    # Malla fina: paso h/2
    t_h2 = np.linspace(0, T, 2*N + 1)
    U_h2 = cauchy(F, U0, t_h2, scheme)

    # Comparamos las soluciones en el tiempo final
    difference = np.linalg.norm(U_h2[-1] - U_h[-1])

    # Error estimado de la solución fina
    error = difference / (2**p - 1)

    return error

