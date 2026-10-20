import matplotlib.pyplot as plt
from numpy import zeros, array
from scipy.optimize import newton


def euler(F, U, At, t):

    return U + At*F(U, t)


def cauchy_problem(F, U0, t, scheme):

    Nv = len(U0)
    N = len(t) - 1
    U = zeros((N+1, Nv))
    U[0, :] = U0
    
    for n in range(0,N):
        U[n+1, :] = scheme(F, U[n, :], t[n+1] - t[n], t[n])
      
    return U



def crank_nicholson(F, U, At, t):
    def G(X):
        return X + A - At/2*(F(X, t + At))

    A = -U - At/2 * F(U, t)

    return newton(func = G, x0 = U)


def rk4(F, U, At, t):
   
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
        
    return U + At * (k1 + 2*k2 + 2*k3 + k4) / 6


def partition(a,b,N):
    """Devuelve un vector de N+1 puntos equiespaciados entre a y b. N es el número de segmentos"""
    return array([a + i*(b-a)/N for i in range( 0, N+1)])


def calcular_deriva(U_simulado, funcion_invariante, parametros_fisicos):
    """
    Calcula la deriva numérica (error absoluto) de cualquier invariante físico.
    """
    # 1. El valor exacto de referencia es el del instante inicial (t=0)
    estado_inicial = U_simulado[0, :]
    valor_referencia = funcion_invariante(estado_inicial, parametros_fisicos)
    
    # 2. Inicializamos el vector de deriva
    num_pasos = len(U_simulado)
    deriva = np.zeros(num_pasos)
    
    # 3. Calculamos la diferencia absoluta en cada paso
    for i in range(num_pasos):
        valor_actual = funcion_invariante(U_simulado[i, :], parametros_fisicos)
        deriva[i] = abs(valor_actual - valor_referencia)
        
    return deriva

import numpy as np

def generar_malla_elipsoide(a, b, c, num_puntos=50):
    """
    Función puramente matemática/geométrica.
    Genera la malla 3D de un elipsoide genérico dados sus tres semiejes (a, b, c).
    """
    u = np.linspace(0, 2 * np.pi, num_puntos)
    v = np.linspace(0, np.pi, num_puntos)
    U, V = np.meshgrid(u, v)
    
    # Parametrización estándar del elipsoide
    X = a * np.cos(U) * np.sin(V)
    Y = b * np.sin(U) * np.sin(V)
    Z = c * np.cos(V)
    
    return X, Y, Z


