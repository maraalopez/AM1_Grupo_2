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


