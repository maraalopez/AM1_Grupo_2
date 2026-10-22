import matplotlib.pyplot as plt
from numpy import zeros, array
from numpy.linalg import norm
from scipy.optimize import fsolve


# DEFINICIÓN DE LOS ESQUEMAS MATEMÁTICOS 

# METODO DE EULER
def euler(F, U, At, t):

    return U + At*F(U, t)

# METODO EULER INVERSO

def euler_inverso(F, U, At, t):
    
    def inverso(U_sig):
        
        return U_sig - U - At * F(U_sig, t + At)
    
    U_aprox = euler(F, U, At, t)  # Aproximacion inicial con Euler
    
    U_sig = fsolve(inverso, U_aprox)
    
    return U_sig

# METODO DE CRANK-NICHOLSON

def crank_nicholson(F, U, At, t):
    
    def crank(U_sig):
        
        return U_sig - U - (At / 2.0) * (F(U, t) + F(U_sig, t + At))
    
    U_aprox = euler(F, U, At, t)
    
    U_sig = fsolve(crank, U_aprox)
    
    return U_sig


# METODO DE RUNGE-KUTTA DE 4º ORDEN

def rk4(F, U, At, t):
   
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
        
    return U + At * (k1 + 2*k2 + 2*k3 + k4) / 6 

# METODO DE CAUCHY


def cauchy(F, U0, t, scheme):
    """
    Argumentos:
      F  : Función que define el sistema diferencial dU/dt = F(U)
      U0 : Array con las condiciones iniciales
      t : time domain (vector)
      scheme: esquema temporal 
    """
    n_v = len(U0)
    N = len(t) - 1
    U = zeros((N+1, n_v))
    U[0, :] = U0
    
    for n in range(0,N):
        U[n+1, :] = scheme(F, U[n, :], t[n+1] - t[n], t[n])
      
    return U

