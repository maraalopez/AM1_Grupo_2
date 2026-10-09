import matplotlib.pyplot as plt
from numpy import zeros, array
from scipy.optimize import newton

# =====================================================================
# DEFINICIÓN DE LOS MÉTODOS INTEGRADORES (FUNCIONES REUTILIZABLES)
# =====================================================================

def euler(F, U, At, t):

    return U + At*F(U, t)

def euler_inverso(F, U0, At, N, max_iter=10):
    """Método de Euler Inverso / Implícito (Avance con derivada en t_{n+1}).
    
    Ecuación: U_{n+1} = U_n + Δt * F(U_{n+1})
    """
    Nv = len(U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0
    
    for n in range(N):
        # Predicción inicial usando Euler explícito
        U_next = U[n, :] + At * F(U[n, :])
        
        # Punto fijo para resolver la ecuación implícita
        for _ in range(max_iter):
            U_next = U[n, :] + At * F(U_next)
            
        U[n+1, :] = U_next
        
    return U

def cauchy_problem(F, U0, t, scheme):
    """Integración numérica por el Método de Euler Explícito.
    
    Argumentos:
      F  : Función que define el sistema diferencial dU/dt = F(U)
      U0 : Array con las condiciones iniciales
      t : time domain (vector)
      scheme: esquema temporal 
    """
    Nv = len(U0)
    N = len(t) - 1
    U = zeros((N+1, Nv))
    U[0, :] = U0
    
    for n in range(0,N):
        U[n+1, :] = scheme(F, U[n, :], t[n+1] - t[n], t[n])
      
    return U


def euler_inverso(F, U0, At, N, max_iter=10):
    """Método de Euler Inverso / Implícito (Avance con derivada en t_{n+1}).
    
    Ecuación: U_{n+1} = U_n + Δt * F(U_{n+1})
    """
    Nv = len(U0)
    U = zeros((N + 1, Nv))
    U[0, :] = U0
    
    for n in range(N):
        # Predicción inicial usando Euler explícito
        U_next = U[n, :] + At * F(U[n, :])
        
        # Punto fijo para resolver la ecuación implícita
        for _ in range(max_iter):
            U_next = U[n, :] + At * F(U_next)
            
        U[n+1, :] = U_next
        
    return U


def crank_nicholson(F, U, At, t):
    def G(X):
        return X + A - At/2*(F(X, t + At))

    A = -U - At/2 * F(U, t)

    return newton(func = G, x0 = U)


def rk4(F, U, At, t):
    """Integración numérica por el Método de Runge-Kutta de 4º Orden."""
   
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
        
    return U + At * (k1 + 2*k2 + 2*k3 + k4) / 6


def oscilador(U,t):
    return array([U[1], -U[0]])

def partition(a,b,N):
    """Devuelve un vector de N+1 puntos equiespaciados entre a y b. N es el número de segmentos"""
    return array([a + i*(b-a)/N for i in range( 0, N+1)])

U = cauchy_problem( F = oscilador, U0 = array([1, 0]), t = partition(a=0, b=10, N=100), scheme = euler)

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Euler")
plt.axis('equal')
plt.grid(True)
plt.show()

U = cauchy_problem( F = oscilador, U0 = array([1, 0]), t = partition(a=0, b=10, N=100), scheme = rk4)

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Runge-Kutta de 4º Orden")
plt.axis('equal')
plt.grid(True)
plt.show()

U = cauchy_problem( F = oscilador, U0 = array([1, 0]), t = partition(a=0, b=10, N=100), scheme = crank_nicholson)

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Crank-Nicolson")
plt.axis('equal')
plt.grid(True)
plt.show()

