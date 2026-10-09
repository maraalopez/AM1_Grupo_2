import matplotlib.pyplot as plt
from numpy import array, zeros

def F(U):
    return array([U[1], -U[0]])

N = 1000
dt = 0.1
Nv = 2
U = zeros((N+1, Nv))
U[0, :] = array([1, 0])

for n in range(0, N):
    # Predicción inicial (Euler)
    U_next = U[n, :] + dt * F(U[n, :])
    
    # Iteración de punto fijo para el paso implícito
    for _ in range(10):
        U_next = U[n, :] + (dt / 2) * (F(U[n, :]) + F(U_next))
        
    U[n+1, :] = U_next

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Crank-Nicolson")
plt.axis('equal')
plt.grid(True)
plt.show()