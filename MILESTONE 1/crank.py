import matplotlib.pyplot as plt
from numpy import array, zeros
from numpy.linalg import norm

def F(U):
    return array([U[1], -U[0]])

N = 10000
dt = 0.01
n_v = 2
tol = 1e-10
U = zeros((N+1, n_v))
U[0, :] = array([1, 0])

for n in range(0, N):
    U_euler= U[n, :] + dt * F(U[n, :]) # Uso Euler para obetener aproximacion inicial  

    while True:
            U_sig = U[n, :] + (dt / 2) * (F(U[n, :]) + F(U_euler))

            if norm(U_sig - U_euler) < tol:
               U_euler = U_sig
            break

            U_euler = U_sig
    U[n+1, :] = U_euler

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Crank-Nicolson")
plt.axis('equal')
plt.grid(True)
plt.show()
