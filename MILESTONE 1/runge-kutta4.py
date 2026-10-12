
import matplotlib.pyplot as plt
from numpy import array, zeros

N = 10000
Dt = 0.01
n_v = 2

U = zeros((N+1, n_v))
U[0, :] = array([1, 0])

def F(U):
    return array([U[1], -U[0]])

for n in range(N):

    # Runge-Kutta de cuarto orden
    k1 = F(U[n, :])
    k2 = F(U[n, :] + Dt/2 * k1)
    k3 = F(U[n, :] + Dt/2 * k2)
    k4 = F(U[n, :] + Dt * k3)

    U[n+1, :] = U[n, :] + Dt/6 * (k1 + 2*k2 + 2*k3 + k4)

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Runge-Kutta de cuarto orden (RK4)")
plt.axis('equal')
plt.grid(True)
plt.show()
