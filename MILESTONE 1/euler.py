import matplotlib.pyplot as plt
from numpy import array, zeros

def F(U):
    return array([U[1], -U[0]])

N = 10000
dt = 0.001
n_v = 2

U = zeros((N+1, n_v))
U[0, :] = array([1, 0])

for n in range(0, N):
    U[n+1, :] = U[n, :] + dt * F(U[n, :])

plt.plot(U[:, 0], U[:, 1])
plt.title("Método de Euler")
plt.axis('equal')
plt.grid(True)
plt.show()