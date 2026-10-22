
import matplotlib.pyplot as plt
from numpy import linspace, zeros, array, sqrt
from esquemas_matematicos import euler, crank_nicholson, rk4, euler_inverso, cauchy

# 1. Definimos el movimiento de Kepler
def F_kepler(U, t=None):
    x, y, vx, vy = U

    r = sqrt(x**2 + y**2)

    ax = -x / r**3
    ay = -y / r**3

    return array([vx, vy, ax, ay])


# 2. Configuración de parámetros
U0 = array([1.0, 0.0, 0.0, 1])
At = 0.01
N = 10000

# 3. Llamadas a los métodos
t = linspace(0, N * At, N + 1)
U_e = cauchy(F_kepler, U0, t,  euler)
U_inv = cauchy(F_kepler, U0, t, euler_inverso)
U_cn = cauchy(F_kepler, U0, t, crank_nicholson)
U_rk = cauchy(F_kepler, U0, t, rk4)

# 4. Representación gráfica de las órbitas


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for ax, U, nombre in zip(axes.flat, [U_e, U_inv, U_cn, U_rk],
                         ['Euler', 'Euler inverso', 'Crank-Nicolson', 'RK4']):
    ax.plot(U[:, 0], U[:, 1])
    ax.plot(0, 0, 'yo')
    ax.set_title(nombre)
    ax.axis('equal')
    ax.grid(True)
plt.tight_layout()
plt.show()