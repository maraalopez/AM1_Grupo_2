
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
# U_inv = cauchy(F_kepler, U0, t, euler_inverso)
# U_cn = cauchy(F_kepler, U0, t, crank_nicholson)
U_rk = cauchy(F_kepler, U0, t, rk4)

# 4. Representación gráfica de las órbitas
plt.figure(figsize=(7, 7))

plt.plot(U_e[:, 0], U_e[:, 1], 'r--', label='Euler')
# plt.plot(U_inv[:, 0], U_inv[:, 1], 'k', label='Euler inverso')
# plt.plot(U_cn[:, 0], U_cn[:, 1], 'g-', label='Crank-Nicolson')
plt.plot(U_rk[:, 0], U_rk[:, 1], 'b:', label='RK4')

# Posición del Sol
plt.plot(0, 0, 'yo', markersize=10, label='Sol')

plt.axis('equal')
plt.grid(True)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparación de métodos: movimiento de Kepler')
plt.show()
