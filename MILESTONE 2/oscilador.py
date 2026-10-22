
import matplotlib.pyplot as plt
from numpy import zeros, array, linspace
from esquemas_matematicos import euler, crank_nicholson, rk4, euler_inverso, cauchy

# 1. Definimos el oscilador
def F_oscilador(U, t=None):         
    return array([U[1], -U[0]])
# 2. Configuración de parámetros
U0 = array([1.0, 0.0])  # Condición inicial
At = 0.1                # Paso de tiempo
N = 1000                # Pasos totales

# 3. Llamadas a los métodos
t = linspace(0, N * At, N + 1)

U_e = cauchy(F_oscilador, U0, t, euler)
U_inv = cauchy(F_oscilador, U0, t, euler_inverso)
U_cn = cauchy(F_oscilador, U0, t, crank_nicholson)
U_rk = cauchy(F_oscilador, U0, t, rk4)

# 4. Representación gráfica
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
resultados = [U_e, U_inv, U_cn, U_rk]
nombres = ['Euler', 'Euler inverso', 'Crank-Nicolson', 'RK4']
 
for ax, U, nombre in zip(axes.flat, resultados, nombres):
        ax.plot(U[:, 0], U[:, 1])
        ax.set_title(nombre)
        ax.set_xlabel('x')
        ax.set_ylabel('v')
        ax.axis('equal')
        ax.grid(True)
 
fig.suptitle('Oscilador armónico: comparación de métodos')
plt.tight_layout()
plt.show()