import matplotlib.pyplot as plt
from numpy import zeros, array
from esquemas_temporales import euler, crank_nicholson, rk4, euler_inverso
# =====================================================================
# Ejemplo de uso de las funciones de integración numérica definidas en esquemas_temporales.py
# =====================================================================

# 1. Definimos la función del sistema dU/dt = F(U)
def F_oscilador(U):
    return array([U[1], -U[0]])

# 2. Configuración de parámetros
U0 = array([1.0, 0.0])  # Condición inicial
At = 0.1                # Paso de tiempo
N = 1000                # Pasos totales

# 3. Llamadas limpias a los métodos
U_e = euler(F_oscilador, U0, At, N)
U_inv = euler_inverso(F_oscilador, U0, At, N)
U_cn = crank_nicholson(F_oscilador, U0, At, N)
U_rk = rk4(F_oscilador, U0, At, N)

# 4. Representación gráfica
plt.figure(figsize=(7, 7))
plt.plot(U_e[:, 0], U_e[:, 1], 'r--', label='Euler')
plt.plot(U_inv[:, 0], U_inv[:, 1], 'k', label='Euler Inverso')
plt.plot(U_cn[:, 0], U_cn[:, 1], 'g-', label='Crank-Nicolson')
plt.plot(U_rk[:, 0], U_rk[:, 1], 'b:', label='RK4')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.title("Comparación de Métodos de Integración Numérica")
plt.show()