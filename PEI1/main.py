import esquemas_temporales as et
import fisica as f
import matplotlib.pyplot as plt
from gui_simple import lanzar_gui

if __name__ == "__main__":
    lanzar_gui()

'''# Definición de la función de física cambiando el orden a (omega, t, I)
def f_sistema(omega, t):
    return f.euler_equations(omega, I, t)


#Definir los valores constantes de la simulación
I = [1, 2, 3] #Momento de inercia
t = et.partition(0, 10, 100) #Intervalo de tiempo
U0 = et.array([0.01, 1, 0.01]) #Condiciones iniciales

#Resolver el problema de Cauchy usando los diferentes métodos de integración temporal
U_euler = et.cauchy_problem( F = f_sistema, U0 = U0, t = t, scheme = et.euler)
U_rk4 = et.cauchy_problem( F = f_sistema, U0 = U0, t = t, scheme = et.rk4)
U_cn = et.cauchy_problem( F = f_sistema, U0 = U0, t = t, scheme = et.crank_nicholson)

plt.plot(U_cn[:, 0], U_cn[:, 1])
plt.title("Método de Crank-Nicolson")
plt.axis('equal')
plt.grid(True)
plt.show()

plt.plot(U_euler[:, 0], U_euler[:, 1])
plt.title("Método de Euler")
plt.axis('equal')
plt.grid(True)
plt.show()

plt.plot(U_rk4[:, 0], U_rk4[:, 1])
plt.title("Método de RK4")
plt.axis('equal')
plt.grid(True)
plt.show()
'''


