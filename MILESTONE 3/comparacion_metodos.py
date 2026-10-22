import numpy as np
import matplotlib.pyplot as plt
from esquemas_matematicos import euler, euler_inverso, crank_nicholson, rk4
from kepler import F_kepler
from richardson_error import richardson_error
from ratio_convergencia import ratio_convergencia

# Problema de Cauchy: órbita de Kepler
F = F_kepler
U0 = np.array([1.0, 0.0, 0.0, 1.0])
T = 10.0

# Métodos y órdenes teóricos
methods = [
    ("Euler", euler, 1),
    ("Euler inverso", euler_inverso, 1),
    ("Crank-Nicolson", crank_nicholson, 2),
    ("RK4", rk4, 4)
]

# Número de pasos de la malla gruesa
# (con N pequeño, Euler y Euler inverso aún no están en la zona asintótica)
N_values = [1000, 2000, 4000, 8000, 16000]

# ---------- Apartados 1 y 2: error por Richardson ----------
errors = {}
for name, scheme, p in methods:
    errors[name] = [richardson_error(F, U0, T, N, scheme, p) for N in N_values]

plt.figure(figsize=(8, 6))
for name, scheme, p in methods:
    plt.loglog(N_values, errors[name], "o-", label=name)
plt.xlabel("Número de pasos N")
plt.ylabel("Error estimado en t = T")
plt.title("Error numérico (extrapolación de Richardson)")
plt.grid(True, which="both")
plt.legend()

# ---------- Apartados 3 y 4: ratio de convergencia ----------
plt.figure(figsize=(8, 6))
for name, scheme, p in methods:
    q, d = ratio_convergencia(F, U0, T, N_values, scheme)
    print(f"{name:15s} q estimado = {q:.3f}  (teórico {p})")
    plt.plot(np.log10(N_values), np.log10(d), "o-", label=f"{name}: q = {q:.2f}")
plt.xlabel("log10(N)")
plt.ylabel("log10|U_2N(T) - U_N(T)|")
plt.title("Ratio de convergencia")
plt.grid(True)
plt.legend()

plt.show()