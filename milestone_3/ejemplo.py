import numpy as np
import matplotlib.pyplot as plt
import richardson as rich
import os
import sys

# Añade la carpeta raíz (AM1_Grupo_2) al path donde Python busca módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import milestone_2.esquemas_temporales as esq


# Solución exacta para la comparación
def exact_sol(t, U0):
    x0, v0 = U0[0], U0[1]
    x = x0 * np.cos(t) + v0 * np.sin(t)
    v = -x0 * np.sin(t) + v0 * np.cos(t)
    return np.array([x, v])

# Condición inicial: x(0) = 1, v(0) = 0
U0 = np.array([1.0, 0.0])
t_span = (0.0, 2.0 * np.pi)  # 1 periodo completo
N_list = [20, 40, 80, 160, 320, 640]

schemes = {
    'Euler Explícito': (esq.euler, 1),
    'Crank-Nicolson': (esq.crank_nicholson, 2),
    'Runge-Kutta 4': (esq.rk4, 4)
}

# --- A) Tasa de convergencia ---
plt.figure(figsize=(8, 5))
print(f"{'Esquema':<20} | {'Tasa Calculada (p)':<20}")
print("-" * 45)

for name, (scheme, p_teorico) in schemes.items():
    h_vals, errors, p_calc = rich.evaluate_convergence_rate(esq.oscilador, U0, t_span, scheme, N_list)
    print(f"{name:<20} | {p_calc:.4f}")
    plt.loglog(h_vals, errors, 'o-', label=f"{name} ($p \\approx {p_calc:.2f}$)")

plt.xlabel('Tamaño de paso $h = \\Delta t$')
plt.ylabel('Error global en $t_f = 2\\pi$')
plt.title('Tasa de convergencia — Oscilador Armónico')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# --- B) Ejemplo de Error por Richardson (entre h1=0.1 y h2=0.05) ---
t1 = np.linspace(0.0, 2.0 * np.pi, 63)   # h1 ≈ 0.1
t2 = np.linspace(0.0, 2.0 * np.pi, 126)  # h2 ≈ 0.05

E_richardson = rich.richardson_error(esq.oscilador, U0, t1, t2, esq.rk4, p=4)
print(f"\nError máximo estimado por Richardson (RK4): {np.max(np.abs(E_richardson)):.6e}")