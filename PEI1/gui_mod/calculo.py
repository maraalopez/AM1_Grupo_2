"""
calculo.py
Parte numérica: resuelve ω(t) con el integrador elegido y calcula
la orientación del satélite. No dibuja nada.
"""
import numpy as np

import fisica as f
import esquemas_temporales as et
from ajustes import ESQUEMAS, TF, SIM_POR_FRAME, w, estado


def resolver():
    """Lee los sliders, integra ω(t) y guarda todo en `estado`."""
    I = [w['I1'].val, w['I2'].val, w['I3'].val]
    w0 = np.array([w['w1'].val, w['w2'].val, w['w3'].val])
    dt = w['dt'].val
    esquema = ESQUEMAS[w['esquema'].value_selected]

    t = et.partition(0, TF, int(TF / dt))

    def F(omega, t_val):                      # las ecuaciones de Euler
        return f.euler_equations(omega, I, t_val)

    with np.errstate(all='ignore'):           # silencia avisos si un esquema diverge
        U = np.asarray(et.cauchy_problem(F=F, U0=w0, t=t, scheme=esquema))
        dT = np.asarray(et.calcular_deriva(U, f.invariant_energy, I))
        dL = np.asarray(et.calcular_deriva(U, f.invariant_angular_momentum, I))

    # Si el esquema "explota" (p. ej. Euler con Δt grande) nos quedamos con la parte válida
    valido = np.isfinite(U).all(axis=1)
    if not valido.all():
        n = max(int(valido.argmin()), 2)
        t, U, dT, dL = t[:n], U[:n], dT[:n], dL[:n]

    estado.update(I=I, w0=w0, t=t, U=U, dT=dT, dL=dL,
                  R=orientacion(t, U),
                  k=0,                                   # fotograma actual
                  paso=max(1, round(SIM_POR_FRAME / dt)))  # saltos entre fotogramas


def rotacion(v):
    """Matriz de rotación de ángulo |v| alrededor del eje v (fórmula de Rodrigues)."""
    ang = np.linalg.norm(v)
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def orientacion(t, U):
    """
    Orientación del satélite en cada instante, a partir de ω(t).
    R[k] es una matriz 3x3 que pasa de ejes del cuerpo a ejes inerciales:
    en cada paso se compone la rotación ω·Δt con la orientación anterior.
    """
    R = [np.eye(3)]
    for k in range(len(t) - 1):
        omega_medio = 0.5 * (U[k] + U[k + 1])
        R.append(R[-1] @ rotacion(omega_medio * (t[k + 1] - t[k])))
    return np.array(R)
