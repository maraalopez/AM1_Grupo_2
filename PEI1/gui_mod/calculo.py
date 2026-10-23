"""
calculo.py
Parte numérica: resuelve ω(t) con el integrador elegido y calcula
la orientación del satélite. No dibuja nada.
"""
import numpy as np

import fisica as f
import esquemas_temporales as et
from ajustes import ESQUEMAS, TF, SIM_POR_FRAME, w, estado


"""def resolver():
    Lee los sliders, integra ω(t) y guarda todo en `estado`.
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
                  paso=max(1, round(SIM_POR_FRAME / dt)))  # saltos entre fotogramas"""


def resolver():
    """Lee la GUI, delega el cálculo al motor físico, y guarda el estado visual."""
    # 1. Extraer los valores crudos de los widgets interactivos
    I = [w['I1'].val, w['I2'].val, w['I3'].val]
    w0 = np.array([w['w1'].val, w['w2'].val, w['w3'].val])
    dt = w['dt'].val
    esquema = ESQUEMAS[w['esquema'].value_selected]
    
    # 2. Enviar los datos al orquestador físico
    t, U, dT, dL, R = f.ejecutar_simulacion(I, w0, dt, esquema, TF)
    
    # 3. Guardar los resultados en el diccionario de la vista
    estado.update(I=I, w0=w0, t=t, U=U, dT=dT, dL=dL, R=R,
                  k=0, paso=max(1, round(SIM_POR_FRAME / dt)))
