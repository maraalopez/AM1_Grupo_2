"""
calculo.py  —  ORQUESTADOR (puente entre capas)
Junta la física (fisica.py) con la capa matemática (esquemas_temporales.py,
analisis_numerico.py). No dibuja nada.
"""
import numpy as np

import fisica as f
import esquemas_temporales as et
import analisis_numerico as an
from ajustes import ESQUEMAS, TF, T_CONV, N_CONV, w, estado


def leer_parametros() -> tuple[list[float], np.ndarray, float, str]:
    """Lee la GUI y construye la condición inicial: giro Ω sobre el eje elegido + ε en los otros."""
    I = [w['I1'].val, w['I2'].val, w['I3'].val]
    puesto = ['eje menor', 'eje intermedio', 'eje mayor'].index(w['eje'].value_selected)
    eje = int(np.argsort(I)[puesto])
    w0 = np.full(3, w['eps'].val)
    w0[eje] = w['Omega'].val
    return I, w0, w['dt'].val, w['esquema'].value_selected


def equilibrios(I: list[float], Omega: float) -> list[np.ndarray]:
    """Rotación pura de velocidad Ω sobre el eje menor, intermedio y mayor (en ese orden)."""
    lista = []
    for eje in np.argsort(I):
        W = np.zeros(3)
        W[eje] = Omega
        lista.append(W)
    return lista


def resolver() -> None:
    """Simula con el esquema elegido, estima el error y analiza la estabilidad."""
    I, w0, dt, nombre = leer_parametros()
    scheme, q, R = ESQUEMAS[nombre]
    t = et.partition(0, TF, int(round(TF / dt)))
    dt = t[1] - t[0]                                  # paso real de la malla

    def F(omega: np.ndarray, t_val: float) -> np.ndarray:
        return f.euler_equations(omega, I, t_val)

    # 1. Error de Richardson a lo largo del tiempo
    with np.errstate(all='ignore'):
        U, E = an.richardson_error_temporal(F, w0, t, scheme, q)
    valido = np.isfinite(U).all(axis=1) & np.isfinite(E).all(axis=1)
    if not valido.all():                              # el esquema ha divergido
        n = max(int(valido.argmin()), 2)
        t, U, E = t[:n], U[:n], E[:n]

    # 2. Estabilidad lineal en los tres equilibrios: z = λ·Δt
    eq = []
    for W in equilibrios(I, w['Omega'].val):
        maxR, z = an.amplificacion_maxima(R, f.jacobian(W, I), dt)
        eq.append((maxR, z))

    # 3. Nube de z = λ·Δt a lo largo de la trayectoria (el Jacobiano cambia con ω)
    idx = np.linspace(0, len(U) - 1, min(150, len(U))).astype(int)
    z_tray = np.concatenate([np.linalg.eigvals(f.jacobian(U[k], I)) * dt for k in idx])

    estado.update(I=I, w0=w0, dt=dt, nombre=nombre, t=t, U=U, E=E,
                  eq=eq, z_tray=z_tray, divergio=not valido.all())


def calcular_convergencia() -> None:
    """Ratio de convergencia de los cuatro esquemas con la condición inicial actual."""
    I, w0, _, _ = leer_parametros()

    def F(omega: np.ndarray, t_val: float) -> np.ndarray:
        return f.euler_equations(omega, I, t_val)

    resultados = {}
    with np.errstate(all='ignore'):
        for nombre, (scheme, q_teo, _) in ESQUEMAS.items():
            q, d = an.ratio_convergencia(F, w0, T_CONV, N_CONV, scheme)
            resultados[nombre] = (q, d, q_teo)
    estado['conv'] = resultados
    estado['conv_param'] = (tuple(I), tuple(w0))
