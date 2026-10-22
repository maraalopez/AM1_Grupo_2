import numpy as np
from esquemas_matematicos import cauchy
 
 
def ratio_convergencia(F, U0, T, N_values, scheme):
    """
    Calcula el ratio de convergencia q de un esquema temporal.
 
    Para cada N se resuelve con N y con 2N pasos y se mide
        d(N) = |U_2N(T) - U_N(T)|
    Como el error global ~ C / N^q, en escala logarítmica:
        log10(d) = cte - q * log10(N)
    así que q es menos la pendiente de la recta log10(d) vs log10(N).
 
    No necesita conocer el orden teórico del esquema: lo obtiene.
 
    F: función del problema de Cauchy
    U0: condición inicial
    T: tiempo final
    N_values: lista de números de pasos
    scheme: método numérico
 
    Devuelve (q, d), con d la lista de diferencias para cada N.
    """
    d = []
    for N in N_values:
        U_N = cauchy(F, U0, np.linspace(0, T, N + 1), scheme)
        U_2N = cauchy(F, U0, np.linspace(0, T, 2 * N + 1), scheme)
        d.append(np.linalg.norm(U_2N[-1] - U_N[-1]))
 
    logN = np.log10(np.array(N_values, dtype=float))
    logd = np.log10(np.array(d))
 
    # Se descartan los puntos dominados por el error de redondeo (d < 1e-11)
    validos = logd > -11
    # Si quedan suficientes puntos, se descarta también el primero
    # (malla más gruesa, aún fuera de la zona asintótica)
    if validos.sum() >= 3:
        validos[np.argmax(validos)] = False
 
    pendiente, _ = np.polyfit(logN[validos], logd[validos], 1)
 
    return -pendiente, d
 