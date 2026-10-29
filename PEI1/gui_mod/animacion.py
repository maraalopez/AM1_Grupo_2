"""
animacion.py
animar(): se llama cada 40 ms y mueve los elementos ya dibujados
(punto del espacio de fases, satélite, cursores de tiempo).
"""
from ajustes import estado
from modelo_satelite import SATELITE


def animar(_):
    """Se llama cada 40 ms. Avanza un fotograma y mueve los elementos."""
    if 'R' not in estado or estado['pausa']:
        return
    estado['k'] = (estado['k'] + estado['paso']) % len(estado['t'])    # vuelve a empezar al acabar
    k, t, U, R = estado['k'], estado['t'], estado['U'], estado['R']

    # espacio de fases: punto actual y tramo ya recorrido
    estado['punto'].set_data_3d([U[k, 0]], [U[k, 1]], [U[k, 2]])
    estado['recorrido'].set_data_3d(U[:k + 1, 0], U[:k + 1, 1], U[:k + 1, 2])

    # satélite: se giran sus vértices con la matriz de rotación del instante k
    estado['sat'].set_verts(list(SATELITE @ R[k].T))
    wk = R[k] @ U[k] * estado['escala_w']                  # ω expresada en ejes inerciales
    estado['flecha_w'].set_data_3d([0, wk[0]], [0, wk[1]], [0, wk[2]])
    p = estado['punta'][max(0, k - 300):k + 1]             # últimos 300 puntos de la traza
    estado['traza'].set_data_3d(p[:, 0], p[:, 1], p[:, 2])
    estado['reloj'].set_text(f't = {t[k]:5.2f} s')

    # cursores de tiempo en las gráficas de deriva
    estado['cursor_T'].set_xdata([t[k], t[k]])
    estado['cursor_L'].set_xdata([t[k], t[k]])
