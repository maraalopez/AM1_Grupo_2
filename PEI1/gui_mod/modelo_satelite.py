"""
modelo_satelite.py
Geometría del satélite (cuerpo + paneles solares) hecha con cajas.
"""
import numpy as np

from ajustes import ROJO


def caja(centro, semilados):
    """Las 6 caras (4 vértices cada una) de un ortoedro."""
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = vertices * semilados + centro
    caras = [[0, 1, 3, 2], [4, 5, 7, 6],      # caras x = -1, +1
             [0, 1, 5, 4], [2, 3, 7, 6],      # caras y = -1, +1
             [0, 2, 6, 4], [1, 3, 7, 5]]      # caras z = -1, +1
    return vertices[caras]


# Cuerpo central + dos paneles solares a lo largo del eje 2 (en ejes del cuerpo)
SATELITE = np.concatenate([
    caja([0, 0, 0], [0.40, 0.30, 0.30]),
    caja([0, -0.80, 0], [0.30, 0.50, 0.015]),
    caja([0, 0.80, 0], [0.30, 0.50, 0.015]),
])
# Colores: cara x = rojo, y = verde, z = azul (como los ejes 1, 2, 3); paneles azul oscuro
COLORES_SAT = [ROJO] * 2 + ['#69db7c'] * 2 + ['#4dabf7'] * 2 + ['#1e40af'] * 12
