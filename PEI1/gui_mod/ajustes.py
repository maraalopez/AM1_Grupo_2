"""
ajustes.py
Colores, constantes y los tres diccionarios compartidos por todos los módulos:

    ax      los paneles (ejes) de la figura
    w       los widgets (sliders, botones...)
    estado  resultados de la simulación y elementos que se animan

Como son diccionarios, cuando un módulo escribe en ellos el resto lo ve.
"""
import esquemas_temporales as et

FONDO, PANEL, TEXTO, GRIS = '#0b1020', '#131a2e', '#e2e8f0', '#7c8aa5'
ROJO, AZUL, VERDE, AMARILLO, ROSA, CIAN = (
    '#ff5a5f', '#4aa8ff', '#a3e635', '#fbbf24', '#f472b6', '#4dd0e1')

ESQUEMAS = {'Euler': et.euler, 'RK4': et.rk4, 'Crank-Nicolson': et.crank_nicholson}

TF = 20.0               # duración de la simulación [s]
SIM_POR_FRAME = 0.1     # segundos simulados que avanza la animación en cada fotograma
NMESH = 25              # resolución de las superficies de Poinsot (más baja = más rápido)

# Tres diccionarios compartidos por todas las funciones:
ax = {}        # los ejes (paneles) de la figura
w = {}         # los widgets (sliders, botones...)
estado = {'pausa': False}   # resultados de la simulación y elementos animados
