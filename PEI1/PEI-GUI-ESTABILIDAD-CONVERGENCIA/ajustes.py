"""
ajustes.py
Colores, constantes y los tres diccionarios compartidos por todos los módulos:

    ax      los paneles (ejes) de la figura
    w       los widgets (sliders, botones...)
    estado  resultados de los cálculos

Como son diccionarios, cuando un módulo escribe en ellos el resto lo ve.
"""
import esquemas_temporales as et
import analisis_numerico as an

FONDO, PANEL, TEXTO, GRIS = '#0b1020', '#131a2e', '#e2e8f0', '#7c8aa5'
ROJO, AZUL, VERDE, AMARILLO, ROSA, CIAN, MORADO = (
    '#ff5a5f', '#4aa8ff', '#a3e635', '#fbbf24', '#f472b6', '#4dd0e1', '#b388ff')

# nombre -> (esquema, orden teórico q, factor de amplificación R(z))
ESQUEMAS: dict = {
    'Euler':          (et.euler,           1, an.R_euler),
    'Euler inverso':  (et.euler_inverso,   1, an.R_euler_inverso),
    'Crank-Nicolson': (et.crank_nicholson, 2, an.R_crank_nicolson),
    'RK4':            (et.rk4,             4, an.R_rk4),
}
COLOR_ESQUEMA: dict = {'Euler': ROJO, 'Euler inverso': AZUL,
                       'Crank-Nicolson': VERDE, 'RK4': MORADO}

# Colores de los tres equilibrios: 0 = eje menor, 1 = intermedio, 2 = mayor
COLOR_EJE: list = [CIAN, AMARILLO, ROSA]
NOMBRE_EJE: list = ['eje menor', 'eje intermedio', 'eje mayor']

TF: float = 20.0                                # duración de la simulación [s]
T_CONV: float = 20.0                            # tiempo final del estudio de convergencia
N_CONV: list = [200, 400, 800, 1600, 3200]      # pasos de las mallas de convergencia
LIM_REGION: tuple = ((-4.0, 2.0), (-3.5, 3.5))  # ventana del plano z en el panel completo

ax: dict = {}                 # los ejes (paneles) de la figura
w: dict = {}                  # los widgets
estado: dict = {}             # resultados de los cálculos
