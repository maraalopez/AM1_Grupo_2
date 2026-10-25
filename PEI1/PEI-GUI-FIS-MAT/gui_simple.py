# gui_simple.py
# ---------------------------------------------------------------------------
#  Interfaz del teorema del eje intermedio. ARCHIVO PRINCIPAL: ejecuta este.
#
#  Módulos (todos con funciones sencillas, sin clases):
#    ajustes.py          colores, constantes y diccionarios compartidos
#    fisica.py           CAPA FÍSICA: ecuaciones, invariantes, Poinsot
#    esquemas_temporales CAPA MATEMÁTICA: esquemas, Cauchy, deriva
#    calculo.py          orquestador: ejecutar_simulacion(), orientacion(), resolver()
#    modelo_satelite.py  geometría del satélite
#    dibujo.py           dibujar_fases / dibujar_actitud / dibujar_derivas
#    animacion.py        animar()
#    panel_control.py    sliders, botones y al_cambiar()
#
#  Flujo:
#    usuario mueve un control -> al_cambiar(): resolver() y redibujar paneles
#    cada 40 ms              -> animar(): mueve punto, satélite y cursores
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ajustes import ax, FONDO, PANEL, TEXTO
from animacion import animar
from panel_control import crear_controles, al_cambiar


def lanzar_gui() -> plt.Figure:
    plt.style.use('dark_background')
    plt.rcParams.update({'font.family': 'monospace', 'font.size': 9,
                         'figure.facecolor': FONDO, 'axes.facecolor': PANEL,
                         'text.color': TEXTO, 'axes.labelcolor': TEXTO,
                         'grid.color': '#2b3652', 'axes.edgecolor': '#2b3652'})

    fig = plt.figure(figsize=(16, 9), facecolor=FONDO)
    fig.text(0.04, 0.955, 'DINÁMICA DE ACTITUD SATELITAL · TEOREMA DEL EJE INTERMEDIO',
             fontsize=17, fontweight='bold', color='white')

    # Disposición de los 4 paneles: fases | actitud | (deriva T arriba, deriva L abajo)
    gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.88, bottom=0.37,
                          wspace=0.15, hspace=0.35, width_ratios=[1.2, 1.2, 1])
    ax['fases'] = fig.add_subplot(gs[:, 0], projection='3d')
    ax['actitud'] = fig.add_subplot(gs[:, 1], projection='3d')
    ax['T'] = fig.add_subplot(gs[0, 2])
    ax['L'] = fig.add_subplot(gs[1, 2])

    crear_controles()
    al_cambiar()                                    # primer cálculo y dibujo

    # La animación llama a animar() cada 40 ms. Hay que guardarla en una variable.
    fig.animacion = FuncAnimation(fig, animar, interval=40, cache_frame_data=False)
    plt.show()
    return fig


if __name__ == '__main__':
    lanzar_gui()
