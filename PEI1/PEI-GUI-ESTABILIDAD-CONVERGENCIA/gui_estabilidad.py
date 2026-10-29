# gui_estabilidad.py
# ---------------------------------------------------------------------------
#  Estabilidad numérica y convergencia de los esquemas temporales aplicados
#  al teorema del eje intermedio. ARCHIVO PRINCIPAL: ejecuta este.
#
#  Módulos:
#    fisica.py              CAPA FÍSICA: ecuaciones de Euler, Jacobiano
#    esquemas_temporales.py CAPA MATEMÁTICA: esquemas y problema de Cauchy
#    analisis_numerico.py   CAPA MATEMÁTICA: R(z), Richardson, convergencia
#    calculo.py             orquestador: resolver(), calcular_convergencia()
#    ajustes.py             colores, constantes y diccionarios compartidos
#    dibujo.py              un dibujar_xxx() por panel
#    panel_control.py       sliders, selectores, botón y al_cambiar()
#
#  Flujo:
#    usuario mueve un control      -> al_cambiar(): resolver() y redibujar
#    pulsa «Calcular convergencia» -> calcular_convergencia() (tarda unos segundos)
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

from ajustes import ax, FONDO, PANEL, TEXTO
from panel_control import crear_controles, al_cambiar


def lanzar_gui() -> plt.Figure:
    plt.style.use('dark_background')
    plt.rcParams.update({'font.family': 'monospace', 'font.size': 9,
                         'figure.facecolor': FONDO, 'axes.facecolor': PANEL,
                         'text.color': TEXTO, 'axes.labelcolor': TEXTO,
                         'grid.color': '#2b3652', 'axes.edgecolor': '#2b3652'})

    fig = plt.figure(figsize=(16, 9), facecolor=FONDO)
    fig.text(0.04, 0.955, 'ESTABILIDAD NUMÉRICA Y CONVERGENCIA · TEOREMA DEL EJE INTERMEDIO',
             fontsize=17, fontweight='bold', color='white')

    # Fila de arriba: región completa | zoom | convergencia
    # Fila de abajo:  error de Richardson (ancho) | diagnóstico
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.91, bottom=0.35,
                          wspace=0.18, hspace=0.42, height_ratios=[2.0, 1])
    ax['region'] = fig.add_subplot(gs[0, 0])
    ax['zoom'] = fig.add_subplot(gs[0, 1])
    ax['conv'] = fig.add_subplot(gs[0, 2])
    ax['richardson'] = fig.add_subplot(gs[1, 0:2])
    ax['diag'] = fig.add_subplot(gs[1, 2])

    crear_controles()
    al_cambiar()                                    # primer cálculo y dibujo
    plt.show()
    return fig


if __name__ == '__main__':
    lanzar_gui()
