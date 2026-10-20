"""
panel_control.py
Sliders, botones y selector de integrador, y la función al_cambiar(),
que se ejecuta cada vez que el usuario toca un control.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from ajustes import w, estado, ESQUEMAS, FONDO, TEXTO, AMARILLO, CIAN
from calculo import resolver
from dibujo import dibujar_fases, dibujar_actitud, dibujar_derivas


def al_cambiar(_=None):
    """Se ejecuta cada vez que el usuario toca un control."""
    if np.linalg.norm([w['w1'].val, w['w2'].val, w['w3'].val]) < 1e-9:
        return                                   # con ω(0) = 0 no hay movimiento
    resolver()
    dibujar_fases()
    dibujar_actitud()
    dibujar_derivas()
    plt.gcf().canvas.draw_idle()


def crear_slider(rect, nombre, vmin, vmax, inicial, color, formato='%.2f'):
    s = Slider(plt.axes(rect, facecolor='#26324a'), nombre, vmin, vmax,
               valinit=inicial, valfmt=formato, color=color,
               track_color='#2b3652')
    s.label.set_color(TEXTO)
    s.valtext.set_color(color)
    s.on_changed(al_cambiar)
    return s


def preset(puesto):
    """Pone ω(0) casi alineada con un eje: 0 = menor inercia, 1 = intermedio, 2 = mayor."""
    eje = int(np.argsort([w['I1'].val, w['I2'].val, w['I3'].val])[puesto])
    valores = [0.01, 0.01, 0.01]
    valores[eje] = 1.0
    for nombre, v in zip(('w1', 'w2', 'w3'), valores):
        w[nombre].eventson = False               # evita recalcular tres veces
        w[nombre].set_val(v)
        w[nombre].eventson = True
    al_cambiar()


def pausar(_):
    estado['pausa'] = not estado['pausa']
    w['b_pausa'].label.set_text('▶ Reanudar' if estado['pausa'] else '|| Pausa')


def crear_boton(rect, texto):
    b = Button(plt.axes(rect), texto, color='#26324a', hovercolor='#34456a')
    b.label.set_color(TEXTO)
    b.label.set_fontsize(8.5)
    return b


def crear_controles():
    # Títulos de cada grupo
    for x, titulo, color in ((0.06, 'MOMENTOS DE INERCIA', AMARILLO),
                             (0.38, 'VELOCIDAD ANGULAR INICIAL', CIAN),
                             (0.70, 'INTEGRADOR NUMÉRICO', '#b388ff')):
        plt.gcf().text(x, 0.285, titulo, color=color, fontweight='bold', fontsize=10)

    # Sliders: columna 1 (inercias) y columna 2 (ω inicial)
    w['I1'] = crear_slider([0.09, 0.22, 0.22, 0.03], 'I₁', 0.1, 5.0, 1.0, AMARILLO)
    w['I2'] = crear_slider([0.09, 0.17, 0.22, 0.03], 'I₂', 0.1, 5.0, 2.0, AMARILLO)
    w['I3'] = crear_slider([0.09, 0.12, 0.22, 0.03], 'I₃', 0.1, 5.0, 3.0, AMARILLO)
    w['w1'] = crear_slider([0.41, 0.22, 0.22, 0.03], 'ω₁(0)', -5.0, 5.0, 0.01, CIAN)
    w['w2'] = crear_slider([0.41, 0.17, 0.22, 0.03], 'ω₂(0)', -5.0, 5.0, 1.0, CIAN)
    w['w3'] = crear_slider([0.41, 0.12, 0.22, 0.03], 'ω₃(0)', -5.0, 5.0, 0.01, CIAN)

    # Columna 3: paso temporal y elección del integrador
    w['dt'] = crear_slider([0.73, 0.22, 0.20, 0.03], 'Δt', 0.005, 0.2, 0.05, '#b388ff', '%.3f')
    ax_radio = plt.axes([0.72, 0.05, 0.13, 0.14], facecolor=FONDO)
    for borde in ax_radio.spines.values():
        borde.set_visible(False)
    w['esquema'] = RadioButtons(ax_radio, list(ESQUEMAS), active=1, activecolor='#b388ff')
    for etiqueta in w['esquema'].labels:
        etiqueta.set_color(TEXTO)
    w['esquema'].on_clicked(al_cambiar)

    # Botones: pausa y atajos de condiciones iniciales
    w['b_pausa'] = crear_boton([0.09, 0.04, 0.10, 0.05], '|| Pausa')
    w['b_pausa'].on_clicked(pausar)
    w['b_min'] = crear_boton([0.41, 0.04, 0.07, 0.05], 'eje menor')
    w['b_mid'] = crear_boton([0.49, 0.04, 0.07, 0.05], 'eje interm.')
    w['b_max'] = crear_boton([0.57, 0.04, 0.07, 0.05], 'eje mayor')
    w['b_min'].on_clicked(lambda _: preset(0))
    w['b_mid'].on_clicked(lambda _: preset(1))
    w['b_max'].on_clicked(lambda _: preset(2))
