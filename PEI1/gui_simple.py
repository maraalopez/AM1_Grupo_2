# gui_simple.py
# ---------------------------------------------------------------------------
#  Interfaz del teorema del eje intermedio  (versión sencilla: solo funciones)
#
#  Cómo está organizado el archivo (de arriba a abajo):
#    1. Configuración        colores y constantes
#    2. Cálculo              resolver()  y  orientacion()
#    3. Modelo del satélite  caja()  y  SATELITE
#    4. Dibujo de paneles    dibujar_fases / dibujar_actitud / dibujar_derivas
#    5. Animación            animar()
#    6. Controles y arranque crear_controles / al_cambiar / lanzar_gui
#
#  Idea general:
#    - Cuando el usuario mueve un control  ->  al_cambiar():
#          resolver()  ->  redibuja los 3 paneles.
#    - Una animación (cada 40 ms) llama a animar(), que solo MUEVE
#      los elementos ya dibujados (el punto, el satélite, el cursor...).
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, RadioButtons, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fisica as f
import esquemas_temporales as et

# ============================ 1. CONFIGURACIÓN =============================
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


# ============================== 2. CÁLCULO =================================
def resolver():
    """Lee los sliders, integra ω(t) y guarda todo en `estado`."""
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
                  paso=max(1, round(SIM_POR_FRAME / dt)))  # saltos entre fotogramas


def rotacion(v):
    """Matriz de rotación de ángulo |v| alrededor del eje v (fórmula de Rodrigues)."""
    ang = np.linalg.norm(v)
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def orientacion(t, U):
    """
    Orientación del satélite en cada instante, a partir de ω(t).
    R[k] es una matriz 3x3 que pasa de ejes del cuerpo a ejes inerciales:
    en cada paso se compone la rotación ω·Δt con la orientación anterior.
    """
    R = [np.eye(3)]
    for k in range(len(t) - 1):
        omega_medio = 0.5 * (U[k] + U[k + 1])
        R.append(R[-1] @ rotacion(omega_medio * (t[k + 1] - t[k])))
    return np.array(R)


# ========================= 3. MODELO DEL SATÉLITE ==========================
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


# ========================= 4. DIBUJO DE LOS PANELES ========================
def estilo_3d(a, titulo):
    a.set_facecolor(PANEL)
    a.set_title(titulo, loc='left', color='white', fontweight='bold', fontsize=11)
    for eje in (a.xaxis, a.yaxis, a.zaxis):
        eje.set_pane_color((1, 1, 1, 0))
    a.tick_params(labelsize=7, colors=GRIS)


def estilo_2d(a, titulo, color):
    a.set_facecolor(PANEL)
    a.set_title(titulo, loc='left', color=color, fontweight='bold', fontsize=11)
    a.grid(True, color='#26324a', linestyle=':')
    a.tick_params(labelsize=8, colors=GRIS)
    for borde in a.spines.values():
        borde.set_color('#26324a')


def dibujar_fases():
    """Espacio de fases: superficies de Poinsot, separatriz y trayectoria."""
    a = ax['fases']
    elev, azim = a.elev, a.azim            # guardamos la vista para no perderla al borrar
    a.clear()
    estilo_3d(a, 'ESPACIO DE FASES  (ω₁, ω₂, ω₃)')
    I, w0, U = estado['I'], estado['w0'], estado['U']

    T0 = f.invariant_energy(w0, I)
    L0 = f.invariant_angular_momentum(w0, I)
    malla_T, malla_L = f.superficies_poinsot(I, T0, np.sqrt(L0), num_puntos=NMESH)
    a.plot_surface(*malla_T, color=ROJO, alpha=0.15, linewidth=0, shade=False)
    a.plot_wireframe(*malla_L, color=AZUL, alpha=0.35, linewidth=0.5, rstride=2, cstride=2)

    sep = np.asarray(f.calcular_separatriz(I, np.linalg.norm(w0)))
    a.plot(sep[:, 0], sep[:, 1], sep[:, 2], color=VERDE, lw=2.5, ls='--')
    a.plot(U[:, 0], U[:, 1], U[:, 2], color='white', alpha=0.3, lw=1)   # trayectoria completa

    # Estos dos se crean vacíos: animar() los irá rellenando
    estado['recorrido'], = a.plot([], [], [], color='white', lw=2)
    estado['punto'], = a.plot([], [], [], 'o', color=AMARILLO, ms=8)

    lim = 1.15 * max(np.abs(m).max() for m in (*malla_T, *malla_L))
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
    a.set_xlabel('ω₁'); a.set_ylabel('ω₂'); a.set_zlabel('ω₃')
    a.set_box_aspect((1, 1, 1), zoom=1.15)
    a.view_init(elev, azim)
    a.legend(handles=[Line2D([0], [0], color=ROJO, lw=5, alpha=0.5, label='Energía (T)'),
                      Line2D([0], [0], color=AZUL, lw=2, label='Momento (L)'),
                      Line2D([0], [0], color=VERDE, lw=2, ls='--', label='Separatriz'),
                      Line2D([0], [0], color='white', lw=2, label='ω(t)')],
             loc='upper left', fontsize=7.5, frameon=False)


def dibujar_actitud():
    """Panel del satélite: se crean los elementos; animar() los mueve."""
    a = ax['actitud']
    a.clear()
    a.set_axis_off()
    I, w0, U = estado['I'], estado['w0'], estado['U']

    # Título que avisa de si el giro inicial es estable o no
    eje = int(np.argmax(np.abs(w0)))                 # eje en torno al que gira al principio
    puesto = list(np.argsort(I)).index(eje)          # 0 = menor inercia, 1 = intermedio, 2 = mayor
    texto, color = [('estable (eje de menor inercia)', VERDE),
                    ('INESTABLE (eje intermedio)', ROJO),
                    ('estable (eje de mayor inercia)', VERDE)][puesto]
    a.set_title('ACTITUD DEL SATÉLITE', loc='left', color='white',
                fontweight='bold', fontsize=11)
    a.text2D(0.02, 0.95, f'Giro inicial en el eje {eje + 1}: {texto}', transform=a.transAxes,
             color=color, fontweight='bold', fontsize=9)

    estado['sat'] = Poly3DCollection(SATELITE, facecolors=COLORES_SAT,
                                     edgecolors=FONDO, linewidths=0.5, alpha=0.95)
    a.add_collection3d(estado['sat'])

    # L es constante en el espacio inercial: se dibuja fijo
    L = np.array(I) * U[0]
    L = 1.4 * L / np.linalg.norm(L)
    a.plot([0, L[0]], [0, L[1]], [0, L[2]], color=AMARILLO, lw=2.5, ls='--', label='L (fijo)')

    estado['flecha_w'], = a.plot([], [], [], color='white', lw=2.5, label='ω')
    estado['traza'], = a.plot([], [], [], color=ROSA, lw=1.5, label=f'eje {eje + 1} (traza)')
    estado['punta'] = estado['R'][:, :, eje] * 1.2      # posición de la punta de ese eje
    estado['escala_w'] = 1.2 / max(np.linalg.norm(U, axis=1).max(), 1e-12)
    estado['reloj'] = a.text2D(0.03, 0.04, '', transform=a.transAxes, color=TEXTO)

    a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_zlim(-1.5, 1.5)
    a.set_box_aspect((1, 1, 1), zoom=1.5)
    a.legend(loc='upper right', fontsize=8, frameon=False)


def dibujar_derivas():
    """Errores ΔT(t) y ΔL(t) en escala logarítmica."""
    t = estado['t']
    for clave, dato, titulo, color in (('T', estado['dT'], 'DERIVA DE ENERGÍA  ΔT(t)', ROJO),
                                       ('L', estado['dL'], 'DERIVA DE MOMENTO  ΔL(t)', AZUL)):
        a = ax[clave]
        a.clear()
        estilo_2d(a, titulo, color)
        a.semilogy(t, np.maximum(dato, 1e-17), color=color, lw=2)   # 1e-17 evita log(0)
        a.fill_between(t, 1e-17, np.maximum(dato, 1e-17), color=color, alpha=0.15)
        a.set_ylabel('error absoluto', color=GRIS)
        estado['cursor_' + clave] = a.axvline(0, color='white', lw=1, alpha=0.7)
    ax['L'].set_xlabel('Tiempo [s]', color=GRIS)
    ax['T'].set_xlim(0, TF)
    ax['L'].set_xlim(0, TF)


# ============================== 5. ANIMACIÓN ===============================
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


# ======================== 6. CONTROLES Y ARRANQUE ==========================
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


def lanzar_gui():
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
