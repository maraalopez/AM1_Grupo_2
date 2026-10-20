"""
dibujo.py
Funciones que dibujan cada panel de la figura:
    dibujar_fases()    espacio de fases 3D
    dibujar_actitud()  satélite 3D
    dibujar_derivas()  errores ΔT(t) y ΔL(t)
Se llaman cada vez que cambia un parámetro. Dejan en `estado` los elementos
vacíos que luego animar() irá moviendo.
"""
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import fisica as f
from ajustes import (ax, estado, FONDO, PANEL, TEXTO, GRIS, ROJO, AZUL, VERDE,
                     AMARILLO, ROSA, NMESH, TF)
from modelo_satelite import SATELITE, COLORES_SAT


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


"""def dibujar_fases():
    Espacio de fases: superficies de Poinsot, separatriz y trayectoria.
    a = ax['fases']
    elev, azim = a.elev, a.azim            # guardamos la vista para no perderla al borrar
    a.clear()
    estilo_3d(a, 'ESPACIO DE FASES  (ω₁, ω₂, ω₃)')
    I, w0, U = estado['I'], estado['w0'], estado['U']

    T0 = f.invariant_energy(w0, I)
    L0 = f.invariant_angular_momentum(w0, I)
    malla_T, malla_L = f.superficies_poinsot(I, T0, L0, num_puntos=NMESH)
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
             loc='upper left', fontsize=7.5, frameon=False)"""


def dibujar_fases():
    a = ax['fases']
    elev, azim = a.elev, a.azim
    a.clear()
    estilo_3d(a, 'ESPACIO DE FASES  (ω₁, ω₂, ω₃)')
    I, w0, U = estado['I'], estado['w0'], estado['U']

    T0 = f.invariant_energy(w0, I)
    L0_cuadrado = f.invariant_angular_momentum(w0, I)
    
    # Pasamos el módulo L (una sola raíz cuadrada)
    L0_modulo = np.sqrt(L0_cuadrado)
    malla_T, malla_L = f.superficies_poinsot(I, T0, L0_modulo, num_puntos=NMESH)

    # Renders de los elipsoides
    a.plot_surface(*malla_T, color=ROJO, alpha=0.2, linewidth=0, shade=True)
    a.plot_wireframe(*malla_L, color=AZUL, alpha=0.35, linewidth=0.6, rstride=2, cstride=2)

    # Dibujamos las dos estructuras de la separatriz (el "8" en 3D)
    lazo_A, lazo_B = f.calcular_separatriz(I, np.linalg.norm(w0))
    a.plot(lazo_A[:, 0], lazo_A[:, 1], lazo_A[:, 2], color=VERDE, lw=2.5, ls='--')
    a.plot(lazo_B[:, 0], lazo_B[:, 1], lazo_B[:, 2], color=VERDE, lw=2.5, ls='--')

    # Trayectoria actual simulada
    a.plot(U[:, 0], U[:, 1], U[:, 2], color='white', alpha=0.4, lw=1)
    
    estado['recorrido'], = a.plot([], [], [], color='white', lw=2)
    estado['punto'], = a.plot([], [], [], 'o', color=AMARILLO, ms=8)

    lim = 1.15 * max(np.abs(m).max() for m in (*malla_T, *malla_L))
    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
    a.set_xlabel('ω₁'); a.set_ylabel('ω₂'); a.set_zlabel('ω₃')
    a.set_box_aspect((1, 1, 1), zoom=1.15)
    a.view_init(elev, azim)


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
