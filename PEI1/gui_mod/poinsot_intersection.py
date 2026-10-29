import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==============================================================================
# 1. GEOMETRÍA Y MATEMÁTICA ANALÍTICA DE POINSOT
# ==============================================================================

def generar_elipsoide(a, b, c, num_puntos=50):
    """Genera la malla 3D (X, Y, Z) para un elipsoide de semiejes (a, b, c)."""
    u = np.linspace(0, 2 * np.pi, num_puntos)
    v = np.linspace(0, np.pi, num_puntos)
    U, V = np.meshgrid(u, v)
    X = a * np.cos(U) * np.sin(V)
    Y = b * np.sin(U) * np.sin(V)
    Z = c * np.cos(V)
    return X, Y, Z

def calcular_separatriz_analitica(I, Omega2, num_puntos=300):
    """
    Calcula de forma analítica e inalterable por errores numéricos los dos lazos
    de la separatriz crítica que forman la intersección exacta de las superficies.
    """
    I1, I2, I3 = I
    theta = np.linspace(0, 2 * np.pi, num_puntos)

    # Amplitudes exactas derivadas de la solución plana w3 = ± k * w1
    A = Omega2 * np.sqrt(I2 * (I3 - I2) / (I1 * (I3 - I1)))
    B = Omega2
    C = Omega2 * np.sqrt(I2 * (I2 - I1) / (I3 * (I3 - I1)))

    # Lazo 1: (+C * cos(theta))
    w1_1 = A * np.cos(theta)
    w2_1 = B * np.sin(theta)
    w3_1 = C * np.cos(theta)

    # Lazo 2: (-C * cos(theta))
    w1_2 = A * np.cos(theta)
    w2_2 = B * np.sin(theta)
    w3_2 = -C * np.cos(theta)

    lazo_1 = np.column_stack((w1_1, w2_1, w3_1))
    lazo_2 = np.column_stack((w1_2, w2_2, w3_2))

    return lazo_1, lazo_2

# ==============================================================================
# 2. CONFIGURACIÓN DE LA INTERFAZ GRÁFICA
# ==============================================================================

def crear_visor_poinsot():
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8), facecolor='#0b1020')
    fig.canvas.manager.set_window_title("Geometría de Poinsot - Intersecciones y Separatriz")

    # Panel 3D principal
    ax_3d = fig.add_subplot(111, projection='3d')
    ax_3d.set_facecolor('#131a2e')
    plt.subplots_adjust(bottom=0.25, top=0.92, left=0.05, right=0.95)

    # Estilo de ejes 3D
    for eje in (ax_3d.xaxis, ax_3d.yaxis, ax_3d.zaxis):
        eje.set_pane_color((1, 1, 1, 0))
    ax_3d.tick_params(labelsize=8, colors='#7c8aa5')
    ax_3d.grid(color='#26324a', linestyle=':', alpha=0.6)

    # Sliders inferiores
    ax_I1 = plt.axes([0.10, 0.14, 0.35, 0.025], facecolor='#26324a')
    ax_I2 = plt.axes([0.10, 0.09, 0.35, 0.025], facecolor='#26324a')
    ax_I3 = plt.axes([0.10, 0.04, 0.35, 0.025], facecolor='#26324a')
    ax_Om = plt.axes([0.58, 0.09, 0.30, 0.025], facecolor='#26324a')

    s_I1 = Slider(ax_I1, 'I₁', 0.5, 5.0, valinit=1.0, color='#fbbf24')
    s_I2 = Slider(ax_I2, 'I₂', 0.6, 6.0, valinit=2.0, color='#fbbf24')
    s_I3 = Slider(ax_I3, 'I₃', 0.7, 7.0, valinit=3.0, color='#fbbf24')
    s_Om = Slider(ax_Om, 'Ω₂', 0.5, 3.0, valinit=1.5, color='#4dd0e1')

    for s in (s_I1, s_I2, s_I3, s_Om):
        s.label.set_color('white')
        s.valtext.set_color('white')

    def actualizar(_=None):
        # Aseguramos la condición I1 < I2 < I3 para evitar colapsos
        i1 = s_I1.val
        i2 = max(s_I2.val, i1 + 0.1)
        i3 = max(s_I3.val, i2 + 0.1)
        om = s_Om.val
        I = [i1, i2, i3]

        elev, azim = ax_3d.elev, ax_3d.azim
        ax_3d.clear()

        # Configurar títulos
        ax_3d.set_title("INTERSECCIÓN DE ELIPSOIDES Y SEPARATRIZ EN EL ESPACIO DE FASES",
                         color='white', fontweight='bold', fontsize=12, pad=15)
        ax_3d.set_xlabel('ω₁', color='#a3e635', fontweight='bold')
        ax_3d.set_ylabel('ω₂ (Eje Intermedio)', color='#ff5a5f', fontweight='bold')
        ax_3d.set_zlabel('ω₃', color='#4aa8ff', fontweight='bold')

        # 1. Semiejes teóricos para la Separatriz
        # Energía 2T
        a_T, b_T, c_T = np.sqrt(i2/i1) * om, om, np.sqrt(i2/i3) * om
        # Momento L
        a_L, b_L, c_L = (i2/i1) * om, om, (i2/i3) * om

        # 2. Superficies de Poinsot
        malla_T = generar_elipsoide(a_T, b_T, c_T, num_puntos=40)
        malla_L = generar_elipsoide(a_L, b_L, c_L, num_puntos=40)

        # Superficie de Energía (Rojo/Salmón translúcido)
        ax_3d.plot_surface(*malla_T, color='#ff5a5f', alpha=0.25, linewidth=0, shade=True)

        # Superficie de Momento Angular (Malla Azul)
        ax_3d.plot_wireframe(*malla_L, color='#4aa8ff', alpha=0.45, linewidth=0.6, rstride=2, cstride=2)

        # 3. Lazos de la Separatriz (Verde Neón)
        lazo1, lazo2 = calcular_separatriz_analitica(I, om)
        ax_3d.plot(lazo1[:, 0], lazo1[:, 1], lazo1[:, 2], color='#a3e635', lw=3, ls='--', label='Lazo 1 Separatriz')
        ax_3d.plot(lazo2[:, 0], lazo2[:, 1], lazo2[:, 2], color='#a3e635', lw=3, ls='--', label='Lazo 2 Separatriz')

        # 4. Puntos de Silla Inestables (Contacto Tangente)
        ax_3d.plot([0, 0], [om, -om], [0, 0], 'o', color='#fbbf24', ms=8, label='Puntos de Silla (0, ±Ω₂, 0)')

        # Ajuste proporcional de ejes
        max_lim = max(a_L, b_T, c_T) * 1.2
        ax_3d.set_xlim(-max_lim, max_lim)
        ax_3d.set_ylim(-max_lim, max_lim)
        ax_3d.set_zlim(-max_lim, max_lim)
        ax_3d.set_box_aspect((1, 1, 1))

        if elev is not None:
            ax_3d.view_init(elev, azim)

        ax_3d.legend(loc='upper right', fontsize=8, frameon=False)
        fig.canvas.draw_idle()

    s_I1.on_changed(actualizar)
    s_I2.on_changed(actualizar)
    s_I3.on_changed(actualizar)
    s_Om.on_changed(actualizar)

    actualizar()
    plt.show()

if __name__ == '__main__':
    crear_visor_poinsot()