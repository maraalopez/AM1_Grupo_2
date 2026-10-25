import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ==============================================================================
# 1. GEOMETRÍA: se reutiliza la capa física y la matemática (sin duplicar código)
# ==============================================================================
import fisica as f
import esquemas_temporales as et

# ==============================================================================
# 2. CONFIGURACIÓN DE LA INTERFAZ GRÁFICA
# ==============================================================================

def crear_visor_poinsot() -> None:
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

    def actualizar(_: float | None = None) -> None:
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

        # 1. Constantes críticas: rotación pura sobre el eje intermedio con velocidad Ω₂
        #    2T = I2 Ω₂²   y   L = I2 Ω₂   (es decir, L² = 2 I2 T: caso separatriz)
        dos_T0 = i2 * om**2
        L0 = i2 * om
        (a_T, b_T, c_T), (a_L, b_L, c_L) = f.semiejes_poinsot(I, dos_T0, L0)

        # 2. Superficies de Poinsot
        malla_T = et.generar_malla_elipsoide(a_T, b_T, c_T, num_puntos=40)
        malla_L = et.generar_malla_elipsoide(a_L, b_L, c_L, num_puntos=40)

        # Superficie de Energía (Rojo/Salmón translúcido)
        ax_3d.plot_surface(*malla_T, color='#ff5a5f', alpha=0.25, linewidth=0, shade=True)

        # Superficie de Momento Angular (Malla Azul)
        ax_3d.plot_wireframe(*malla_L, color='#4aa8ff', alpha=0.45, linewidth=0.6, rstride=2, cstride=2)

        # 3. Lazos de la Separatriz (Verde Neón)
        lazo1, lazo2 = f.separatriz(I, dos_T0)
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