# gui_reactive.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import fisica as f
import esquemas_temporales as et

plt.style.use('dark_background')
# Opción A: Estilo "Terminal/Telemetría" (Recomendado para cuadros de mando)
plt.rcParams['font.family'] = 'monospace'

# Opción B: Estilo "Moderno y Limpio"
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Verdana', 'Arial']

# Hacer que todos los textos sean un poco más grandes por defecto
plt.rcParams['font.size'] = 10

def lanzar_gui():
    # 1. Crear la figura principal 
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("TELEMETRÍA Y DINÁMICA DE ACTITUD SATELITAL", 
                fontsize=18, fontweight='heavy', color='white')
    plt.subplots_adjust(bottom=0.4, wspace=0.3) # wspace separa un poco los gráficos
    
    # 2. Crear los TRES paneles (1 fila, 3 columnas)
    # El primer panel (índice 1) es el espacio de fases 3D
    ax_phase = fig.add_subplot(1, 3, 1, projection='3d')
    ax_phase.set_title("Espacio de Fases 3D (Poinsot)")
    ax_phase.set_xlabel(r'$\omega_1$')
    ax_phase.set_ylabel(r'$\omega_2$')
    ax_phase.set_zlabel(r'$\omega_3$')

    # Al configurar ax_phase:
    ax_phase.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_phase.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_phase.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax_phase.grid(color='gray', alpha=0.3, linestyle='dashed')

    # Los paneles 2 y 3 son los normales en 2D para las derivas
    ax_T = fig.add_subplot(1, 3, 2)
    ax_L = fig.add_subplot(1, 3, 3)
    
    # Inicializar las líneas de las gráficas 2D
    line_T, = ax_T.plot([], [], 'r-', label='ΔT(t)')
    line_L, = ax_L.plot([], [], 'b-', label='ΔL(t)')
    
    ax_T.set(title="Deriva 2T", xlabel="Tiempo [s]", ylabel="Error ΔT")
    ax_L.set(title="Deriva L²", xlabel="Tiempo [s]", ylabel="Error ΔL")
    ax_T.grid(True)
    ax_L.grid(True)

    ax_T.set_title("Deriva Energía (2T)", color='cyan', fontweight='bold')
    ax_T.set_xlabel("Tiempo [s]", color='lightgray')
    ax_T.set_ylabel("Error absoluto", color='lightgray')

    ax_L.set_title("Deriva Momento (L²)", color='magenta', fontweight='bold')
    ax_L.set_xlabel("Tiempo [s]", color='lightgray')
    ax_L.set_ylabel("Error absoluto", color='lightgray')

    # Para las gráficas 2D
    ax_T.grid(color='gray', linestyle=':', alpha=0.5)
    ax_L.grid(color='gray', linestyle=':', alpha=0.5)


    # 2. Definir los Sliders y RadioButtons
    s_I1 = Slider(plt.axes([0.1, 0.25, 0.35, 0.03]), 'I1', 0.1, 5.0, valinit=1.0, color='orange')
    s_I2 = Slider(plt.axes([0.1, 0.20, 0.35, 0.03]), 'I2', 0.1, 5.0, valinit=2.0, color='orange')
    s_I3 = Slider(plt.axes([0.1, 0.15, 0.35, 0.03]), 'I3', 0.1, 5.0, valinit=3.0, color='orange')
    # El slider de dt lo bajamos a y=0.12
    ax_dt = plt.axes([0.1, 0.10, 0.35, 0.03])
    s_dt = Slider(ax_dt, 'Δt', 0.001, 0.2, valinit=0.05, color='crimson')
    
    s_w1 = Slider(plt.axes([0.55, 0.25, 0.35, 0.03]), 'ω1(0)', -5.0, 5.0, valinit=0.01, color='teal')
    s_w2 = Slider(plt.axes([0.55, 0.20, 0.35, 0.03]), 'ω2(0)', -5.0, 5.0, valinit=1.0, color='teal')
    s_w3 = Slider(plt.axes([0.55, 0.15, 0.35, 0.03]), 'ω3(0)', -5.0, 5.0, valinit=0.01, color='teal')


    # Coordenadas (x, y) relativas a la ventana (de 0 a 1)
    fig.text(0.1, 0.32, "PARÁMETROS FÍSICOS (Inercias)", color='lightgray', fontweight='bold')
    fig.text(0.55, 0.30, "CONDICIONES INICIALES (ω)", color='lightgray', fontweight='bold')
    fig.text(0.1, 0.13, "MOTOR NUMÉRICO", color='lightgray', fontweight='bold')

    # Y los RadioButtons los ponemos más abajo, en y=0.02
    ax_radio = plt.axes([0.1, 0.02, 0.35, 0.08])
    # 1. Crear los RadioButtons
    radio_esquemas = RadioButtons(ax_radio, ('Euler', 'RK4', 'Crank-Nicolson'), active=2)


    # 4. Cambiar el color del texto a blanco para que resalte en el fondo oscuro
    for label in radio_esquemas.labels:
        label.set_color('white')


    # 3. Definir la función de actualización (callback) dentro del mismo entorno
    def update(val):
        I = [s_I1.val, s_I2.val, s_I3.val]
        U0 = np.array([s_w1.val, s_w2.val, s_w3.val])
        dt = s_dt.val
        
        t = et.partition(0, 15.0, int(15.0 / dt))
        
        esquema_str = radio_esquemas.value_selected
        if esquema_str == 'Euler': scheme = et.euler
        elif esquema_str == 'RK4': scheme = et.rk4
        else: scheme = et.crank_nicholson
            
        def f_sistema(omega, t_val):
            return f.euler_equations(omega, I, t_val)
            
        U = et.cauchy_problem(F=f_sistema, U0=U0, t=t, scheme=scheme)
        
        T0 = f.invariant_energy(U0, I)
        L0 = f.invariant_angular_momentum(U0, I)
        modulo_L0 = np.sqrt(L0)
        
        # Llamamos a la función genérica pasándole la función de energía
        delta_T = et.calcular_deriva(U, f.invariant_energy, I)
        
        # Llamamos a la misma función genérica pasándole la de momento angular
        delta_L = et.calcular_deriva(U, f.invariant_angular_momentum, I)

        # --- DIBUJO DEL ESPACIO DE FASES 3D ---
        
        # 1. Borramos lo que hubiera dibujado en el paso anterior
        ax_phase.clear()
        ax_phase.set_title("Espacio de Fases 3D (Poinsot)")
        ax_phase.set_xlabel(r'$\omega_1$')
        ax_phase.set_ylabel(r'$\omega_2$')
        ax_phase.set_zlabel(r'$\omega_3$')

        # 3. Generamos las mallas desde tu capa física
        malla_T, malla_L = f.superficies_poinsot(I, T0, modulo_L0, num_puntos=50)

        # (Después de dibujar los elipsoides malla_T y malla_L...)

        # Calculamos la separatriz asumiendo que Omega de referencia es la magnitud de U0
        Omega_norm = np.linalg.norm(U0)
        separatriz = f.calcular_separatriz(I, Omega_norm)

        # La dibujamos en color verde brillante o amarillo para que destaque
        ax_phase.plot(separatriz[:, 0], separatriz[:, 1], separatriz[:, 2], 
                      color='lime', linewidth=3, linestyle='--', label='Separatriz')


        # 4. Dibujamos el Elipsoide de Energía (rojo semitransparente)
        ax_phase.plot_surface(malla_T[0], malla_T[1], malla_T[2], 
                              color='red', alpha=0.2, edgecolor='none')

        # 5. Dibujamos el Elipsoide de Momento Angular (malla azul)
        ax_phase.plot_wireframe(malla_L[0], malla_L[1], malla_L[2], 
                                color='blue', alpha=0.2, linewidth=0.5)

        # 6. Superponemos la trayectoria real de tu integración numérica
        ax_phase.plot(U[:, 0], U[:, 1], U[:, 2], color='black', linewidth=2, label='ω(t)')
        ax_phase.legend()
        
        # --- ACTUALIZACIÓN DE LAS GRÁFICAS 2D ---
        line_T.set_data(t, delta_T)
        line_L.set_data(t, delta_L)

        # En tu función update, después de set_data:
        ax_T.clear() # Borramos para re-dibujar el relleno
        ax_T.plot(t, delta_T, color='cyan', linewidth=2)
        ax_T.fill_between(t, 0, delta_T, color='cyan', alpha=0.2)

        ax_L.clear()
        ax_L.plot(t, delta_L, color='magenta', linewidth=2)
        ax_L.fill_between(t, 0, delta_L, color='magenta', alpha=0.2)
        ax_T.relim(); ax_T.autoscale_view()
        ax_L.relim(); ax_L.autoscale_view()
        
        # Refrescar pantalla
        fig.canvas.draw_idle()


    # 4. Conectar los widgets y mostrar
    for s in [s_I1, s_I2, s_I3, s_dt, s_w1, s_w2, s_w3]:
        s.on_changed(update)
    radio_esquemas.on_clicked(update)

    update(None)
    plt.show()