import numpy as np
import esquemas_temporales as et 

def euler_equations(omega, I, t):

        # omega: angular velocity vector [wx, wy, wz]
        # I: moments of inertia [Ix, Iy, Iz]
        # domega_dt: time derivative of angular velocity vector [dwx/dt, dwy/dt, dwz/dt]
        domega_dt = np.zeros(3)
        domega_dt[0] = (I[1] - I[2]) * omega[1] * omega[2] / I[0]
        domega_dt[1] = (I[2] - I[0]) * omega[2] * omega[0] / I[1]
        domega_dt[2] = (I[0] - I[1]) * omega[0] * omega[1] / I[2]
        
        return domega_dt

def jacobian(omega, I):
    # Jacobian matrix of the Euler equations
    J = np.zeros((3, 3))
    J[0, 1] = (I[1] - I[2]) * omega[2] / I[0]
    J[0, 2] = (I[1] - I[2]) * omega[1] / I[0]
    J[1, 0] = (I[2] - I[0]) * omega[2] / I[1]
    J[1, 2] = (I[2] - I[0]) * omega[0] / I[1]
    J[2, 0] = (I[0] - I[1]) * omega[1] / I[2]
    J[2, 1] = (I[0] - I[1]) * omega[0] / I[2]
    
    return J


def invariant_energy(omega, I):

    return I[0] * omega[0]**2 + I[1] * omega[1]**2 + I[2] * omega[2]**2

def invariant_angular_momentum(omega, I):

    return I[0]**2 * omega[0]**2 + I[1]**2 * omega[1]**2 + I[2]**2 * omega[2]**2



def superficies_poinsot(I, dos_T0, L0_cuadrado, num_puntos=50):
    """
    Traduce las constantes de movimiento (2T0, L0^2) y la inercia (I) 
    en dimensiones geométricas (semiejes) para construir las superficies.
    """
    # 1. Elipsoide de ENERGÍA
    # Como 'dos_T0' ya equivale a 2T, dividimos directamente entre la inercia
    a_T = np.sqrt(dos_T0 / I[0])
    b_T = np.sqrt(dos_T0 / I[1])
    c_T = np.sqrt(dos_T0 / I[2])
    
    malla_T = et.generar_malla_elipsoide(a_T, b_T, c_T, num_puntos)
    
    # 2. Elipsoide de MOMENTO ANGULAR
    # Tu función original devuelve L^2, así que sacamos el módulo L0 primero
    #L0 = np.sqrt(L0_cuadrado)
    L0 = L0_cuadrado  # Ya es el módulo, no necesitamos la raíz cuadrada
    
    a_L = L0 / I[0]
    b_L = L0 / I[1]
    c_L = L0 / I[2]
    
    malla_L = et.generar_malla_elipsoide(a_L, b_L, c_L, num_puntos)
    
    return malla_T, malla_L

"""def calcular_separatriz(I, Omega_norm, num_pasos=1000):

    Calcula la curva de la separatriz crítica integrando las ecuaciones de Euler
    desde el punto de silla inestable a lo largo de su autovector principal.

    # 1. El equilibrio inestable es el giro puro en el eje intermedio (I2)
    w_eq = np.array([0.0, Omega_norm, 0.0])
    
    # 2. Calculamos la Jacobiana en ese punto
    J = jacobian(w_eq, I)
    
    # 3. Extraemos autovalores y autovectores
    autovalores, autovectores = np.linalg.eig(J)
    
    # 4. Buscamos el autovalor real positivo (la inestabilidad)
    idx_positivo = np.argmax(np.real(autovalores))
    autovector_inestable = np.real(autovectores[:, idx_positivo])
    
    # 5. Aplicamos una perturbación infinitesimal en esa dirección
    epsilon = 1e-4
    w_inicial = w_eq + epsilon * autovector_inestable
    

   # 6. Definimos la función auxiliar
    def f_sistema(w, t_val):
        return euler_equations(w, I, t_val)
    
    # 7. Integramos la trayectoria pasando nuestra función normal
    # (Usamos RK4 por defecto para garantizar que la separatriz sea suave)
    t_sep = et.partition(0, 20.0, num_pasos) 
    trayectoria_separatriz = et.cauchy_problem(F=f_sistema, U0=w_inicial, t=t_sep, scheme=et.rk4)
    
    return trayectoria_separatriz"""

# En fisica.py
def calcular_separatriz(I, Omega_norm, num_pasos=800):
    w_eq = np.array([0.0, Omega_norm, 0.0])
    J = jacobian(w_eq, I)
    autovalores, autovectores = np.linalg.eig(J)
    
    idx_positivo = np.argmax(np.real(autovalores))
    autovector_inestable = np.real(autovectores[:, idx_positivo])
    
    epsilon = 1e-4
    t_sep = et.partition(0, 25.0, num_pasos)
    
    def f_sistema(w, t_val):
        return euler_equations(w, I, t_val)

    # 1. Integramos la rama principal desde la perturbación
    w_ini = w_eq + epsilon * autovector_inestable
    sep1 = np.asarray(et.cauchy_problem(F=f_sistema, U0=w_ini, t=t_sep, scheme=et.rk4))

    # 2. Generamos las 4 ramas heteroclínicas por simetría física
    rama1 = sep1                            # Rama (+w1, +w2, +w3)
    rama2 = sep1 * np.array([-1, 1, -1])    # Cierra el primer lazo (alrededor de w1)
    rama3 = sep1 * np.array([1, 1, -1])     # Abre el segundo lazo (alrededor de w3)
    rama4 = sep1 * np.array([-1, 1, 1])     # Cierra el segundo lazo

    # Unimos en dos bucles cerrados que se cruzan en el punto de silla (0, w2, 0)
    lazo_A = np.vstack((rama1, rama2[::-1]))
    lazo_B = np.vstack((rama3, rama4[::-1]))

    return lazo_A, lazo_B
