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


def jacobian(F, I, omega, eps=1e-8):
    """
    Calcula la matriz Jacobiana numéricamente dada la función de la EDO (F).
    Returns:
    list: Matriz Jacobiana como lista de listas.
    """
    n = len(omega)
    J = np.zeros((n, n), dtype=float)
    
    for j in range(n):
        # Creamos copias del vector de estado para perturbar solo la componente j
        w_plus = list(omega)
        w_minus = list(omega)
        
        w_plus[j] += eps
        w_minus[j] -= eps
        
        # Evaluamos la EDO en los puntos perturbados
        f_plus = np.asarray(F(I, w_plus), dtype=float)
        f_minus = np.asarray(F(I, w_minus), dtype=float)
        
        # Columna j del Jacobiano (derivada respecto a w_j)
        J[:, j] = (f_plus - f_minus) / (2.0 * eps)
        
    return J.tolist()

"""""

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

    """""

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
    #L0 = np.sqrt(L0_cuadrado)
    L0 = L0_cuadrado
    
    a_L = L0 / I[0]
    b_L = L0 / I[1]
    c_L = L0 / I[2]
    
    malla_L = et.generar_malla_elipsoide(a_L, b_L, c_L, num_puntos)
    
    return malla_T, malla_L


def calcular_separatriz(I, Omega_norm, num_pasos=800):
    w_eq = np.array([0.0, Omega_norm, 0.0])
    
    def F(omega, t_val):
        return euler_equations(omega, I, t_val)
    
    J = jacobian(F, I, w_eq, eps=1e-8)
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

def orientacion(t, U):
    """
    Calcula la evolución de la actitud del satélite (matrices de rotación 3x3).
    Transfiere la cinemática física a la capa matemática para obtener las 
    matrices de rotación incrementales.
    """
    R = [np.eye(3)]
    
    for k in range(len(t) - 1):
        # 1. Calculamos la velocidad angular media en el intervalo
        omega_medio = 0.5 * (U[k] + U[k + 1])
        
        # 2. Calculamos el vector de desplazamiento angular instantáneo (v = w * dt)
        dt = t[k + 1] - t[k]
        vector_giro = omega_medio * dt
        
        # 3. matriz exacta de Rodrigues
        R_incremental = et.rotacion(vector_giro)
        
        # 4. Componemos la actitud actualizando la orientación anterior
        R.append(R[-1] @ R_incremental)
        
    return np.array(R)


def ejecutar_simulacion(I, w0, dt, esquema_func, TF=50.0):
    """
    Motor principal de la simulación. 
    Totalmente independiente de interfaces gráficas.
    """
    t = et.partition(0, TF, int(TF / dt))
    
    def F(omega, t_val):
        return euler_equations(omega, I, t_val)
        
    with np.errstate(all='ignore'):
        U = np.asarray(et.cauchy_problem(F=F, U0=w0, t=t, scheme=esquema_func))
        dT = np.asarray(et.calcular_deriva(U, invariant_energy, I))
        dL = np.asarray(et.calcular_deriva(U, invariant_angular_momentum, I))
        
    # Recorte (truncamiento) de la simulación si el método diverge
    valido = np.isfinite(U).all(axis=1)
    if not valido.all():
        n = max(int(valido.argmin()), 2)
        t, U, dT, dL = t[:n], U[:n], dT[:n], dL[:n]
        
    # Calculamos la actitud con la función que acabas de mover a física
    R = orientacion(t, U) 
    
    return t, U, dT, dL, R

