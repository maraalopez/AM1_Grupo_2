import numpy as np
from scipy.interpolate import interp1d
import os
import sys

# Añade la carpeta raíz (AM1_Grupo_2) al path donde Python busca módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from milestone_2.esquemas_temporales import cauchy_problem

def richardson_error(F, U0, t, scheme, p):
    """
    Evalúa el error del método numérico mediante Extrapolación de Richardson.
    
    Parámetros:
    -----------
    F      : función del lado derecho del PVI, F(U, t)
    U0     : array con las condiciones iniciales
    t      : malla temporal gruesa de longitud N + 1
    scheme : función del esquema temporal de un paso (e.g. euler, rk4, crank_nicholson)
    p      : orden de precisión del esquema (e.g. p=1 para Euler, p=2 para Crank-Nicolson, p=4 para RK4)
    
    Devuelve:
    --------
    E_fine   : estimación del error absoluto en la malla fina (matriz de dimensión (N+1, Nv))
    U_ext    : solución extrapolada de orden p+1 evaluada en los puntos de t
    E_coarse : estimación del error absoluto en la malla gruesa
    """
    N = len(t) - 1
    t0, tf = t[0], t[-1]
    
    # 1. Solución en la malla gruesa (N pasos, tamaño de paso h)
    U_coarse = cauchy_problem(F, U0, t, scheme)
    
    # 2. Construcción de la malla fina (2N pasos, tamaño de paso h/2)
    t_fine = np.linspace(t0, tf, 2 * N + 1)
    
    # 3. Solución en la malla fina
    U_fine_full = cauchy_problem(F, U0, t_fine, scheme)
    
    # Submuestreo de la malla fina para tomar solo los puntos coincidentes con t (paso de 2 en 2)
    U_fine = U_fine_full[::2, :]
    
    # 4. Cálculo del error estimado para la malla fina y la malla gruesa
    E_fine = (U_fine - U_coarse) / (2**p - 1)
    E_coarse = (2**p) * E_fine
    
    # 5. Solución extrapolada de mayor orden (orden p + 1)
    U_ext = U_fine + E_fine
    
    return E_fine, U_ext, E_coarse


def evaluate_convergence_rate(F, U0, t_span, scheme, N_list, exact_sol=None):
    """
    Calcula la tasa de convergencia p de un esquema temporal.
    
    Parámetros:
    -----------
    F         : función F(U, t) del lado derecho de la EDO
    U0        : condición inicial
    t_span    : tupla o lista (t0, tf)
    scheme    : función del esquema de un paso (e.g. euler, rk4)
    N_list    : lista o array con distintos valores de N (pasos de tiempo)
    exact_sol : función u_exacta(t) opcional para evaluar el error exacto
    
    Devuelve:
    --------
    h_list : array con los tamaños de paso h
    errors : array con los errores globales para cada h
    p      : orden de convergencia estimado (pendiente del ajuste lineal)
    """
    t0, tf = t_span[0], t_span[-1]
    h_list = []
    errors = []
    
    for N in N_list:
        t = np.linspace(t0, tf, N + 1)
        h = t[1] - t[0]
        
        # Solución numérica
        U_num = cauchy_problem(F, U0, t, scheme)
        
        # Cálculo del error global en t = tf
        if exact_sol is not None:
            U_exact_tf = exact_sol(tf)
        else:
            # Si no hay solución analítica, usamos una malla de referencia muy fina (N_ref = 10000)
            t_ref = np.linspace(t0, tf, 10001)
            U_ref = cauchy_problem(F, U0, t_ref, scheme)
            U_exact_tf = U_ref[-1, :]
            
        err = np.linalg.norm(U_num[-1, :] - U_exact_tf)
        
        h_list.append(h)
        errors.append(err)
        
    h_array = np.array(h_list)
    err_array = np.array(errors)
    
    # Ajuste por mínimos cuadrados de ln(Error) = p * ln(h) + C
    p, _ = np.polyfit(np.log(h_array), np.log(err_array), 1)
    
    return h_array, err_array, p