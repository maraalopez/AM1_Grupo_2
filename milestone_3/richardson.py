import numpy as np
from scipy.interpolate import interp1d
import os
import sys

# Añade la carpeta raíz (AM1_Grupo_2) al path donde Python busca módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from milestone_2.esquemas_temporales import cauchy_problem


def richardson_error(F, U0, t1, t2, scheme, p):
    """
    Calcula la estimación del error absoluto de integración numérica
    mediante extrapolación de Richardson para dos mallas cualesquiera.
    
    Parámetros:
    -----------
    F      : función F(U, t) del lado derecho del PVI
    U0     : vector de condiciones iniciales
    t1     : malla temporal gruesa (paso h1)
    t2     : malla temporal fina (paso h2 < h1)
    scheme : función del esquema temporal de un paso (e.g. euler, rk4)
    p      : orden numérico del esquema
    
    Devuelve:
    --------
    E_h2   : array numpy de dimensión (len(t2), len(U0)) con el error estimado 
             en cada instante de la malla fina t2.
    """
    h1 = t1[1] - t1[0]
    h2 = t2[1] - t2[0]
    
    # 1. Solución numérica en ambas mallas usando cauchy_problem
    U1 = cauchy_problem(F, U0, t1, scheme)  # Solución con h1
    U2 = cauchy_problem(F, U0, t2, scheme)  # Solución con h2
    
    # 2. Interpolación de U1 sobre los puntos de la malla t2. ¿Hacemos interpolación, o usamos la malla gruesa?
    interp_func = interp1d(t1, U1, axis=0, kind='cubic')
    U1_interp = interp_func(t2)
    
    # 3. Estimación del error en la malla fina
    factor = (h2**p) / (h1**p - h2**p)
    E_h2 = factor * (U2 - U1_interp)
    
    return E_h2


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