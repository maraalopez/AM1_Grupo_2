import numpy as np

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

def compute_invariants_drift(omega_series, I):
    # T0 y L0 iniciales
    T0 = invariant_energy(omega_series[0], I)
    L0 = invariant_angular_momentum(omega_series[0], I)

    # Evaluar en toda la serie temporal
    T_t = np.array([invariant_energy(w, I) for w in omega_series])
    L_t = np.array([invariant_angular_momentum(w, I) for w in omega_series])

    delta_T = np.abs(T_t - T0)
    delta_L = np.abs(L_t - L0)
    return delta_T, delta_L
