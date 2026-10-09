from numpy import array, zeros
from typing import Callable

def Euler(U: array, t: float, dt: float, F: Callable[[array], array]) -> array:
    """
    Performs one step of the Euler method for solving ODEs.

    Parameters:
    U (array): Current state vector.
    t (float): Current time.
    dt (float): Time step size.
    F (Callable[[array], array]): Function that computes the derivative.

    Returns:
    array: Updated state vector after one time step.
    """
    return U + dt * F(U)