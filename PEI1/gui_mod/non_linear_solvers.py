from numpy import array, zeros
from numpy.linalg import norm, solve



def newton(func: callable, x0: array) -> array:
    x = x0.copy()
    for _ in range(50):
        residual = func(x)
        if norm(residual) < 1e-10:
            return x
        J = Jacobian(func, x, h=1e-8)
        x = x - solve(J, residual)

    raise RuntimeError("Newton's method did not converge after 50 iterations")


def Jacobian(G: callable, x: array, h: float) -> array:
    n = len(x)
    J = zeros((n, n))
    for i in range(n):
        e = zeros(n)
        e[i] = 1
        x_plus = x + h * e
        x_minus = x - h * e
        J[:, i] = (G(x_plus) - G(x_minus)) / (2 * h)
    return J