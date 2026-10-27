

from numpy import zeros
from numpy.linalg import norm



def Gauss (A, b, x0):
    x = zeros(len(b))
    M = zeros(len(b))
    r = b - A * x0
    for i in range(len(b)):
        for j in range(len(b)):
            if j<=i:

                M[i] = A[i][j]

    k = 1
    while k < 100 and norm(r) > 1e-10*norm(b):
        k += 1
        r = b - A * x
        x = x0 + (M**-1) * (b - A * x0)
        x0 = x
    return x



