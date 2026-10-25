import math
import numpy as np

def cholesky_factorization(M):
    """
    [INPUT]
    M: symmetric matrix

    [OUTPUT]
    L: lower triangular (L.T@L = M)
    ok: ok=false descomposition failed (M is not positive defined), otherwise ok=true
    """

    n = M.shape[0]
    L = np.zeros((n, n))

    for j in range(n):
        d = M[j, j] - sum(L[j, :j]**2)

        if d <= 0: return None, False

        L[j, j] = math.sqrt(d)
        for i in range(j+1, n):
            L[i, j] = (M[i, j] - L[i, :j]@L[j, :j]) / L[j, j]

    return L, True

def shifted_cholesky(H, delta, debug: bool = False):
    """
    [INPUT]
    H: symmetric matrix
    delta: positive constant to ensure eigenvalues are greatter than 0
    debug: show traces of computation

    [OUTPUT]
    L: cholesky factorization (H = L @ L.T)
    t: shift positive constant to ensure H + tI is positive definited
    """

    u = min(np.diag(H))
    t = 0 if u > 0 else delta - u
    I = np.identity(H.shape[0])
    ok = False
    while not ok:
        L, ok = cholesky_factorization(H + t*I)
        if debug:
            print(f"ok={ok}, t={t}\nH+tI={H+t*I}\nL={L}\n\n")

        if not ok:
            t = max(2*t, delta)
        
    return L, t


if __name__=='__main__':

    # M is definted positite - test for 'cholesky' function
    M = np.array([
        [4, 2, -2],
        [2, 10, 5],
        [-2, 5, 6]
        ])

    L, ok = cholesky(M)
    print(L, ok)

    print("="*30)

    # H is not definited positive - test for 'shifted_cholesky' function
    # eigenvalues(H) = {3, -1}
    H = np.array([
        [1, 2],
        [2, 1]
        ])
    L, t = shifted_cholesky(H, delta=0.1, debug=True)
    print(L, t)

