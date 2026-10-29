# linear conjugated gradient
import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode='fixed') # show only three decimals

def linear_cg(A, b, x0, M, epsilon, k, debug: bool = False, collect: bool = False):
    """
    f(x) = x.T@A@x / 2 - b.T@x
    [INPUT]
    A: matrix symmetric positive definite
    b: vector of f function
    x0: starting point
    M: preconditioner matrix
    epsilon: maximum error
    k: max number of iterations

    [OUTPUT] (x, converged)
    converged: flag indicating if the sucession converged
    x: if converged=1, then solution to Ax=b, otherwise last point (iteration limit)
    x_list: point in the sucession
    """

    x = x0
    r = A@x0 -b
    invM = np.linalg.inv(M)
    y = invM@r
    p = -y
    d0 = r@y

    x_list = x0

    while k > 0:
        if np.linalg.norm(r) < epsilon*np.linalg.norm(b): # scale invariance
            # why not ||r|| < epsilon, because for c >0 (cA) x = cb, then r = cA - cb, if c = 1e6, then ||r|| < epsilon could not be reach, to avoid that problem whe compute the relative error ||r|| / ||b|| < epsilon
            return x, 1, x_list

        w = A@p
        alpha = d0 / (p@w)
        x = x + alpha*p; r = r + alpha*w
        y =invM@r; d1 = r@y; beta = d1/d0; d0 = d1
        p = -y + beta*p

        if collect:
            x_list = np.vstack([x_list, x])

        if debug:
            print(f"[k={k}] r={r}, p={p}, beta={beta:10.3f}")

        k -= 1

    return x, 0, x_list
