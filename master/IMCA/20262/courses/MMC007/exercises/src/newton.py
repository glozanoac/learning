import numpy as np
from armijo_backtrack import backtracking
from cholesky import cholesky_factorization, shifted_cholesky
from linear_systems import forward_substitution, backward_substitution

def modified_newton(f, df, d2f, x0, delta, epsilon, k, debug: bool = False, collect: bool = False):
    """
    [INPUT]
    f: function to minimize
    df: gradient of f
    d2f: hessian of f
    x0: starting point
    delta: positive constant (control 'shifted_cholesky')
    epsilon: maximum error df = 0
    k: maximum number of iterations
    debug: show steps of computation

    kwargs_armijo: armijo backtracking parameters to override

    [OUTPUT]
    x1: stationary point of f (converged=1), otherwise last iterations
    converged: 1 if the sucession xk converged to an stationary point of f with at most an error of epsilon
    x_list: points of succession
    """

    x_list = x0

    while k > 0:
        g = df(x0)
        if np.linalg.norm(g) < epsilon: 
            return x0, 1, x_list

        if collect:
            x_list = np.vstack([x_list, x0])

        H = d2f(x0)
        L, t = shifted_cholesky(H, delta)
        # forward and backward substitution
        y = forward_substitution(L, -g); p = backward_substitution(L.T, y)
        alpha = backtracking(f, x0, p, g, a0=1, rho=0.5, c1=1e-4)
        x1 = x0 + alpha*p

        if debug:
            print(f"[k={k}] x0={x0}, x1={x1}, g=df(x0)={g}, H=d2f(x0)={H}, x1/x0**2={x1/x0**2}, t={t}, alpha={alpha}, p={p}")
            print()
         
        x0 = x1
        k -=1

    return x0, 0, x_list
