import math
import numpy as np

# Armijo

def armijo_backtracking(f, x, p, g, a0, rho, c1, alpha_A: float = 0, debug:bool = False):
    """
    [INPUT]
    f: function
    x: point to find alpha
    p: direction search
    g: gradient of f in the point x
    a0: initial alpha
    rho: contractor factor
    c1: armijo constant

    [OUTPUT]
    alpha: step size that satify the armijo condition
    """
    alpha = a0
    while f(x + alpha*p) > f(x) + c1*alpha*g@p: 
        alpha = rho*alpha

        if debug:
            print(f"alpha={alpha}, left={f(x + alpha*p)}, right={f(x) + c1*alpha*g@p}, armijo interval ={[0, alpha_A]}, isin={0 <= alpha and  alpha <= alpha_A}")
    return alpha


# Wolfe

def weak_wolfe(f, df, x, p, g, alpha, c1, c2, debug: bool = False):
    """
    [INPUT]
    f: objetive function
    df: gradient of f
    x: point in iteration
    p: direction vector
    g: df(x)
    alpha: intial value of step size
    c1, c2: Armijo and Wolf constants
    """
    a, b = 0, math.inf
    i = 0
    found = False
    while not found:
        if debug:
            print(f"[iter {i}] found={found}, [a, b]={[a, b]}, alpha={alpha}, phi(alpha)={f(x+alpha*p)}, dphi(alpha)={df(x+alpha*p)@p} [armijo={f(x)+c1*alpha*g@p}, wolfe={c2*g@p}]")
        if f(x + alpha*p) > f(x) + c1*alpha*g@p: # alpha violates the armijo condition (alpha not in [0, alpha_A], step is to large)
            if debug:
                print(f"i={i}, ARMINJO")

            b = alpha; alpha = (a+b)/2
        elif df(x+alpha*p) @ p < c2*g@p: # alpha violate the weak wolf condition (alpha not in [alpa_WW, +INF], step is to small)

            if debug:
                print(f"i={i}, WOLFE")

            a = alpha
            alpha = 2*a if math.isinf(b) else (a+b)/2
        else: found = True

        i += 1

    if debug:
        print(f"i={i}, found={found}, [a, b]={[a, b]}, alpha={alpha}")

    return alpha


