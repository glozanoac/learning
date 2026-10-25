import numpy as np

def backtracking(f, x, p, g, a0, rho, c1, alpha_A: float = 0, debug:bool = False):
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


