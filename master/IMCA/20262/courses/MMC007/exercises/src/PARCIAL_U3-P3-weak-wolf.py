# Agenda=PARCIAL; Section=U3; Question=P3 PRACTICE
#
# Objetive:
# - compute step size to satify weak wolfe conditions
#
# NOTE: Confirm the computed step size with the computed intervasl for weak wolf in PARCIAL_U3-P1-armijo-wolfe.py

import math
import numpy as np

def weak_wolf(f, df, x, p, g, alpha, c1, c2, debug: bool = False):
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

if __name__=="__main__":
    def f(x): return 10*x[0]**2 + x[1]**2
    def df(x): return np.array([20*x[0], 2*x[1]])

    x = np.array([1, 1]) # x0: point on sucession (k=0)
    c1 = 1e-4
    c2 = 0.9
    # alpha = 1 # too large (violate armijo conditions)
    # alpha = 1e-3 # too small (violate weak wolfe conditions)
    alpha = 1 # initial guess
    g = df(x)
    p = -df(x) # search direction (descent one)

    # since p = -df(x) and g = df(x), then s= g @ p = = -p @ p = -||p||^2
    alpha = weak_wolf(f, df, x, p, g, alpha, c1, c2, debug=True)
    print(alpha)


