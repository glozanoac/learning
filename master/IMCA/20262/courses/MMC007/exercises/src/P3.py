# QUESTION: P3
# SECTION: U3. Line search: Armijo and Wolfe 
#
#
# Objetive:
# - compute step size to satify weak wolfe conditions
#
# NOTE: Confirm the computed step size with the computed intervasl for weak wolf in P1.py

import math
import numpy as np

from line_search import weak_wolf

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


