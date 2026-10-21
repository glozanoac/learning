# Agenda=PARCIAL; Section=U3; Question=P3 PRACTICE
#
# Objetive:
# - compute step size to satify weak wolfe conditions
#
# NOTE: Confirm the computed step size with the computed intervasl for weak wolf in PARCIAL_U3-P1-armijo-wolfe.py

import math
import numpy as np

def f(x): return 10*x[0]**2 + x[1]**2
def df(x): return np.array([20*x[0], 2*x[1]])

x = np.array([1, 1]) # x0: point on sucession (k=0)
p = -df(x) # search direction (descent one)
c1 = 1e-4
c2 = 0.9
alpha = 1e-3 # initial guess

# since p = -df(x) and g = df(x), then s= g @ p = = -p @ p = -||p||^2
s = -np.linalg.norm(p)**2

a, b = 0, math.inf
i = 0
found = False
while not found:
    print(f"[iter {i}] found={found}, [a, b]={[a, b]}, alpha={alpha}, phi(alpha)={f(x+alpha*p)}, dphi(alpha)={df(x+alpha*p)@p} [armijo={f(x)+c1*alpha*s}, wolfe={c2*s}]")
    if f(x + alpha*p) > f(x) + c1*alpha*s: # alpha violates the armijo condition (alpha not in [0, alpha_A], step is to large)
        print(f"i={i}, ARMINJO")
        b = alpha; alpha = (a+b)/2
    elif df(x+alpha*p) @ p < c2*s: # alpha violate the weak wolf condition (alpha not in [alpa_WW, +INF], step is to small)

        print(f"i={i}, WOLFE")
        a = alpha
        alpha = 2*a if math.isinf(b) else (a+b)/2
    else: found = True

    i += 1



print(f"i={i}, found={found}, [a, b]={[a, b]}, alpha={alpha}")
