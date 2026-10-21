# Agenda: PARCIAL
# Question: (Section U3) P1 PRACTICE
#
# Objective:
#   Computation of step size that satisfy Armijo condition for an step of the iteration
#
# Context (in step k):
# Our intention is to find an 'alpha', such that f(xk + alpha*pk) <= f(xk) + c1*alpha*df(xk)@pk
# where: pk= search direction at step k, xk=point of sucession at step k, df(xk)= gradient of f at point xk
#
# for the given function: f(x1, x2) = 10*x1**2 + x2**2
#
# The Arminjo interval is [0, alpha_A]
# with alpha_A solution of f(xk + alpha*pk) = f(xk) + c1*alpha*df(xk)@pk
# ... (pencil, pen and coffee), gives you [check: extra/PARCIAL_U3-P1-armijo-wolfe-INTERVALS.pdf]:
# alpha_A = (404*(1-c1))/4004 = 0.100889010989011
# Therefore, we are searching an 'alpha' that belong to [0, 0.100889010989011] interval
#
#
# [PART2]
# Is the computed step also satisfy weak and strong wolf conditions?
# 
# Let's compute weak and strong wolf acceptance interval
# weak wolf = [alpha_WW, alpha_A]
# ... (pencil, pen and coffee), gives you [check: extra/PARCIAL_U3-P1-armijo-wolfe-INTERVALS.pdf]:
# alpha_WW=(404*(1-c2))/ 8008
#
#
# strong wolf = [alpha_WW, min(alpha_A, alpha_SW)]
# ... (pencil, pen and coffee), gives you [check: extra/PARCIAL_U3-P1-armijo-wolfe-INTERVALS.pdf]:
# alpha_SW=(404*(1+c2))/ 8008

import numpy as np

# function and gradient
def f(x): return 10*x[0]**2 + x[1]**2
def df(x): return np.array([20*x[0], 2*x[1]])

x = np.array([1, 1]) # point of succesion (at an specific k)

alpha = 1 # initial step size
rho = 0.5 # contraction factor
c1 = 1e-4 # arminjo constant
p = -df(x) # search direction (in this case gradient descent)

# NOTE: p = -df(x), then s = df(x) @ p  = - p @ p = -||p||^2
s = -np.linalg.norm(p)**2

#alpha_A = 0, 0.100889010989011
alpha_A = (404*(1-c1))/4004
while f(x + alpha*p) > f(x) + c1*alpha*s: 
    alpha = rho*alpha

    print(f"alpha={alpha}, left={f(x + alpha*p)}, right={f(x) + c1*alpha*s}, armijo interval ={[0, alpha_A]}, isin={0 <= alpha and  alpha <= alpha_A}")


print("\n\n")
for c2 in [0.1, 0.9]:

    #alpha_A = (404*(1-c1))/4004
    alpha_WW=(404*(1-c2))/ 8008
    alpha_SW=(404*(1+c2))/ 8008

    # weak wolf = [alpha_WW, alpha_A]
    print(f"\n[c2={c2}] weak wolf interval: {[alpha_WW, alpha_A]}, alpha={alpha}, isin={alpha_WW <= alpha and alpha <= alpha_A}")

    # strong wolf = [alpha_WW, min(alpha_A, alpha_SW)]
    print(f"[c2={c2}] strong wolf interval: {[alpha_WW, min(alpha_SW, alpha_A)]} (alpha_SW={alpha_SW}, alpha_A={alpha_A}), alpha={alpha}, isin={alpha_WW <= alpha and alpha <= min(alpha_SW, alpha_A)}")
