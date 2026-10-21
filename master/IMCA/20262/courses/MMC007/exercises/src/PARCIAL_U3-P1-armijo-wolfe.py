# Agenda: PARCIAL
# Question: (Section U3) P1 PRACTICE
#
# Objective:
#   Computation of step size that satisfy Armijo condition for an step of the iteration
#
# Context:
#Given and direction search $p_k$ and an **adecuated** $\alpha_k$, the sucession $x_{k+1} = x_k + \alpha_k p_k$ converges to an stationary point of $f$ if $p_k$ is a descent direction ($\nabla f(x_k) p_k < 0$) and the $\alpha_k$ are *adecuated* (Armijo + Wolfe conditions) selected in each step. (more details in: **Zoutendijk Theorem**)
#
#If the funtion $f$ is more conditioned (e.g. convex), then this **stationary point** could be a *minimum*.
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
# Would the calculated step also satisfy the weak and strong Wolfe conditions?
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

def backtracking(f, x, p, g, alpha, rho, c1, debug:bool = False):
    while f(x + alpha*p) > f(x) + c1*alpha*s: 
        alpha = rho*alpha

        if debug:
            print(f"alpha={alpha}, left={f(x + alpha*p)}, right={f(x) + c1*alpha*s}, armijo interval ={[0, alpha_A]}, isin={0 <= alpha and  alpha <= alpha_A}")
    return alpha

if __name__=="__main__":
    # function and gradient
    def f(x): return 10*x[0]**2 + x[1]**2
    def df(x): return np.array([20*x[0], 2*x[1]])

    x = np.array([1, 1]) # point of succesion (at an specific k)

    alpha = 1 # initial step size
    rho = 0.5 # contraction factor
    c1 = 1e-4 # arminjo constant
    g = df(x) # gradient of f in the point x
    p = -g    # search direction (in this case gradient descent)
    # NOTE: p = -df(x), then s = df(x) @ p  = - p @ p = -||p||^2
    s = -np.linalg.norm(p)**2

    alpha_A = (404*(1-c1))/4004
    #alpha_A = 0, 0.10088901098901


    alpha = backtracking(f, x, p, g, alpha, rho, c1, debug=True) 

    print("\n" + "="*30)
    print("Would the calculated step also satisfy the weak and strong Wolfe conditions?")

    for c2 in [0.1, 0.9]:

        #alpha_A = (404*(1-c1))/4004
        alpha_WW=(404*(1-c2))/ 8008
        alpha_SW=(404*(1+c2))/ 8008

        # weak wolf = [alpha_WW, alpha_A]
        print(f"\n[c2={c2}] weak wolf interval: {[alpha_WW, alpha_A]}, alpha={alpha}, isin={alpha_WW <= alpha and alpha <= alpha_A}")

        # strong wolf = [alpha_WW, min(alpha_A, alpha_SW)]
        print(f"[c2={c2}] strong wolf interval: {[alpha_WW, min(alpha_SW, alpha_A)]} (alpha_SW={alpha_SW}, alpha_A={alpha_A}), alpha={alpha}, isin={alpha_WW <= alpha and alpha <= min(alpha_SW, alpha_A)}")
