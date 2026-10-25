import numpy as np
from wolf import weak_wolf

def steepest_descent(f, df, x0, epsilon, k, debug: bool = False, collect: bool = False):
    """
    [INPUT]
    f: function to minimize
    df: gradient of objective function
    x0:starting point
    epsilon: error of stationary point
    k: maximum number of steps 

    [OUTPUT]
    (x, convergedi, x_list): stationary point in case converged=1, otherwise last computed xk (it hits maximum iteration)
    x_list: list of points in the sucession
    """


    # f: cuadratic function (Q: matrix related to f)
    # for the case in Question=P4 PRACTICE x*=0 (generally this is unknow), but the idea here is 
    # verify that the sucession {xk} satisfy some properties (studied in lessons)
    # - convergence rate is smaller that ((k-1)/(k+1))**2, where k: condition number of matrix
    kQ = 10 # Q =diag(20, 2)
    x_star = np.array([0, 0])
    g0 = g1 = np.array([0, 0]) # df(x0), df(x1) 

    x_list = x0

    while k > 0:
        g1 = g = df(x0)


        if np.linalg.norm(g) < epsilon: 
            print(f"[CONVEGE - iter {k}] error={np.linalg.norm(g)}, epsilon={epsilon}, x0={x0}")
            return x0, 1, x_list

        p = -g # steepest descent direction
        alpha = weak_wolf(f, df, x0, p, g, alpha=1, c1=1e-4, c2=0.9)

        x1 = x0 + alpha*p

        if debug:
            # r: convergency rate
            # error: will tend to 0, if error < epsilon, we have found an stationary point at x0
            print(f"[iter {k}] alpha={alpha}, g0@g1={g0@g1}, error={np.linalg.norm(g)}, r={[(f(x1)-f(x_star))/(f(x0) - f(x_star)), ((kQ-1)/(kQ+1))**2]}")

        x0 = x1
        g0 = g1 # saved just to prove that <df(x0), df(x1)> = 0
        x_list = np.vstack([x_list, x0])

        k -=1

    return x0, 0, x_list
