# QUESTION: E1
# SECTION: U7. Newton's method and modi␜ed Hessian
#
import numpy as np
from newton import modified_newton


if __name__=="__main__":
    def f(x): return np.exp(x) -x -1
    def df(x): return  np.exp(x) - 1
    def d2f(x): return np.exp(x)

    xk = modified_newton(f, df, d2f, x0=np.array([1]), delta=1, epsilon=1e-3, k=10, debug=True)
    print(xk)
