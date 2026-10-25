import numpy as np
from newton import modified_newton

# agend=PARCIAL, section=U7, problem=P21, V22

if __name__=="__main__":
    def f(x): return np.exp(x)
    def df(x): return  np.exp(x) - 1
    def d2f(x): return np.exp(x)

    xk = modified_newton(f, df, d2f, x0=np.array([1]), delta=1, epsilon=1e-3, k=10, debug=True)
    print(xk)
