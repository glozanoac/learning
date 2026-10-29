#QUESTION: P19
#SECTION: U18. Conjugate directions and linear conjugate gradient

import numpy as np
import matplotlib.pyplot as plt
from linear_cg import linear_cg

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

if __name__=="__main__":
    # item A
    print("="*10+ " [Item A] " + "="*10)
    A = np.diag([1.0, 1.0, 10.0, 10.0])
    b = np.array([1, 1, 1, 1], dtype=np.float32)

    x0 = np.array([0, 0, 0, 0], dtype=np.float32)
    M = np.identity(4)

    x_cg, converged, cg_path = linear_cg(A, b, x0, M, epsilon=1e-4, k=10, debug=True, collect=True)
    print(x_cg, converged)


    normA = lambda x: x@A@x
    print("="*10+ " [Item B] " + "="*10)
    x_star = np.linalg.inv(A)@b # analytic solution
    KA = 10 # largest / smallest eigenvalues of A
    rate = (KA-1)/(KA+1)
    k = 20
    while k > 0:
        # with g = A@x0 -b the secession diverge from x*. I put g = -(A@x0 -b)
        # also it keep constant the rate, but not equal to the teorical one.
        g = -(A@x0 - b) # g = df(x0), Is this a descent direction? NO, since df(x0)@g = ||df(x0)|| >= 0
        alpha=(g@g)/(g@A@g)
        x1 = x0 + alpha*g

        print(f"[k={k}] x0={x0}, x1={x1}, ||x1-x*||/||x0-x*||={normA(x1-x_star)/normA(x0-x_star)}, rate={rate}")
        x0 = x1
        k -= 1

