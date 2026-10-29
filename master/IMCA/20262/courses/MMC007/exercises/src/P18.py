#QUESTION: P18
#SECTION: U18. Conjugate directions and linear conjugate gradient

import numpy as np
import matplotlib.pyplot as plt
from linear_cg import linear_cg

if __name__=="__main__":
    A = np.array([[4, 1],[1, 3]], dtype=np.float32)
    b = np.array([1, 2], dtype=np.float32)

    def f(x): return x@A@x/2 - b@x

    x0 = np.array([0.0, 0.0])
    M = np.identity(2)

    x_cg, converged, cg_path = linear_cg(A, b, x0, M, epsilon=1e-4, k=10, debug=True, collect=True)
    print(x_cg, converged)


    # GRAPHICS & COMPARISON

    def exact_step(x, p):
        r = A @ x - b
        return x + (-(r @ p) / (p @ A @ p)) * p

    def coordinate_descent(x0, n_iter=6):
        xs = [x0]
        x = x0.copy()
        for k in range(n_iter):
            e = np.zeros(2); e[k % 2] = 1.0
            x = exact_step(x, e)
            xs.append(x)
        return np.array(xs)

    def steepest_descent(x0, n_iter=10):
        xs = [x0]
        x = x0.copy()
        for _ in range(n_iter):
            r = A @ x - b
            if np.linalg.norm(r) < 1e-10:
                break
            x = exact_step(x, -r)
            xs.append(x)
        return np.array(xs)

    # --- iterates ---
    x_star = np.linalg.solve(A, b)
    cg_path = np.atleast_2d(cg_path)
    cd_path = coordinate_descent(x0)
    sd_path = steepest_descent(x0)

    # --- contours of f ---
    X1, X2 = np.meshgrid(np.linspace(-0.08, 0.42, 300), np.linspace(-0.05, 0.72, 300))
    Z = 0.5 * (A[0, 0]*X1**2 + 2*A[0, 1]*X1*X2 + A[1, 1]*X2**2) - b[0]*X1 - b[1]*X2
    f_star = f(x_star)
    levels = f_star + np.array([1e-3, 4e-3, 3e-2, 1.5e-1, 4e-1, 7e-1])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(X1, X2, Z, levels=levels, colors='0.7', linewidths=0.8)

    ax.plot(cd_path[:, 0], cd_path[:, 1], ':', color='0.35', lw=1.5,
            marker='x', ms=5, label='coordinate directions')
    ax.plot(sd_path[:, 0], sd_path[:, 1], '--', color='tab:red', lw=1.5,
            marker='>', ms=5, mfc='none', label='steepest descent')
    ax.plot(cg_path[:, 0], cg_path[:, 1], '--', color='blue', lw=2.5,
            marker='o', ms=7, label='CG')

    ax.plot(*x_star, 'kx', ms=10, mew=2)
    ax.annotate(r'$x^*$', x_star, xytext=(-12, 12), textcoords='offset points', fontsize=12)
    ax.annotate(r'$x_0$', x0, xytext=(8, -14), textcoords='offset points', fontsize=12)

    ax.set_xlim(-0.08, 0.42); ax.set_ylim(-0.05, 0.72)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1$', fontsize=14); ax.set_ylabel(r'$x_2$', fontsize=14)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor='k')

    plt.tight_layout()
    plt.savefig('P18-cg_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("x* (analytic):", x_star, "| x (CG):", x_cg, "| converged:", converged)



