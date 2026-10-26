import numpy as np
from wolfe import weak_wolfe

def quasi_newton(f, df, x0, H0, epsilon, k, debug: bool = False, collect: bool = False):
    """
    [INPUT]
    f: function to optimize
    df: gradient of f
    x0: starting point
    H0: initial positive defined maxtrix on sucession Hk
    epsilon: positive threshold to say that xk had converged to an stationary point
    k: maximum number of iterations
    debug: show steps of computation
    collect: collect point of sucession {xk}

    [OUTPUT]
    converged: is 1 if the sucession xk converged to the stationary point ||df(xk)|| < epsilon, otherwise 0
    x1: if converged=1 xk is stationary point, else xk is last point in sucession
    x_list: list of point in sucession (collected only when collect=True)
    """

    # NOTATION: g0 = g(xk), g1 = g(xk+1), similar for other objects like H, if index is not necesary just keep name of variable (e.g. we use s instead of sk)

    x_star = np.array([0.695884, -1.347942]) # only to demostrate that ||xk+1 - x*|| / ||xk - x*|| -> 0
    x_list = x0
    g0 = df(x0)
    while k > 0:
        if np.linalg.norm(g0) < epsilon:
            return x0, 1, x_list
        p = -H0@g0
        alpha = weak_wolfe(f, df, x0, p, g0, alpha=1, c1=1e-4, c2=0.9)
        s = alpha*p; x1 = x0 +s; g1 = df(x1); y=g1-g0

        if s@y <= 0: H1 = H0
        else: # DFP
            vs = s.reshape(-1, 1)
            vy = y.reshape(-1, 1)
            H1 = H0 + vs@vs.T/(s@y) - H0@vy @ vy.T@H0/(y.T@H0@y)

        if debug:
            print(f"[k={k}] x0={x0}, x1={x1}, f(x1)={f(x1)}, ||g1||={np.linalg.norm(g1)}, alpha={alpha}, s@y={s@y}, e={np.linalg.norm(x1-x_star)/np.linalg.norm(x0-x_star)}")
            print()

        x0 = x1
        H0 = H1
        g0 = g1

        if collect: x_list = np.vstack([x_list, x0])
        k -= 1

    return x0, 0, x_list


if __name__=="__main__":
    def f(x): return x[0]**4 +x[0]*x[1] + (1+x[1])**2
    def df(x): return np.array([4*x[0]**3+x[1], x[0] + 2*(1+x[1])])

    H0 = np.identity(2)
    x0 = np.array([1, 1])

    x1, converged, x_list = quasi_newton(f, df, x0, H0, epsilon=1e-6, k=10, debug=True, collect=True)


    print(x1, converged)

    ## PLOT
    import matplotlib.pyplot as plt

    x_star = np.array([0.695884, -1.347942])

    # Square neighborhood containing all iterates and x*
    P = np.vstack([x_list, x_star])
    c = (P.min(axis=0) + P.max(axis=0)) / 2
    r = 0.6 * (P.max(axis=0) - P.min(axis=0)).max()
    u = np.linspace(c[0]-r, c[0]+r, 200)
    v = np.linspace(c[1]-r, c[1]+r, 200)
    U, V = np.meshgrid(u, v)
    Z = f(np.array([U, V]))

    pts = {'$x_0$': x_list[0], '$x^*$': x_star}

    fig = plt.figure(figsize=(12, 5))

    # 3D surface + iterates
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(U, V, Z, cmap='viridis', alpha=0.6)
    ax.contour(U, V, Z, levels=30, offset=Z.min(), cmap='viridis')
    for lab, p in pts.items():
        ax.scatter(*p, f(p), color='red', s=30)
        ax.text(*p, f(p), lab)
    ax.plot(x_list[:, 0], x_list[:, 1], f(x_list.T), 'b.-', lw=1.5, zorder=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('$f$')

    # Level curves + iterates
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(U, V, Z, levels=30, cmap='viridis')
    ax2.clabel(cs, fontsize=7)
    for lab, p in pts.items():
        ax2.plot(*p, 'ro')
        ax2.annotate(lab, p, textcoords='offset points', xytext=(5, 5))
    ax2.plot(x_list[:, 0], x_list[:, 1], 'b.-', lw=1.5, ms=6, label='iterates (DFP)')
    ax2.plot(*x_list[0], 'bs')
    ax2.plot(*x_list[-1], 'b*', ms=10)
    ax2.set_xlim(u[0], u[-1]); ax2.set_ylim(v[0], v[-1])
    ax2.set_aspect('equal')
    ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
    ax2.legend()

    plt.tight_layout()
    plt.savefig("quasi_newton.png", dpi=200, bbox_inches='tight')
    plt.show()
