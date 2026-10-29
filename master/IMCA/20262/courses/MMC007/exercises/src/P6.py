# QUESTION: P6
# SECTION: U7. Newton's method and modi␜ed Hessian

import numpy as np
from newton import modified_newton


if __name__=="__main__":
    # agend=PARCIAL, section=U7, problem=P21, V22

    def f(x): return x[0]**2 + x[1]**2 + 4*x[0]*x[1] + x[0]**4 + x[1]**4
    def df(x): return np.array([2*x[0] + 4*x[1] + 4*x[0]**3, 2*x[1] + 4*x[0] + 4*x[1]**3])
    def d2f(x): return np.array([[2+12*x[0]**2, 4], [4, 2+12*x[1]**2]])

    # f has 2 optimial global point (s1, s2)
    # x0=[-1, 1/4] converges to s1
    # x0=[1/4, 0]  converges to s2
    xk, converged, x_list = modified_newton(f, df, d2f, 
                                            x0=np.array([1/4, 0]), delta=1, epsilon=1e-5, k=100, 
                                            debug=True, collect=True)
    print(xk, converged)


    ## PLOT
    import matplotlib.pyplot as plt
    # Grid on a square neighborhood
    #c, r = np.array([0.3, -0.3]), 0.85
    c, r = np.array([0, 0]), 1.2
    x1 = np.linspace(c[0]-r, c[0]+r, 200)
    x2 = np.linspace(c[1]-r, c[1]+r, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(np.array([X1, X2]))

    pts = {'$x_0$': np.array([0.25, 0.0]),
           'saddle': np.array([0.0, 0.0]),
           '$x^*$': np.array([1, -1]) / np.sqrt(2)}

    fig = plt.figure(figsize=(12, 5))

    # 3D surface + iterates
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.7)
    ax.contour(X1, X2, Z, levels=20, offset=Z.min(), cmap='viridis')
    for lab, p in pts.items():
        ax.scatter(*p, f(p), color='red', s=30)
        ax.text(*p, f(p), lab)
    ax.plot(x_list[:, 0], x_list[:, 1], f(x_list.T), 'b.-', lw=1.5, zorder=10)
    ax.set_xlabel('$x_1$'); ax.set_ylabel('$x_2$'); ax.set_zlabel('$f$')

    # Level curves + iterates
    ax2 = fig.add_subplot(1, 2, 2)
    cs = ax2.contour(X1, X2, Z, levels=20, cmap='viridis')
    ax2.clabel(cs, fontsize=7)
    for lab, p in pts.items():
        ax2.plot(*p, 'ro')
        ax2.annotate(lab, p, textcoords='offset points', xytext=(5, 5))
    ax2.plot(x_list[:, 0], x_list[:, 1], 'b.-', lw=1.5, ms=6, label='iterates')
    ax2.plot(*x_list[0], 'bs')
    ax2.plot(*x_list[-1], 'b*', ms=10)
    ax2.set_xlim(x1[0], x1[-1]); ax2.set_ylim(x2[0], x2[-1])
    ax2.set_aspect('equal')
    ax2.set_xlabel('$x_1$'); ax2.set_ylabel('$x_2$')
    ax2.legend()

    plt.tight_layout()

    plt.savefig("P6.png", dpi=200, bbox_inches='tight')
    plt.show()


