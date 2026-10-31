import numpy as np
from projected_gradient import projected_gradient_basic

if __name__=="__main__":
    # f(x, y) = 2*x**2 + y**2 -2*x*y +2*x -4*y
    def df(x): return np.array([4*x[0] - 2*x[1] + 2, 2*x[1] - 2*x[0] - 4])
    l, u = 0, 2
    def proj(y): return np.minimum(np.maximum(y, l), u)

    alpha=1.0/5
    epsilon=1e-6
    x0 = np.array([0.0, 0.0])
    x, conv, (x_list, proj_list) = projected_gradient_basic(df, proj, x0, alpha, epsilon, k=20, debug = True, collect = True)

    print(x, conv)

    # PLOT
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    def f(x, y): return 2*x**2 + y**2 - 2*x*y + 2*x - 4*y
    x_bar = np.array([0.5, 2.0])    # constrained minimizer
    x_star = np.array([1.0, 3.0])   # unconstrained minimizer

    fig, ax = plt.subplots(figsize=(6, 7))

    # box C
    ax.add_patch(Rectangle((l, l), u - l, u - l, fc=(0.9, 0.9, 1.0), ec="blue", lw=1.5, zorder=1,
                           label=r"$C=[0,2]^2$"))

    # level curves through the iterates and through f(x_bar)
    X, Y = np.meshgrid(np.linspace(-0.6, 2.3, 400), np.linspace(-0.2, 3.3, 400))
    levels = np.unique(np.round(np.r_[f(x_list[:, 0], x_list[:, 1]), f(*x_bar)], 6))
    ax.contour(X, Y, f(X, Y), levels=levels, colors="gray", linewidths=0.7, linestyles="solid", zorder=2)

    # projected iterates x_k (blue)
    ax.plot(x_list[:, 0], x_list[:, 1], "-o", color="blue", ms=4, lw=1.5, zorder=4,
            label=rf"$x_k = P_C(y_k)$, $\alpha={alpha:.3g}$")

    # unprojected points y_k outside C (red circles) and segment y_k -> x_k
    first = True
    for i in range(1, len(x_list)):
        y, xk = proj_list[i], x_list[i]
        if np.linalg.norm(y - xk) > 1e-12:
            ax.plot([xk[0], y[0]], [xk[1], y[1]], "--", color="red", lw=0.8, zorder=3)
            ax.plot(y[0], y[1], "o", mfc="none", mec="red", ms=5, zorder=5,
                    label=r"$y_k = x_{k-1} - \alpha\nabla f(x_{k-1})$" if first else None)
            first = False

    # constrained minimizer, unconstrained minimizer and x0
    ax.plot(*x_bar, "kx", ms=9, mew=2, zorder=6, label=r"$\bar{x}=(1/2,\,2)$ (constrained)")
    ax.plot(*x_star, "*", color="green", ms=12, zorder=6, label=r"$x^*=(1,\,3)$ (unconstrained)")
    ax.annotate(r"$\bar{x}$", x_bar, xytext=(2, 6), textcoords="offset points")
    ax.annotate(r"$x^*$", x_star, xytext=(6, 4), textcoords="offset points")
    ax.annotate(r"$x_0$", x0, xytext=(5, -12), textcoords="offset points")

    ax.set_xlim(-0.6, 2.3)
    ax.set_ylim(-0.2, 3.3)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(r"Projected gradient on $C=[0,2]^2$, $f(x)=2x_1^2+x_2^2-2x_1x_2+2x_1-4x_2$", fontsize=9)
    plt.tight_layout()
    plt.savefig("P9-projected_gradient.png", dpi=200)
    plt.show()
