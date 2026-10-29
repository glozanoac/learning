# QUESTION: P20
# SECTION: U19. Krylov subspaces and rate of convergence of CG

import pandas as pd
import numpy as np
from linear_cg import linear_cg

np.set_printoptions(precision=3, suppress=True, floatmode='fixed')

if __name__=="__main__":
    A = np.diag([1.0+11.0*i for i in range(10)])
    b = np.ones(10)
    init = x0 = np.zeros(10)
    M = np.identity(10)
    x_star = np.linalg.inv(A)@b

    normA = lambda x: np.sqrt(x@A@x)

    epsilon=1e-4
    x_cg, converged, cg_path = linear_cg(A, b, x0, M, epsilon=epsilon, k=20, debug=True, collect=True)
    print("="*20 + " CG " + "="*20)
    print(x_cg, converged, normA(x_cg - x_star))

    # steepest descent
    sd_path = x0
    k = 30
    while k > 0:
        g = A@x0 -b
        if np.linalg.norm(g) < epsilon*np.linalg.norm(b):
            break
        p = -g
        alpha = (g@g)/(g@A@g)
        x1 = x0 + alpha*p

        sd_path = np.vstack([sd_path, x1])

        k -= 1
        x0 = x1

    x_sd = x1

    print("="*20 + " SD " + "="*20)
    print(x_sd, k, normA(x_sd - x_star))

    ## BUILD TABLE TO COMPARE CONVEGENCY RATE

    # compute table
    cg_rate = np.apply_along_axis(normA, 1, cg_path-x_star)/normA(init-x_star)
    sd_rate = np.apply_along_axis(normA, 1, sd_path-x_star)/normA(init-x_star)


    #fill with zeros to match size
    n = max(len(cg_rate), len(sd_rate))
    cg_rate = np.pad(cg_rate, (0, n-len(cg_rate)))
    sd_rate = np.pad(sd_rate, (0, n-len(sd_rate)))

    cg_bound = 2*(9/11)**np.arange(n)
    sd_bound = (99/101)**np.arange(n)

    conv = pd.DataFrame(np.column_stack([np.arange(n), cg_rate, cg_bound, sd_rate, sd_bound]), columns=["k", "cg", "cg_bound", "sd", "sd_bound"])

    # PLOT
    import matplotlib.pyplot as plt

    k = conv["k"]
    floor = 1e-6                                   # lower y-limit, like the figure

    # log scale can't show 0 → mask exact zeros, clip tiny values to the floor
    cg = conv["cg"].where(conv["cg"] > 0)
    cg = cg.clip(lower=floor)

    fig, ax = plt.subplots(figsize=(9, 4))

    ax.semilogy(k, cg, "o-", color="blue", lw=2.5, ms=6, label="CG")
    ax.semilogy(k, conv["cg_bound"], "--", color="blue", lw=0.8,
                label=r"$2\left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$")
    ax.semilogy(k, conv["sd"], "o-", color="firebrick", lw=1.5, ms=4,
                mfc="lightcoral", label="steepest descent")
    ax.semilogy(k, conv["sd_bound"], "--", color="lightcoral", lw=1.2,
                label=r"$\left(\frac{\kappa-1}{\kappa+1}\right)^k$")

    ax.set_xlim(k.min(), k.max())
    ax.set_ylim(floor, 3)
    ax.set_xlabel(r"$k$", fontsize=14)
    ax.set_ylabel(r"$\|x_k - x^*\|_A \,/\, \|x_0 - x^*\|_A$", fontsize=13)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
              fancybox=False, edgecolor="black")

    plt.tight_layout()
    plt.savefig("P20-cg_conv_rate.png", bbox_inches="tight")
    plt.show()

