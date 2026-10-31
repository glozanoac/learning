import numpy as np

def projected_gradient_basic(df, proj, x0, alpha, epsilon, k, debug: bool = False, collect: bool = False):
    x0 = proj(x0)

    x_list = x0
    proj_list = x0
    while k > 0:
        y = x0 - alpha*df(x0)
        x1 = proj(y)
        if np.linalg.norm(x1-x0) < epsilon:
            return x1, 1, (x_list, proj_list)

        print(f"[k={k}] x0={x0}, x1={x1}, y={y}, ||x1-x0||={np.linalg.norm(x1-x0)}, epsilon={epsilon}")
        if collect:
            x_list = np.vstack([x_list, x1])
            proj_list = np.vstack([proj_list, y])

        x0 = x1
        k -= 1

    return x1, 0, (x_list, proj_list)
