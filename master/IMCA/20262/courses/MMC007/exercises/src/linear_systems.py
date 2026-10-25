import numpy as np

def forward_substitution(L, b):
    """
    Solve Lx = b linear system

    [INPUT]
    L: lower triangular matrix
    b: array

    [OUTPUT]
    x: solution of system
    """
    n = L.shape[0]

    x = np.zeros(n)
    for i in range(n):
        x[i] = (b[i] - L[i, :i]@x[:i])/L[i, i]

    return x

def backward_substitution(U, b):
    """
    Solve Ux = b linear system

    [INPUT]
    U: lower triangular matrix
    b: array

    [OUTPUT]
    x: solution of system
    """
    n = U.shape[0]

    x = np.zeros(n)
    i = n-1
    while i >= 0:
        x[i] = (b[i] - U[i, i+1:]@x[i+1:])/U[i, i]
        i -= 1

    return x




if __name__=="__main__":
    L = np.array([
        [2, 0, 0],
        [1, 3, 0],
        [4, -1, 5]
        ])

    b = np.array([2, 7, 17])
    x = forward_substitution(L, b)
    print(x)

    U = np.array([
        [2, 1, 4],
        [0, 3, -1],
        [0, 0, 5]
        ])

    c = np.array([16, 3, 15])

    y = backward_substitution(U, c)
    print(y)


