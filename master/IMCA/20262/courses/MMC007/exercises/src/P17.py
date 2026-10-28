#QUESTION: P17
#UNIT: U18. Conjugate directions and linear conjugate gradient
import numpy as np
np.set_printoptions(precision=3, suppress=True, floatmode='fixed') # show only three decimals

A = np.array([[4, 1], [1, 3]], dtype=np.float32)
b = np.array([1, 2], dtype=np.float32)

# objetive function
def phi(x): return x@A@x/2 - b@x

x_star = np.array([1.0/11, 7.0/11]) # solution of Ax=b
x0 = np.array([0, 0], dtype=np.float32)


def steps(directions, x0):
    for i, p in enumerate(directions):
        alpha = - ((A@x0 - b)@p) / (p@A@p)
        x1 = x0 + alpha*p

        print(f"step={i:2d}, x0={x0},x1={x1}, p={p}, alpha={alpha:8.3f},gap={phi(x1) -phi(x_star):10.3e}, df(x0)@p={(A@x0-b)@p:5.2f}, df(x1)={A@x1-b}")

        x0 = x1

    print(f"xn={x1}, ||xn-x*||={np.linalg.norm(x1 - x_star)}")

# cyclic coordinate directions
print("="*20)
print("[ITEM A] Cyclic coordinate directions")

directions = [
    np.array([1, 0]), # e1
    np.array([0, 1]), # e2

    np.array([1, 0]), # e1
    np.array([0, 1]), # e2

    np.array([1, 0]), # e1
    np.array([0, 1]), # e2

    np.array([1, 0]), # e1
    np.array([0, 1]), # e2


]

steps(directions, x0)

print("="*20)
print("[ITEM B] Conjugated directions")

directions = [
    np.array([1.0, 0.0]), # e1
    np.array([-1.0/4, 1.0])
]


steps(directions, x0)



print("="*20)
print("[ITEM C] Steepest descent directions")
k = 4
p0 = np.array([0, 0])
while k>= 0:
    p1=-(A@x0-b)

    alpha = - ((A@x0 - b)@p1) / (p1@A@p1)
    x1 = x0 + alpha*p1


    print(f"step={k:2d}, x0={x0},x1={x1}, p=p1={p1}, p1@p0={p1@p0} , alpha={alpha:8.3f},gap={phi(x1) -phi(x_star):10.3e}, df(x0)@p1={(A@x0-b)@p1:5.2f}, df(x1)={A@x1-b}")

    x0 = x1
    p0 = p1
    k -= 1


print(f"xn={x1}, ||xn-x*||={np.linalg.norm(x1 - x_star)}")
