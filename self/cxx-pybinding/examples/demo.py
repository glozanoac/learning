"""Use the C++ kernels from Python: correctness check + a steepest-descent loop."""
import time
import numpy as np
import fastla

rng = np.random.default_rng(0)
n = 1_000_000
x = rng.standard_normal(n)
y = rng.standard_normal(n)

# 1) Correctness against NumPy
assert np.isclose(fastla.dot(x, y), x @ y)
assert np.isclose(fastla.norm2(x), np.linalg.norm(x))
z = y.copy()
fastla.axpy(2.0, x, z)              # z <- 2x + z, in place
assert np.allclose(z, 2.0 * x + y)
print("dot / norm2 / axpy match NumPy")

# 2) Python lists are accepted too (converted to float64)
print("dot([1,2,3],[4,5,6]) =", fastla.dot([1, 2, 3], [4, 5, 6]))

# 3) Tiny optimization use: minimize f(w) = 0.5*||w - c||^2 with fixed step
c = rng.standard_normal(5)
w = np.zeros(5)
for k in range(100):
    g = w - c                        # gradient
    if fastla.norm2(g) < 1e-10:
        break
    fastla.axpy(-0.5, g, w)          # w <- w - 0.5 g
print(f"steepest descent: {k} iters, ||w - c|| = {fastla.norm2(w - c):.2e}")

# 4) Timing
t = time.perf_counter(); fastla.dot(x, y); t_c = time.perf_counter() - t
t = time.perf_counter(); x @ y;            t_np = time.perf_counter() - t
print(f"dot, n={n}: fastla {t_c*1e3:.2f} ms | numpy {t_np*1e3:.2f} ms")

# 5) Error handling: C++ exceptions become Python exceptions
try:
    fastla.dot(x, y[:10])
except ValueError as e:
    print("ValueError:", e)
