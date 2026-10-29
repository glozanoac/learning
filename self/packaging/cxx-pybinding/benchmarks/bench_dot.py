"""Compare fastla kernels against NumPy across vector sizes."""
import timeit

import numpy as np

import fastla

rng = np.random.default_rng(0)

print(f"{'n':>10} | {'fastla.dot':>12} | {'numpy @':>12} | ratio")
print("-" * 52)
for n in [10, 1_000, 100_000, 10_000_000]:
    x, y = rng.standard_normal(n), rng.standard_normal(n)
    reps = max(3, 10_000_000 // n)
    t_c = min(timeit.repeat(lambda: fastla.dot(x, y), number=reps, repeat=5)) / reps
    t_np = min(timeit.repeat(lambda: x @ y, number=reps, repeat=5)) / reps
    print(f"{n:>10} | {t_c*1e6:>9.2f} us | {t_np*1e6:>9.2f} us | {t_c/t_np:.2f}x")
