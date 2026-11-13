# yupana

A C library of fast OpenMP kernels for dynamic programming and stochastic
programming. The goal is to learn and refine numerical methods from the
literature, keeping them stable and efficient, with performance close to
standard libraries.

> *Yupana*: the Inca calculating tool.

## Layout

| Directory  | Contents                                              |
|------------|-------------------------------------------------------|
| `include/` | Public headers (`csr.h`, `spmv_kernels.h`)            |
| `src/`     | Kernel implementations                                |
| `test/`    | Correctness checks against NumPy/SciPy                |
| `bench/`   | Timings and scaling experiments                       |
| `papers/`  | Notes and papers by topic: CSR, SpMV, DP, solvers, stochastic |

## Build

Requires CMake >= 3.16 and a C compiler with OpenMP support.

```bash
cmake -S . -B build
cmake --build build
OMP_NUM_THREADS=4 ./build/hello
```
