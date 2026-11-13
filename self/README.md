# Self-Directed Learning

Everything I study outside the master's program.

## Structure

| Directory | Purpose |
|---|---|
| [`studies/`](./studies/) | Foundations, one track per subject (linear optimization, dynamic programming, scientific programming, mining): book notes, proofs and prototype code. |
| [`research/`](./research/) | Reproduction of state-of-the-art mine planning papers, and `yupana`, my research on numerical methods (C + OpenMP kernels). |

## How the pieces connect

```txt
studies/<math tracks>              studies/scientific-programming/
learn the theory, prove it,        learn C, OpenMP and HPC clusters,
prototype it in Python
   |                                     |
   v                                     v
research/<topic>/                  research/yupana/
reproduce mine planning   --(1)-->  (2) open a research line on that
papers                              numerical method and implement
                          <--(3)--  an efficient, stable kernel
```

1. A topic needs an optimized numerical method.
2. `yupana` studies that method from the literature and adds an efficient implementation to its core.
3. The topic uses the new kernel, which stays available for later topics.

[<- Back to repository root](../README.md)
