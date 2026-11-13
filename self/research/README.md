# Research & Paper Reproductions

> **I AM PUSHING 'studies/' PATH FIRST - I HOPE THAT IN 6 MONTHS (2027 April) I WILL BE PUSHING THIS BRANCH**

This directory is dedicated to studying, deconstructing, and reproducing state-of-the-art (SOTA) research articles in open-pit mining optimization and operational research, with a focus on mine planning.

## Numerical Base: yupana

`yupana/` is itself a research project on numerical optimization, developed from scratch: a C + OpenMP library of numerical kernels (sparse matrix storage, SpMV, iterative solvers, dynamic and stochastic programming building blocks), where methods from the literature are studied, refined, and made fast and numerically stable.

It is also expected to serve as the base for the numerical computations of the mining research topics in this directory: the kernels handle the heavy numerical work, and each topic only describes its problem and algorithm on top of them.

## Directory Structure

Each research topic is organized by problem domain:

```txt
research/
|- yupana/                   # Numerical base library (C + OpenMP)
    |- include/, src/        # C kernels
    |- bindings/             # Python bindings (Pybind11 / Nanobind) used by all topics
    |- papers/               # Research behind the kernels, by numerical topic
    |- test/, bench/
|- <topic_name>/             # e.g., open_pit_scheduling, fleet_dispatch
    |- papers/               # Paper breakdowns, math derivations, and reading notes
    |- benchmarks/           # Dataset instances (e.g., standard block models)
    |- src/                  # Reproduction of the paper (Python calling yupana): experiments, validation, visualization
```

See [`yupana/README.md`](./yupana/README.md) for build instructions.

[<- Back to self](../README.md)
