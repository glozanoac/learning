# Research & Paper Reproductions

This directory is dedicated to studying, deconstructing, and reproducing state-of-the-art (SOTA) research articles in open-pit mining optimization and operational research.

While the `studies/` directory focuses on foundational textbooks and core theoretical proofs, this space serves as an applied laboratory.

## Directory Structure

Each research topic is organized by problem domain:

```txt
research/
|- <topic_name>/             # e.g., open_pit_scheduling, fleet_dispatch
    |- papers/               # Paper breakdowns, math derivations, and reading notes
    |- benchmarks/           # Dataset instances (e.g., standard block models)
    |- src/                  # Code implementations
        |- python/           # Paper validation, visualization, and prototype logic
        |- cpp/              # High-performance C++ implementation of paper algorithms
        |- bindings/         # Pybind11 / Nanobind integration for benchmarking
```
