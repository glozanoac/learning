# Scientific Programming

C, shared-memory parallelism with OpenMP, and HPC clusters with Slurm: the tools used to build `research/yupana`.

Analysis of the books:

```bibtex
@book{kernighan1988c,
  author    = {Kernighan, Brian W. and Ritchie, Dennis M.},
  title     = {The {C} Programming Language},
  edition   = {2nd},
  publisher = {Prentice Hall},
  address   = {Englewood Cliffs, NJ},
  year      = {1988},
  isbn      = {0-13-110362-8}
}
```

LESSONS: `01-10`
```bibtex
@book{chapman2007openmp,
  author    = {Chapman, Barbara and Jost, Gabriele and van der Pas, Ruud},
  title     = {Using {OpenMP}: Portable Shared Memory Parallel Programming},
  publisher = {MIT Press},
  address   = {Cambridge, MA},
  series    = {Scientific and Engineering Computation},
  year      = {2007},
  isbn      = {978-0-262-53302-7}
}
```


LESSONS: `12-`
```bibtex
@book{vanderpas2017openmp,
  author    = {van der Pas, Ruud and Stotzer, Eric and Terboven, Christian},
  title     = {Using {OpenMP} -- The Next Step: Affinity, Accelerators, Tasking, and {SIMD}},
  publisher = {MIT Press},
  address   = {Cambridge, MA},
  series    = {Scientific and Engineering Computation},
  year      = {2017},
  isbn      = {978-0-262-53478-9}
}
```

LESSONS: `-`
```bibtex
@book{deakin2023gpu,
  author    = {Deakin, Tom and Mattson, Timothy G.},
  title     = {Programming Your {GPU} with {OpenMP}: Performance Portability for {GPUs}},
  publisher = {MIT Press},
  address   = {Cambridge, MA},
  series    = {Scientific and Engineering Computation},
  year      = {2023},
  isbn      = {978-0-262-54753-6}
}
```

## Lessons

Each lesson folder has the notes (`.tex/.pdf`) and the C example(s). Build an example with `gcc -fopenmp -O2 file.c` and run it with `OMP_NUM_THREADS=4 ./a.out`.

| Folder | Topic |
|---|---|
| [`00-0-setup-local/`](./00-0-setup-local/GUIDE.md) | Local setup: C, OpenMP and GPU offload toolchain |
| [`00-1-slurm-cluster/`](./00-1-slurm-cluster/) | Ansible guides to provision a Slurm cluster (single controller; HA with accounting, login node and NFS) |
| `01-openmp-fork-join/` | Fork-join model, first parallel region |
| `02-openmp-shared-private/` | Shared and private variables |
| `03-openmp-sync-reduction/` | Race conditions, synchronization and reductions |
| `04-openmp-parallel-for/` | Parallel loops |
| `05-openmp-schedule/` | Loop scheduling (static, dynamic, guided) |
| `06-openmp-barrier-single-sections/` | Barrier, single and sections |
| `07-openmp-firstprivate-lastprivate/` | Data clauses: firstprivate and lastprivate |
| `08-openmp-false-sharing/` | False sharing |
| `09-openmp-locks/` | Locks |
| `10-openmp-thread-sanitizer/` | Finding data races with ThreadSanitizer |
| [`11-openmp-spmv-csr/`](./11-openmp-spmv-csr/spmv/) | Parallel sparse matrix-vector product (CSR) |
| `12-openmp-tasks/` | Tasks: Fibonacci and quicksort |
| `20-io/` | Sparse matrix storage and I/O: dense vs CSR, COO to CSR |

[<- Back to studies](../README.md)
