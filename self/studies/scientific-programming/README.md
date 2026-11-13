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

LESSONS: `20-`
```bibtex
@book{saad2003iterative,
  author    = {Saad, Yousef},
  title     = {Iterative Methods for Sparse Linear Systems},
  edition   = {2nd},
  publisher = {SIAM},
  address   = {Philadelphia, PA},
  year      = {2003},
  isbn      = {978-0-89871-534-7}
}
```

## Lessons

Each lesson folder has the notes (`.tex/.pdf`) and the C example(s).

| Folder | Topic | Book |
|---|---|---|
| [`00-0-setup-local/`](./00-0-setup-local/GUIDE.md) | Local setup: C, OpenMP and GPU offload toolchain | All four books |
| [`00-1-slurm-cluster/`](./00-1-slurm-cluster/) | Ansible guides to provision a Slurm cluster (single controller; HA with accounting, login node and NFS) | - |
| [`01-openmp-fork-join/`](./01-openmp-fork-join/) | Fork-join model, first parallel region | Chapman et al. (2007) |
| [`02-openmp-shared-private/`](./02-openmp-shared-private/) | Shared and private variables | Chapman et al. (2007) |
| [`03-openmp-sync-reduction/`](./03-openmp-sync-reduction/) | Race conditions, synchronization and reductions | Chapman et al. (2007) |
| [`04-openmp-parallel-for/`](./04-openmp-parallel-for/) | Parallel loops | Chapman et al. (2007) |
| [`05-openmp-schedule/`](./05-openmp-schedule/) | Loop scheduling (static, dynamic, guided) | Chapman et al. (2007) |
| [`06-openmp-barrier-single-sections/`](./06-openmp-barrier-single-sections/) | Barrier, single and sections | Chapman et al. (2007) |
| [`07-openmp-firstprivate-lastprivate/`](./07-openmp-firstprivate-lastprivate/) | Data clauses: firstprivate and lastprivate | Chapman et al. (2007) |
| [`08-openmp-false-sharing/`](./08-openmp-false-sharing/) | False sharing | Chapman et al. (2007) |
| [`09-openmp-locks/`](./09-openmp-locks/) | Locks | Chapman et al. (2007) |
| [`10-openmp-thread-sanitizer/`](./10-openmp-thread-sanitizer/) | Finding data races with ThreadSanitizer | Chapman et al. (2007) |
| [`11-openmp-spmv-csr/`](./11-openmp-spmv-csr/spmv/) | Parallel sparse matrix-vector product (CSR) | Chapman et al. (2007), van der Pas et al. (2017) |
| [`12-openmp-tasks/`](./12-openmp-tasks/) | Tasks: Fibonacci and quicksort | van der Pas et al. (2017) |
| [`20-io/`](./20-io/) | Sparse matrix storage and I/O: dense vs CSR, COO to CSR | Saad (2003) |

[<- Back to studies](../README.md)
