# Scientific Programming

C, shared-memory parallelism with OpenMP, and HPC clusters with Slurm: the tools used to build `research/yupana`.

> 0XX: Advanced C
> 1XX: CPU OpenMP
> 2XX: GPU OpenMP
> 3XX: others


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
| [`13-openmp-task-depend/`](./13-openmp-task-depend/) | Task dependencies (depend): two-stage pipeline and sparse triangular solve | van der Pas et al. (2017), Saad (2003) |
| [`14-openmp-extrae-paraver/`](./14-openmp-extrae-paraver/) | Tracing tasks with Extrae and Paraver (Gantt chart of Lesson 13) | van der Pas et al. (2017) |
| [`20-io/`](./20-io/) | Sparse matrix storage and I/O: dense vs CSR, COO to CSR | Saad (2003) |

[<- Back to studies](../README.md)
