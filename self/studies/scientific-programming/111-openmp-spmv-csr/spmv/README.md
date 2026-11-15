# Parallel CSR SpMV (y = A x) with OpenMP

Sparse matrix-vector product for large SuiteSparse graphs (10^8 rows,
10^8-10^9 nonzeros) on a single shared-memory machine. Design and results:
`spmv_design.pdf`.

## Files

| File | Purpose |
|---|---|
| `csr.h`, `csr.c` | CSR type (64-bit `row_ptr`, 32-bit `col`) and binary file I/O |
| `mtx2csr.c` | Matrix Market -> binary CSR converter (run once per matrix) |
| `spmv_kernels.c` | serial reference + 4 parallel kernels: static, dynamic, nnz, merge |
| `spmv_bench.c` | loads a `.csr`, checks every kernel against the serial result, times them |
| `gen_graph.c` | synthetic symmetric graph with hub rows, for testing |
| `example.mtx` | 6x6 running example of the notes; y must be `20 6 0 27 4 56` |

## Build and test

    make
    make test          # prints y : 20 6 0 27 4 56 and max error 0 for every kernel

## Run on mawi_201512020130 (laptop, 32 GB)

128,568,730 rows, 270,234,840 nonzeros, symmetric, integer values.
Needs about 4.3 GB for the CSR matrix plus 2.1 GB for x and y.

    wget https://suitesparse-collection-website.herokuapp.com/MM/MAWI/mawi_201512020130.tar.gz
    tar xzf mawi_201512020130.tar.gz
    ./mtx2csr mawi_201512020130/mawi_201512020130.mtx mawi_130.csr    # once, a few minutes

    export OMP_PROC_BIND=close OMP_PLACES=cores
    for t in 1 2 4 $(nproc); do OMP_NUM_THREADS=$t ./spmv_bench mawi_130.csr 10; done

Delete the `.mtx` and the `.tar.gz` afterwards if disk space is short: the
`.csr` file is all the benchmark needs.

## Output columns

* `imbalance`: work (rows + nonzeros) of the busiest thread divided by the
  average, for each partition. 1.00 is perfect.
* `best (s)`, `mean (s)`: over the repetitions (default 10, after one warm-up).
* `GFLOP/s` = 2 nnz / time. `GB/s` = best-case bytes / time, with
  bytes = 12 nnz + 8 (M+1) + 8 N + 8 M.
* `max error`: max |y - y_serial| / max |y_serial|; `WRONG` is printed above 1e-12.
