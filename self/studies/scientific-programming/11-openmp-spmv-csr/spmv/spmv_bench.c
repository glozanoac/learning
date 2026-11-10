/*
 * spmv_bench.c -- load a binary CSR matrix, check and time the SpMV kernels.
 *
 * Usage:   OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores \
 *          ./spmv_bench matrix.csr [repetitions]
 *
 * x[j] = (j mod 1000) + 1, so for the 6x6 example of the notes x = (1..6)
 * and the result must be y = (20, 6, 0, 27, 4, 56).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "csr.h"
#include "spmv_kernels.h"

typedef void (*kernel_fn)(const csr_t *, const double *, double *);

static void print_stats(const csr_t *A)
{
    int64_t maxlen = 0, empty = 0;
    #pragma omp parallel for reduction(max:maxlen) reduction(+:empty)
    for (int64_t i = 0; i < A->M; i++) {
        int64_t len = A->row_ptr[i + 1] - A->row_ptr[i];
        if (len > maxlen) maxlen = len;
        if (len == 0) empty++;
    }
    printf("matrix   : %lld x %lld, nnz = %lld\n",
           (long long)A->M, (long long)A->N, (long long)A->nnz);
    printf("rows     : avg %.2f nnz/row, max %lld (%.2f%% of nnz), %lld empty\n",
           (double)A->nnz / A->M, (long long)maxlen,
           100.0 * maxlen / A->nnz, (long long)empty);
    printf("memory   : CSR %.2f GB + x,y %.2f GB\n",
           (12.0 * A->nnz + 8.0 * (A->M + 1)) / 1e9, 8.0 * (A->M + A->N) / 1e9);
}

/* Smallest i with row_ptr[i] >= target (same rule as spmv_nnz). */
static int64_t lb(const int64_t *rp, int64_t M, int64_t target)
{
    int64_t lo = 0, hi = M;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (rp[mid] < target) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* Load imbalance = (work of the busiest thread) / (average work), where
   work = rows + nonzeros (one step per row end and per multiply-add).
   1.00 is perfect; 2.00 means the busiest thread does twice the average,
   so the kernel cannot be faster than half the ideal speed-up.         */
static void print_imbalance(const csr_t *A, int T)
{
    const int64_t *rp = A->row_ptr;
    double avg = (double)(A->M + A->nnz) / T;
    double worst_static = 0.0, worst_nnz = 0.0;
    int64_t chunk = (A->M + T - 1) / T;            /* GCC's static split */
    for (int t = 0; t < T; t++) {
        int64_t r0 = t * chunk < A->M ? t * chunk : A->M;
        int64_t r1 = (t + 1) * chunk < A->M ? (t + 1) * chunk : A->M;
        double w = (double)(r1 - r0) + (double)(rp[r1] - rp[r0]);
        if (w > worst_static) worst_static = w;

        r0 = (t == 0)     ? 0    : lb(rp, A->M, A->nnz * t / T);
        r1 = (t == T - 1) ? A->M : lb(rp, A->M, A->nnz * (t + 1) / T);
        w = (double)(r1 - r0) + (double)(rp[r1] - rp[r0]);
        if (w > worst_nnz) worst_nnz = w;
    }
    printf("imbalance: static %.2f, nnz %.2f, merge 1.00  (busiest thread / average, T = %d)\n",
           worst_static / avg, worst_nnz / avg, T);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s matrix.csr [repetitions]\n", argv[0]);
        return 1;
    }
    int reps = (argc > 2) ? atoi(argv[2]) : 10;

    csr_t A;
    double t = omp_get_wtime();
    if (csr_read(argv[1], &A) != 0) return 1;
    printf("loaded %s in %.2f s, threads = %d\n", argv[1],
           omp_get_wtime() - t, omp_get_max_threads());
    print_stats(&A);
    print_imbalance(&A, omp_get_max_threads());

    double *x    = malloc((size_t)A.N * sizeof(double));
    double *y    = malloc((size_t)A.M * sizeof(double));
    double *yref = malloc((size_t)A.M * sizeof(double));
    if (!x || !y || !yref) { fprintf(stderr, "out of memory\n"); return 1; }

    /* first touch of x and y, spread over the threads */
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < A.N; j++) x[j] = (double)(j % 1000) + 1.0;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < A.M; i++) { y[i] = 0.0; yref[i] = 0.0; }

    t = omp_get_wtime();
    spmv_serial(&A, x, yref);
    double t_serial = omp_get_wtime() - t;

    if (A.M <= 10) {                      /* tiny matrix: show the result */
        printf("y        :");
        for (int64_t i = 0; i < A.M; i++) printf(" %g", yref[i]);
        printf("\n");
    }

    /* Bytes moved by one product, best case (notes, Sec. 4.2), with a
       64-bit row_ptr: values + col + row_ptr + x + y.                     */
    double bytes = 12.0 * A.nnz + 8.0 * (A.M + 1) + 8.0 * A.N + 8.0 * A.M;
    double flops = 2.0 * A.nnz;

    const char *names[]   = { "static", "dynamic", "nnz", "merge" };
    kernel_fn   kernels[] = { spmv_static, spmv_dynamic, spmv_nnz, spmv_merge };

    printf("\nserial reference: %.4f s\n\n", t_serial);
    printf("%-8s %10s %10s %9s %9s %10s\n",
           "kernel", "best (s)", "mean (s)", "GFLOP/s", "GB/s", "max error");
    for (int m = 0; m < 4; m++) {
        kernels[m](&A, x, y);             /* warm-up, and the copy we check */

        double err = 0.0, ymax = 0.0;
        #pragma omp parallel for reduction(max:err, ymax)
        for (int64_t i = 0; i < A.M; i++) {
            double d = fabs(y[i] - yref[i]);
            if (d > err) err = d;
            if (fabs(yref[i]) > ymax) ymax = fabs(yref[i]);
        }
        double rel = (ymax > 0.0) ? err / ymax : err;

        double best = 1e30, sum = 0.0;
        for (int r = 0; r < reps; r++) {
            t = omp_get_wtime();
            kernels[m](&A, x, y);
            t = omp_get_wtime() - t;
            sum += t;
            if (t < best) best = t;
        }
        printf("%-8s %10.4f %10.4f %9.2f %9.2f %10.2e%s\n", names[m], best, sum / reps,
               flops / best / 1e9, bytes / best / 1e9, rel, rel > 1e-12 ? "  WRONG" : "");
    }

    free(x); free(y); free(yref);
    csr_free(&A);
    return 0;
}
