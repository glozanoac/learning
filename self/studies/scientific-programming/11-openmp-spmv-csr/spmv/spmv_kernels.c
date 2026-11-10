/*
 * spmv_kernels.c -- parallel CSR sparse matrix-vector product y = A x.
 *
 * All kernels compute the same thing (Algorithm 3 of the notes
 * "Matrix-Vector Multiplication y = Ax: Dense versus CSR Storage").
 * They differ only in HOW the work is split among threads:
 *
 *   static  : equal number of ROWS per thread            (Lesson 4)
 *   dynamic : chunks of rows handed out on demand        (Lesson 5)
 *   nnz     : equal number of NONZEROS per thread        (notes, Alg. 5)
 *   merge   : equal number of ROWS + NONZEROS per thread,
 *             long rows may be split between threads     (Merrill & Garland 2016)
 */
#include <stdlib.h>
#include <omp.h>
#include "spmv_kernels.h"

#define DYNAMIC_CHUNK 4096   /* rows per chunk; 1 would be far too fine for 10^8 rows */

/* ------------------------------------------------------------------ */
void spmv_serial(const csr_t *A, const double *x, double *y)
{
    const int64_t *rp = A->row_ptr; const int32_t *ci = A->col; const double *v = A->val;
    for (int64_t i = 0; i < A->M; i++) {
        double s = 0.0;
        for (int64_t k = rp[i]; k < rp[i + 1]; k++)
            s += v[k] * x[ci[k]];          /* indirect access (gather) to x */
        y[i] = s;
    }
}

/* ------------------------------------------------------------------ */
/* Rows are independent: each y[i] is written by exactly one iteration,
   so no synchronization is needed (notes, Alg. 4).                    */
void spmv_static(const csr_t *A, const double *x, double *y)
{
    const int64_t *rp = A->row_ptr; const int32_t *ci = A->col; const double *v = A->val;
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < A->M; i++) {
        double s = 0.0;
        for (int64_t k = rp[i]; k < rp[i + 1]; k++)
            s += v[k] * x[ci[k]];
        y[i] = s;
    }
}

void spmv_dynamic(const csr_t *A, const double *x, double *y)
{
    const int64_t *rp = A->row_ptr; const int32_t *ci = A->col; const double *v = A->val;
    #pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
    for (int64_t i = 0; i < A->M; i++) {
        double s = 0.0;
        for (int64_t k = rp[i]; k < rp[i + 1]; k++)
            s += v[k] * x[ci[k]];
        y[i] = s;
    }
}

/* ------------------------------------------------------------------ */
/* Smallest i in [0, M] with row_ptr[i] >= target (binary search).     */
static int64_t lower_bound(const int64_t *rp, int64_t M, int64_t target)
{
    int64_t lo = 0, hi = M;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (rp[mid] < target) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* nnz-balanced partition (notes, Alg. 5): thread t gets the contiguous
   rows start(t) .. start(t+1)-1, holding about nnz/T nonzeros. Each thread
   computes its own boundaries, so there is no shared partition array.
   Weakness: a single row longer than nnz/T cannot be split.            */
void spmv_nnz(const csr_t *A, const double *x, double *y)
{
    const int64_t *rp = A->row_ptr; const int32_t *ci = A->col; const double *v = A->val;
    #pragma omp parallel
    {
        int T = omp_get_num_threads(), t = omp_get_thread_num();
        int64_t r0 = (t == 0)     ? 0    : lower_bound(rp, A->M, A->nnz * t / T);
        int64_t r1 = (t == T - 1) ? A->M : lower_bound(rp, A->M, A->nnz * (t + 1) / T);
        for (int64_t i = r0; i < r1; i++) {
            double s = 0.0;
            for (int64_t k = rp[i]; k < rp[i + 1]; k++)
                s += v[k] * x[ci[k]];
            y[i] = s;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Merge-path SpMV (Merrill & Garland, SC16).
 *
 * Think of the work as a path of M + nnz steps: one step per nonzero
 * (multiply-add) and one step per row end (store y[i]). Cutting this path
 * into T equal pieces balances BOTH kinds of work: many short rows
 * (per-row overhead) and a few huge rows (per-nonzero work). A piece may
 * start or end in the middle of a row; the partial sum of an unfinished
 * row is kept as a "carry" and added after the parallel region.         */

/* Find where diagonal d of the merge path crosses: returns the number of
   finished rows (*row) and of consumed nonzeros (*k), with *row + *k = d. */
static void merge_search(const int64_t *rp, int64_t M, int64_t nnz,
                         int64_t d, int64_t *row, int64_t *k)
{
    int64_t lo = (d > nnz) ? d - nnz : 0;
    int64_t hi = (d < M)   ? d       : M;
    while (lo < hi) {
        int64_t mid = lo + (hi - lo) / 2;
        if (rp[mid + 1] <= d - mid - 1) lo = mid + 1;  /* row mid ends before this point */
        else                            hi = mid;
    }
    *row = lo;
    *k   = d - lo;
}

void spmv_merge(const csr_t *A, const double *x, double *y)
{
    const int64_t *rp = A->row_ptr; const int32_t *ci = A->col; const double *v = A->val;
    int T = omp_get_max_threads();
    int64_t *carry_row = malloc((size_t)T * sizeof(int64_t));
    double  *carry_val = malloc((size_t)T * sizeof(double));
    int64_t total = A->M + A->nnz;

    #pragma omp parallel num_threads(T)
    {
        int t = omp_get_thread_num();
        int64_t row, k, row_end, k_end;
        merge_search(rp, A->M, A->nnz, total * t / T,       &row,     &k);
        merge_search(rp, A->M, A->nnz, total * (t + 1) / T, &row_end, &k_end);

        /* rows that END inside my piece: compute and store */
        double s = 0.0;
        for (; row < row_end; row++) {
            for (; k < rp[row + 1]; k++)
                s += v[k] * x[ci[k]];
            y[row] = s;             /* may be only the tail of the row:  */
            s = 0.0;                /* the earlier part arrives as a carry */
        }
        /* the row that is still open at the end of my piece */
        for (; k < k_end; k++)
            s += v[k] * x[ci[k]];
        carry_row[t] = row_end;
        carry_val[t] = s;
    }   /* implicit barrier: every y[row] above is written */

    /* fix-up: add the partial sums of rows split between threads (O(T)) */
    for (int t = 0; t < T; t++)
        if (carry_row[t] < A->M)
            y[carry_row[t]] += carry_val[t];

    free(carry_row);
    free(carry_val);
}
