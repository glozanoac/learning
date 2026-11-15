/*
 * trisolve_tasks.c -- OpenMP Lesson 13, 2nd example: lower triangular solve L x = b
 * with task dependencies.
 * Sources: Saad (2003), "Iterative Methods for Sparse Linear Systems":
 *            Ch. 2 (5-point Laplacian), Ch. 3 (CSR), Ch. 11 (level scheduling);
 *          van der Pas, Stotzer & Terboven (2017), chapter on tasking.
 *
 * L = lower part (with diagonal) of the 5-point Laplacian on an m x m grid,
 * natural ordering r = j*m + i.  Row r needs x[r-1] (left) and x[r-m] (below).
 * The grid is cut into B x B tiles; a tile needs its left and lower tiles.
 *
 * Compile:  gcc -Wall -O2 -fopenmp -o trisolve_tasks trisolve_tasks.c -lm
 * Run:      OMP_NUM_THREADS=8 ./trisolve_tasks [m] [B]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef struct {
    long  n;      /* rows                                   */
    long *rp;     /* row pointers, n+1                      */
    int  *ci;     /* column indices; diagonal stored last   */
    double *va;   /* values                                 */
} csr_t;

/* Lower part of the 5-point Laplacian: -1 at (i,j-1), -1 at (i-1,j), 4 on the diagonal. */
static csr_t build_lower(int m)
{
    csr_t L;
    L.n  = (long)m * m;
    L.rp = malloc((L.n + 1) * sizeof(long));
    L.ci = malloc(3 * L.n * sizeof(int));
    L.va = malloc(3 * L.n * sizeof(double));
    long k = 0;
    for (int j = 0; j < m; j++)
        for (int i = 0; i < m; i++) {
            long r = (long)j * m + i;
            L.rp[r] = k;
            if (j > 0) { L.ci[k] = (int)(r - m); L.va[k++] = -1.0; }   /* below */
            if (i > 0) { L.ci[k] = (int)(r - 1); L.va[k++] = -1.0; }   /* left  */
            L.ci[k] = (int)r; L.va[k++] = 4.0;                         /* diagonal */
        }
    L.rp[L.n] = k;
    return L;
}

/* Forward substitution for one row: x[r] = (b[r] - sum L[r][c] x[c]) / L[r][r]. */
static inline void solve_row(const csr_t *L, const double *b, double *x, long r)
{
    double s = b[r];
    long last = L->rp[r + 1] - 1;                 /* the diagonal */
    for (long k = L->rp[r]; k < last; k++)
        s -= L->va[k] * x[L->ci[k]];
    x[r] = s / L->va[last];
}

/* All rows of tile (tx, ty), in natural order. */
static void solve_tile(const csr_t *L, const double *b, double *x,
                       int m, int B, int tx, int ty)
{
    int i1 = (tx + 1) * B < m ? (tx + 1) * B : m;
    int j1 = (ty + 1) * B < m ? (ty + 1) * B : m;
    for (int j = ty * B; j < j1; j++)
        for (int i = tx * B; i < i1; i++)
            solve_row(L, b, x, (long)j * m + i);
}

static void solve_serial(const csr_t *L, const double *b, double *x)
{
    for (long r = 0; r < L->n; r++)
        solve_row(L, b, x, r);
}

/* Level scheduling: tiles on the same anti-diagonal tx + ty = d are
   independent; a barrier separates consecutive diagonals.              */
static void solve_levels(const csr_t *L, const double *b, double *x, int m, int B)
{
    int nt = (m + B - 1) / B;
    #pragma omp parallel
    for (int d = 0; d <= 2 * (nt - 1); d++) {
        int t0 = d - (nt - 1) > 0 ? d - (nt - 1) : 0;
        int t1 = d < nt - 1 ? d : nt - 1;
        #pragma omp for schedule(dynamic)
        for (int tx = t0; tx <= t1; tx++)
            solve_tile(L, b, x, m, B, tx, d - tx);
        /* implicit barrier: diagonal d is finished */
    }
}

/* Tasks: tile (tx, ty) waits only for tiles (tx-1, ty) and (tx, ty-1).
   dep[][] has one extra row and column of never-written sentinels, so the
   tiles on the left and bottom edges need no special case.             */
static void solve_tasks(const csr_t *L, const double *b, double *x, int m, int B)
{
    int nt = (m + B - 1) / B;
    char (*dep)[nt + 1] = calloc((size_t)(nt + 1) * (nt + 1), 1);
    #pragma omp parallel
    #pragma omp single
    for (int ty = 0; ty < nt; ty++)
        for (int tx = 0; tx < nt; tx++) {
            #pragma omp task depend(in: dep[ty + 1][tx], dep[ty][tx + 1]) \
                             depend(out: dep[ty + 1][tx + 1])
            solve_tile(L, b, x, m, B, tx, ty);
        }
    free(dep);
}

static double max_diff(const double *a, const double *b, long n)
{
    double d = 0.0;
    for (long r = 0; r < n; r++)
        if (fabs(a[r] - b[r]) > d) d = fabs(a[r] - b[r]);
    return d;
}

int main(int argc, char **argv)
{
    int m = argc > 1 ? atoi(argv[1]) : 4000;
    int B = argc > 2 ? atoi(argv[2]) : 128;
    csr_t L = build_lower(m);
    double *b  = malloc(L.n * sizeof(double));
    double *x0 = malloc(L.n * sizeof(double));
    double *x  = malloc(L.n * sizeof(double));
    for (long r = 0; r < L.n; r++) { b[r] = 1.0; x0[r] = x[r] = 0.0; }  /* touch pages before timing */

    printf("grid %d x %d: n = %ld rows, nnz = %ld, tile %d x %d, %d x %d tiles, threads = %d\n",
           m, m, L.n, L.rp[L.n], B, B, (m + B - 1) / B, (m + B - 1) / B, omp_get_max_threads());

    double ts = 1e30, tl = 1e30, tt = 1e30;       /* best of 5 runs */
    for (int rep = 0; rep < 5; rep++) {
        double t = omp_get_wtime();
        solve_serial(&L, b, x0);
        ts = fmin(ts, omp_get_wtime() - t);

        t = omp_get_wtime();
        solve_levels(&L, b, x, m, B);
        tl = fmin(tl, omp_get_wtime() - t);

        t = omp_get_wtime();
        solve_tasks(&L, b, x, m, B);
        tt = fmin(tt, omp_get_wtime() - t);
    }
    printf("serial : %.3f s\n", ts);
    solve_levels(&L, b, x, m, B);
    printf("levels : %.3f s   speed-up %.2f   max |x - x_serial| = %.1e\n",
           tl, ts / tl, max_diff(x, x0, L.n));
    solve_tasks(&L, b, x, m, B);
    printf("tasks  : %.3f s   speed-up %.2f   max |x - x_serial| = %.1e\n",
           tt, ts / tt, max_diff(x, x0, L.n));
    return 0;
}
