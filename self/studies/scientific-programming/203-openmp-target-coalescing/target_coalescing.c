/*
 * target_coalescing.c -- OpenMP Lesson 33: the order of memory accesses on
 * the GPU (coalescing).
 * Source: Deakin & Mattson (2023), Ch. 5 (Using a GPU Effectively),
 *         Sec. 5.2.7 (Memory Coalescence).
 *
 * A is an N x N matrix stored by rows: A[r*N + c].  Two kernels read all of A:
 *   row sums    : thread r sums row r     -> neighbour threads are N apart
 *   column sums : thread c sums column c  -> neighbour threads are adjacent
 * Both run on the GPU and, for comparison, on the CPU.
 *
 * Compile:  clang -O2 -fopenmp --offload-arch=sm_61 \
 *                 -o target_coalescing target_coalescing.c
 * Run:      OMP_TARGET_OFFLOAD=MANDATORY ./target_coalescing [N]
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* ---- GPU ---------------------------------------------------------------- */
static void row_sums_gpu(const double *A, double *s, long N)
{
    #pragma omp target teams distribute parallel for \
            map(present, alloc: A[0:N*N], s[0:N])
    for (long r = 0; r < N; r++) {
        double sum = 0.0;
        for (long k = 0; k < N; k++)
            sum += A[r * N + k];          /* thread r+1 reads N elements further */
        s[r] = sum;
    }
}

static void col_sums_gpu(const double *A, double *s, long N)
{
    #pragma omp target teams distribute parallel for \
            map(present, alloc: A[0:N*N], s[0:N])
    for (long c = 0; c < N; c++) {
        double sum = 0.0;
        for (long k = 0; k < N; k++)
            sum += A[k * N + c];          /* thread c+1 reads the next element */
        s[c] = sum;
    }
}

/* ---- CPU: the same loops ------------------------------------------------ */
static void row_sums_cpu(const double *A, double *s, long N)
{
    #pragma omp parallel for
    for (long r = 0; r < N; r++) {
        double sum = 0.0;
        for (long k = 0; k < N; k++)
            sum += A[r * N + k];
        s[r] = sum;
    }
}

static void col_sums_cpu(const double *A, double *s, long N)
{
    #pragma omp parallel for
    for (long c = 0; c < N; c++) {
        double sum = 0.0;
        for (long k = 0; k < N; k++)
            sum += A[k * N + c];
        s[c] = sum;
    }
}

/* time one kernel; on the GPU copy s back to check it (not timed) */
static void run(const char *name, void (*f)(const double *, double *, long),
                const double *A, double *s, long N, int gpu)
{
    for (long i = 0; i < N; i++) s[i] = 0.0;
    if (gpu) {
        #pragma omp target update to(s[0:N])
    }
    double t = omp_get_wtime();
    f(A, s, N);
    t = omp_get_wtime() - t;
    if (gpu) {
        #pragma omp target update from(s[0:N])
    }
    long bad = 0;
    for (long i = 0; i < N; i++) bad += (s[i] != (double)N);
    printf("%-16s: %8.1f ms  %7.1f GB/s   correct: %s\n",
           name, 1e3 * t, 8.0 * N * N / t / 1e9, bad ? "NO" : "yes");
}

int main(int argc, char **argv)
{
    long N = argc > 1 ? atol(argv[1]) : 8192;        /* 8192^2 doubles = 512 MiB */
    double *A = malloc(N * N * sizeof(double));
    double *s = malloc(N * sizeof(double));
    for (long i = 0; i < N * N; i++) A[i] = 1.0;     /* every sum is N */

    printf("N = %ld: matrix of %.0f MiB, stored by rows\n",
           N, N * N * 8.0 / (1 << 20));

    #pragma omp target enter data map(to: A[0:N*N]) map(alloc: s[0:N])
    row_sums_gpu(A, s, N);                           /* warm-up, not timed */
    run("gpu row sums", row_sums_gpu, A, s, N, 1);
    run("gpu column sums", col_sums_gpu, A, s, N, 1);
    #pragma omp target exit data map(delete: A[0:N*N], s[0:N])

    run("cpu row sums", row_sums_cpu, A, s, N, 0);
    run("cpu column sums", col_sums_cpu, A, s, N, 0);

    free(A); free(s);
    return 0;
}
