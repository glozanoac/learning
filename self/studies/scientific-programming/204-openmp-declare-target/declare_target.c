/*
 * declare_target.c -- OpenMP Lesson 204: functions and global variables on
 * the GPU with declare target.
 * Source: Deakin & Mattson (2023), Appendix A.2 (Directives and Constructs
 *         for the GPU).
 *
 * f(x) = scale * x * x + 1 is called inside a GPU kernel.  f and the global
 * variable scale are in a declare target block, so the compiler builds a
 * device version of f and keeps a device copy of scale.
 *
 * Compile:  clang -O2 -fopenmp --offload-arch=sm_61 \
 *                 -o declare_target declare_target.c
 * Run:      OMP_TARGET_OFFLOAD=MANDATORY ./declare_target
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#pragma omp declare target
double scale = 2.0;                    /* one copy on the host, one on the GPU */

double f(double x)                     /* compiled for the host and the GPU    */
{
    return scale * x * x + 1.0;
}
#pragma omp end declare target

/* y[i] = f(x[i]) on the GPU */
static void apply_gpu(const double *x, double *y, long n)
{
    #pragma omp target teams distribute parallel for \
            map(to: x[0:n]) map(from: y[0:n])
    for (long i = 0; i < n; i++)
        y[i] = f(x[i]);
}

/* compare with f on the host, which uses the host copy of scale */
static void check(const char *step, const double *x, const double *y, long n)
{
    long bad = 0;
    for (long i = 0; i < n; i++)
        bad += (y[i] != f(x[i]));
    printf("%-34s host scale = %.1f   y[1] = %5.1f   correct: %s\n",
           step, scale, y[1], bad ? "NO" : "yes");
}

int main(void)
{
    long n = 1L << 20;
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) x[i] = (double)(i % 8);

    int on_host = 1;
    #pragma omp target map(from: on_host)
    on_host = omp_is_initial_device();
    printf("kernels run on: %s\n", on_host ? "CPU" : "GPU");

    apply_gpu(x, y, n);
    check("1. first kernel:", x, y, n);

    scale = 3.0;                       /* changes only the host copy */
    apply_gpu(x, y, n);
    check("2. scale = 3 on the host:", x, y, n);

    #pragma omp target update to(scale)
    apply_gpu(x, y, n);
    check("3. after target update to(scale):", x, y, n);

    free(x); free(y);
    return 0;
}
