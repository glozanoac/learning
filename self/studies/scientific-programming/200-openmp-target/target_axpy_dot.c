/*
 * target_axpy_dot.c -- OpenMP Lesson 30: offloading to the GPU with
 * target, map and teams distribute parallel for.
 * Source: Deakin & Mattson (2023), Ch. 3 (target construct) and Ch. 4 (map).
 *
 * Kernels: y = a*x + y (axpy) and s = x.y (dot), run NITER times,
 *   cpu      : parallel for on the CPU threads
 *   gpu map  : every call copies x and y between CPU and GPU memory
 *   gpu data : x and y are copied once (target data); the calls reuse them
 *
 * Compile:  clang -O2 -fopenmp --offload-arch=sm_61 \
 *                 -o target_axpy_dot target_axpy_dot.c
 * Run:      ./target_axpy_dot [log2 n]
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NITER 20

/* ---- CPU ---------------------------------------------------------------- */
static void axpy_cpu(long n, double a, const double *x, double *y)
{
    #pragma omp parallel for
    for (long i = 0; i < n; i++)
        y[i] += a * x[i];
}

static double dot_cpu(long n, const double *x, const double *y)
{
    double s = 0.0;
    #pragma omp parallel for reduction(+: s)
    for (long i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

/* ---- GPU: the loop runs on the device; map says what to copy ------------ */
static void axpy_gpu(long n, double a, const double *x, double *y)
{
    #pragma omp target teams distribute parallel for \
            map(to: x[0:n]) map(tofrom: y[0:n])
    for (long i = 0; i < n; i++)
        y[i] += a * x[i];
}

static double dot_gpu(long n, const double *x, const double *y)
{
    double s = 0.0;
    #pragma omp target teams distribute parallel for reduction(+: s) \
            map(to: x[0:n], y[0:n]) map(tofrom: s)
    for (long i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

/* ---- helpers ------------------------------------------------------------ */
static void init(long n, double *x, double *y)
{
    for (long i = 0; i < n; i++) { x[i] = 1.0; y[i] = 2.0; }
}

/* exact result after NITER steps with a = 0.5: y = 2 + 0.5*NITER, s = n*y */
static const char *check(long n, const double *y, double s)
{
    double yk = 2.0 + 0.5 * NITER;
    for (long i = 0; i < n; i++)
        if (y[i] != yk) return "NO";
    return s == (double)n * yk ? "yes" : "NO";
}

static void report(const char *name, long n, double t, const double *y, double s)
{
    double ms = 1e3 * t / NITER;                      /* per iteration          */
    double gbs = 40.0 * n / (t / NITER) / 1e9;        /* 40 bytes per element   */
    printf("%-9s: %8.2f ms/iter  %7.1f GB/s   correct: %s\n",
           name, ms, gbs, check(n, y, s));
}

int main(int argc, char **argv)
{
    long n = 1L << (argc > 1 ? atoi(argv[1]) : 25);   /* 2^25 doubles = 256 MiB */
    const double a = 0.5;
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    double s = 0.0, t;

    /* 1. Where does a target region run? (also starts up the GPU) */
    int on_host = 1;
    #pragma omp target map(from: on_host)
    on_host = omp_is_initial_device();
    printf("devices: %d   target region ran on: %s\n",
           omp_get_num_devices(), on_host ? "CPU (host)" : "GPU (device)");
    printf("n = %ld doubles (%.0f MiB per vector), %d iterations\n",
           n, n * 8.0 / (1 << 20), NITER);

    /* 2. CPU */
    init(n, x, y);
    t = omp_get_wtime();
    for (int k = 0; k < NITER; k++) { axpy_cpu(n, a, x, y); s = dot_cpu(n, x, y); }
    report("cpu", n, omp_get_wtime() - t, y, s);

    /* 3. GPU, copies on every call */
    init(n, x, y);
    t = omp_get_wtime();
    for (int k = 0; k < NITER; k++) { axpy_gpu(n, a, x, y); s = dot_gpu(n, x, y); }
    report("gpu map", n, omp_get_wtime() - t, y, s);

    /* 4. GPU, data kept on the device: inside target data the maps in
          axpy_gpu and dot_gpu find x and y already present: no copies */
    init(n, x, y);
    t = omp_get_wtime();
    #pragma omp target data map(to: x[0:n]) map(tofrom: y[0:n])
    {
        for (int k = 0; k < NITER; k++)
            { axpy_gpu(n, a, x, y); s = dot_gpu(n, x, y); }
    }   /* y is copied back here, once */
    report("gpu data", n, omp_get_wtime() - t, y, s);

    free(x); free(y);
    return 0;
}
