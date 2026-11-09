/*
 * pi_omp.c - estimate pi = integral_0^1 4/(1+x^2) dx with the midpoint rule,
 * first serially and then with an OpenMP parallel reduction, and report
 * where each thread ran plus the speedup.
 *
 * Build: gcc -O2 -fopenmp -o pi_omp pi_omp.c
 * Run:   OMP_NUM_THREADS=4 ./pi_omp [steps]
 */
#define _GNU_SOURCE
#include <math.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static const double PI_REF = 3.14159265358979323846;

static double pi_serial(long n)
{
    const double h = 1.0 / (double)n;
    double sum = 0.0;
    for (long i = 0; i < n; i++) {
        const double x = h * ((double)i + 0.5);
        sum += 4.0 / (1.0 + x * x);
    }
    return h * sum;
}

static double pi_parallel(long n)
{
    const double h = 1.0 / (double)n;
    double sum = 0.0;
#pragma omp parallel for reduction(+ : sum) schedule(static)
    for (long i = 0; i < n; i++) {
        const double x = h * ((double)i + 0.5);
        sum += 4.0 / (1.0 + x * x);
    }
    return h * sum;
}

int main(int argc, char **argv)
{
    const long n = (argc > 1) ? atol(argv[1]) : 400000000L;
    char host[256];
    gethostname(host, sizeof host);

    printf("Host: %s | steps: %ld | max threads: %d\n\n",
           host, n, omp_get_max_threads());

    /* Thread placement: shows Slurm's CPU allocation + OMP_PROC_BIND at work */
#pragma omp parallel
    {
#pragma omp critical
        printf("  thread %2d of %2d on CPU %d\n",
               omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
    }

    double t0 = omp_get_wtime();
    const double ps = pi_serial(n);
    const double ts = omp_get_wtime() - t0;

    t0 = omp_get_wtime();
    const double pp = pi_parallel(n);
    const double tp = omp_get_wtime() - t0;

    printf("\n%-9s %18s %12s %10s\n", "version", "pi", "abs error", "time [s]");
    printf("%-9s %18.15f %12.3e %10.4f\n", "serial", ps, fabs(ps - PI_REF), ts);
    printf("%-9s %18.15f %12.3e %10.4f\n", "openmp", pp, fabs(pp - PI_REF), tp);
    printf("\nSpeedup: %.2fx with %d threads (efficiency %.0f%%)\n",
           ts / tp, omp_get_max_threads(), 100.0 * ts / tp / omp_get_max_threads());

    return fabs(pp - PI_REF) < 1e-8 ? EXIT_SUCCESS : EXIT_FAILURE;
}
