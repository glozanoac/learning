/*
 * target_teams.c -- OpenMP Lesson 32: how many teams and threads on the GPU,
 * with num_teams and thread_limit.
 * Source: Deakin & Mattson (2023), Ch. 5 (Using a GPU Effectively) and
 *         Ch. 6 (Managing a GPU's Hierarchical Parallelism).
 *
 * The same kernel y = 2 x runs with different numbers of teams and threads
 * per team.  x and y stay on the GPU; each launch is timed (best of NREP).
 *
 * Compile:  clang -O2 -fopenmp --offload-arch=sm_61 \
 *                 -o target_teams target_teams.c
 * Run:      OMP_TARGET_OFFLOAD=MANDATORY ./target_teams [log2 n]
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NREP 10

/* y = 2 x with nteams teams of at most nthreads threads; 0 = let the
   runtime choose.  info[] returns what was actually used.              */
static void scale(const double *x, double *y, long n,
                  int nteams, int nthreads, int *info)
{
    if (nteams > 0) {
        #pragma omp target teams distribute parallel for \
                num_teams(nteams) thread_limit(nthreads) \
                map(present, alloc: x[0:n], y[0:n]) map(from: info[0:2])
        for (long i = 0; i < n; i++) {
            y[i] = 2.0 * x[i];
            if (i == 0) {                      /* what the launch really used */
                info[0] = omp_get_num_teams();
                info[1] = omp_get_num_threads();
            }
        }
    } else {
        #pragma omp target teams distribute parallel for \
                map(present, alloc: x[0:n], y[0:n]) map(from: info[0:2])
        for (long i = 0; i < n; i++) {
            y[i] = 2.0 * x[i];
            if (i == 0) {                      /* what the launch really used */
                info[0] = omp_get_num_teams();
                info[1] = omp_get_num_threads();
            }
        }
    }
}

/* best time of NREP launches; prints teams, threads and bandwidth */
static void measure(const double *x, double *y, long n, int nteams, int nthreads)
{
    int info[2] = {0, 0};
    double best = 1e30;
    for (int r = 0; r < NREP; r++) {
        double t = omp_get_wtime();
        scale(x, y, n, nteams, nthreads, info);
        t = omp_get_wtime() - t;
        if (t < best) best = t;
    }
    printf("%7d %8d %10ld %9.2f %8.1f\n", info[0], info[1],
           (long)info[0] * info[1], 1e3 * best, 16.0 * n / best / 1e9);
}

int main(int argc, char **argv)
{
    long n = 1L << (argc > 1 ? atoi(argv[1]) : 25);   /* 2^25 doubles = 256 MiB */
    double *x = malloc(n * sizeof(double));
    double *y = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) x[i] = 1.0;

    #pragma omp target enter data map(to: x[0:n]) map(alloc: y[0:n])

    printf("n = %ld doubles; 16 bytes per element (read x, write y)\n", n);
    printf("  teams  threads      total        ms     GB/s\n");

    printf("runtime default:\n");
    measure(x, y, n, 0, 0);

    printf("A. 256 threads per team, more and more teams:\n");
    int teams[] = {1, 2, 5, 10, 20, 40, 80, 160, 640, 2560};
    for (int k = 0; k < 10; k++)
        measure(x, y, n, teams[k], 256);

    printf("B. 5 teams (one per multiprocessor), more and more threads:\n");
    int threads[] = {32, 64, 128, 256, 512, 1024};
    for (int k = 0; k < 6; k++)
        measure(x, y, n, 5, threads[k]);

    #pragma omp target exit data map(from: y[0:n]) map(delete: x[0:n])
    long bad = 0;
    for (long i = 0; i < n; i++) bad += (y[i] != 2.0);
    printf("correct: %s\n", bad ? "NO" : "yes");
    free(x); free(y);
    return 0;
}
