/*
 * target_update.c -- OpenMP Lesson 31: data that lives on the GPU,
 * with target enter data, target exit data and target update.
 * Source: Deakin & Mattson (2023), Ch. 4 (Memory Movement).
 *
 * A vector x stays on the GPU while a kernel adds 1 to every element NSTEP
 * times.  Every EVERY steps the host looks at x: not at all, only x[0],
 * or the whole vector.
 *
 * Compile:  clang -O2 -fopenmp --offload-arch=sm_61 \
 *                 -o target_update target_update.c
 * Run:      OMP_TARGET_OFFLOAD=MANDATORY ./target_update [log2 n]
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define NSTEP 200
#define EVERY 10

/* ---- data lifetime on the device -------------------------------------- */
static void to_device(double *x, long n)
{
    #pragma omp target enter data map(to: x[0:n])      /* allocate + copy in */
}

static void from_device(double *x, long n)
{
    #pragma omp target exit data map(from: x[0:n])     /* copy out + free    */
}

/* ---- the kernel: x must already be on the device ---------------------- */
static void add_one(double *x, long n)
{
    #pragma omp target teams distribute parallel for map(present, alloc: x[0:n])
    for (long i = 0; i < n; i++)
        x[i] += 1.0;
}

/* NSTEP kernels; mode 0: no reads, 1: read x[0], 2: read all of x */
static double run(double *x, long n, int mode)
{
    for (long i = 0; i < n; i++) x[i] = 0.0;
    double t = omp_get_wtime();
    to_device(x, n);
    for (int k = 1; k <= NSTEP; k++) {
        add_one(x, n);
        if (k % EVERY == 0) {
            if (mode == 1) {
                #pragma omp target update from(x[0:1])
            } else if (mode == 2) {
                #pragma omp target update from(x[0:n])
            }
        }
    }
    from_device(x, n);
    return omp_get_wtime() - t;
}

static const char *check(const double *x, long n)
{
    for (long i = 0; i < n; i++)
        if (x[i] != NSTEP) return "NO";
    return "yes";
}

int main(int argc, char **argv)
{
    long n = 1L << (argc > 1 ? atoi(argv[1]) : 25);   /* 2^25 doubles = 256 MiB */
    double *x = malloc(n * sizeof(double));

    int on_host = 1;                                  /* also starts up the GPU */
    #pragma omp target map(from: on_host)
    on_host = omp_is_initial_device();
    printf("ran on: %s, n = %ld (%.0f MiB), %d steps, reads every %d\n",
           on_host ? "CPU" : "GPU", n, n * 8.0 / (1 << 20), NSTEP, EVERY);

    const char *name[3] = {"no reads", "read x[0]", "read all x"};
    for (int mode = 0; mode < 3; mode++) {
        double t = run(x, n, mode);
        printf("%-11s: %6.3f s   correct: %s\n", name[mode], t, check(x, n));
    }
    free(x);
    return 0;
}
