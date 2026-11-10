/*
 * locks.c -- OpenMP Lesson 9: locks.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 4 (lock routines of the runtime library).
 *
 * Compile:
 *     gcc -Wall -fopenmp -o locks locks.c
 * Run (use as many threads as you have cores, see `nproc`):
 *     OMP_NUM_THREADS=2 ./locks
 */

#include <stdio.h>
#include <omp.h>

#define N     20000000   /* values to put into the histogram          */
#define NBINS 1000       /* number of bins (try 2 in the experiment)  */
//#define NBINS 2       /* number of bins (try 2 in the experiment)  */

int hist[NBINS];
omp_lock_t lock[NBINS];  /* ONE lock per bin */

/* A cheap deterministic "random" bin for value i. */
static int bin_of(long i)
{
    return (int)((i * 2654435761UL) % NBINS);
}

int main(void)
{
    double t;
    long total;

    /* ===== Part 1: histogram protected by critical =====
       critical = ONE lock for the whole block. Two threads that update
       DIFFERENT bins still wait for each other.                         */
    for (int b = 0; b < NBINS; b++) hist[b] = 0;
    t = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        int b = bin_of(i);
        #pragma omp critical
        hist[b]++;
    }
    t = omp_get_wtime() - t;
    total = 0;
    for (int b = 0; b < NBINS; b++) total += hist[b];
    printf("critical      : %.3f s   total = %ld\n", t, total);

    /* ===== Part 2: histogram protected by one lock per bin =====
       Threads wait only if they update the SAME bin at the same time.   */
    for (int b = 0; b < NBINS; b++) {
        hist[b] = 0;
        omp_init_lock(&lock[b]);         /* 1. create (before use)       */
    }
    t = omp_get_wtime();
    #pragma omp parallel for
    for (long i = 0; i < N; i++) {
        int b = bin_of(i);
        omp_set_lock(&lock[b]);          /* 2. acquire: wait if taken    */
        hist[b]++;
        omp_unset_lock(&lock[b]);        /* 3. release: ALWAYS do it     */
    }
    t = omp_get_wtime() - t;
    for (int b = 0; b < NBINS; b++)
        omp_destroy_lock(&lock[b]);      /* 4. free (after use)          */
    total = 0;
    for (int b = 0; b < NBINS; b++) total += hist[b];
    printf("lock per bin  : %.3f s   total = %ld\n", t, total);

    return 0;
}
