/*
 * schedule.c -- OpenMP Lesson 5: how loop iterations are assigned to threads.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 4 (the schedule clause, the nowait clause),
 *         Ch. 5 (load imbalance).
 *
 * Compile (no -O flag, so work() is not optimised away):
 *     gcc -Wall -fopenmp -o schedule schedule.c
 * Run (use as many threads as you have cores, see `nproc`):
 *     OMP_NUM_THREADS=2 ./schedule
 */

#include <stdio.h>
#include <omp.h>

#define CHUNK 1       /* chunk size of 'dynamic' scheduler          */ 
#define N    1000      /* number of iterations                       */
#define K    200       /* iteration i does i*K units of work         */
#define MAXT 64        /* max threads we keep results for            */

/* Unequal work: iteration 999 costs ~1000x more than iteration 1. */
static void work(int i)
{
    volatile double x = 0.0;           /* volatile: keep the loop alive */
    for (long j = 0; j < (long)i * K; j++)
        x += 1.0;
}

/* Print when each thread finished its share, and the total time. */
static void report(const char *name, const double finish[], int p, double total)
{
    printf("%-18s total %.3f s   finish per thread:", name, total);
    for (int k = 0; k < p; k++)
        printf(" %.3f", finish[k]);
    printf("\n");
}

int main(void)
{
    int p = omp_get_max_threads();
    double finish[MAXT];
    double t0;

    /* ---- 1) static: iterations cut into p contiguous blocks BEFORE the
       loop starts. Thread 0 gets the cheap iterations, the last thread
       the expensive ones -> load imbalance.                             */
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < N; i++)
            work(i);
        /* nowait removed the barrier after the loop, so each thread
           records the moment IT finished, not when the slowest did.   */
        finish[omp_get_thread_num()] = omp_get_wtime() - t0;
    }
    report("static", finish, p, omp_get_wtime() - t0);

    /* ---- 2) dynamic, chunk 1: a thread takes ONE iteration, and when it
       is done it asks for the next free one. Threads that got cheap work
       simply take more iterations -> balanced. Cost: one request per
       iteration (a little overhead).                                    */
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for schedule(dynamic, CHUNK) nowait
        for (int i = 0; i < N; i++)
            work(i);
        finish[omp_get_thread_num()] = omp_get_wtime() - t0;
    }
    report("dynamic,1", finish, p, omp_get_wtime() - t0);

    /* ---- 3) guided: like dynamic, but chunks start large and shrink.
       Fewer requests than dynamic,1 and still balanced at the end.      */
    t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for schedule(guided) nowait
        for (int i = 0; i < N; i++)
            work(i);
        finish[omp_get_thread_num()] = omp_get_wtime() - t0;
    }
    report("guided", finish, p, omp_get_wtime() - t0);

    return 0;
}
