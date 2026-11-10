/*
 * fix_race.c -- OpenMP Lesson 3: three ways to update a shared variable safely.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 4 (critical and atomic constructs, reduction clause),
 *         Ch. 5 (performance cost of synchronization).
 *
 * Compile (no -O flag, as in Lesson 2):
 *     gcc -Wall -fopenmp -o fix_race fix_race.c
 * Run:
 *     OMP_NUM_THREADS=4 ./fix_race
 */

#include <stdio.h>
#include <omp.h>

#define N 1000000    /* increments done by EACH thread (same as Lesson 2) */

int main(void)
{
    int nthreads = omp_get_max_threads();   /* team size the regions will use */
    double t;

    printf("threads = %d, expected = %d\n\n", nthreads, nthreads * N);

    /* ---- 0) The bug from Lesson 2: unprotected shared update ---- */
    int c0 = 0;
    t = omp_get_wtime();                     /* wall-clock time in seconds */
    #pragma omp parallel
    {
        for (int i = 0; i < N; i++)
            c0++;                            /* DATA RACE */
    }
    printf("race      : %9d   %.4f s\n", c0, omp_get_wtime() - t);

    /* ---- 1) critical: only ONE thread at a time may run the block ----
       Works for any code (several statements, function calls, I/O).
       Cost: threads wait in line for a lock on every increment.        */
    int c1 = 0;
    t = omp_get_wtime();
    #pragma omp parallel
    {
        for (int i = 0; i < N; i++) {
            #pragma omp critical
            c1++;
        }
    }
    printf("critical  : %9d   %.4f s\n", c1, omp_get_wtime() - t);

    /* ---- 2) atomic: the single update x++ / x += e is done indivisibly ----
       Only for ONE simple update statement, but uses a hardware
       instruction instead of a lock, so it is cheaper than critical.   */
    int c2 = 0;
    t = omp_get_wtime();
    #pragma omp parallel
    {
        for (int i = 0; i < N; i++) {
            #pragma omp atomic
            c2++;
        }
    }
    printf("atomic    : %9d   %.4f s\n", c2, omp_get_wtime() - t);

    /* ---- 3) reduction: each thread gets a PRIVATE copy of c3, set to 0
       (the identity of +). At the end of the region OpenMP adds all the
       private copies into the shared c3. No synchronization in the loop. */
    int c3 = 0;
    t = omp_get_wtime();
    #pragma omp parallel reduction(+:c3)
    {
        for (int i = 0; i < N; i++)
            c3++;                            /* private copy: no race */
    }
    printf("reduction : %9d   %.4f s\n", c3, omp_get_wtime() - t);

    return 0;
}
