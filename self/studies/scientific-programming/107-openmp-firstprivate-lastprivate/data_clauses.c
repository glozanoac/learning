/*
 * data_clauses.c -- OpenMP Lesson 7: firstprivate, lastprivate, default(none).
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 4 (data-sharing attribute clauses).
 *
 * Compile:
 *     gcc -Wall -fopenmp -o data_clauses data_clauses.c
 * Run:
 *     OMP_NUM_THREADS=4 ./data_clauses
 */

#include <stdio.h>
#include <omp.h>

#define N 8

int main(void)
{
    int offset = 100;   /* a value every thread needs as a STARTING point */
    int last   = -1;    /* we want the value from the LAST iteration      */
    int a[N];
    int n = N;

    /* ======== Part 1: firstprivate ========
       private(offset)      -> each thread gets a copy, UNINITIALISED.
       firstprivate(offset) -> each thread gets a copy, INITIALISED with
                               the value offset had before the region (100).
       default(none)        -> the compiler refuses any variable whose
                               sharing we did not state explicitly.        */
    printf("--- Part 1: firstprivate ---\n");
    //#pragma omp parallel default(none) firstprivate(offset)
    #pragma omp parallel default(none) private(offset)
    {
        int id = omp_get_thread_num();   /* declared inside: private */
        offset += id;                    /* changes only THIS thread's copy */
        printf("  thread %d: offset = %d\n", id, offset);
    }
    /* The private copies are thrown away at the end of the region. */
    printf("  after the region: offset = %d\n", offset);

    /* ======== Part 2: lastprivate ========
       lastprivate(last) -> private inside the loop, and after the loop the
       original variable receives the value from the SEQUENTIALLY LAST
       iteration (i = n-1), no matter which thread executed it.           */
    printf("--- Part 2: lastprivate ---\n");
    #pragma omp parallel for default(none) shared(a, n) lastprivate(last)
    //#pragma omp parallel for default(none) shared(a) lastprivate(last)
    //#pragma omp parallel for default(none) shared(a, n) private(last)
    for (int i = 0; i < n; i++) {
        last = i * i;
        a[i] = last;
    }
    printf("  after the loop: last = %d   (a[n-1] = %d)\n", last, a[n - 1]);

    return 0;
}
