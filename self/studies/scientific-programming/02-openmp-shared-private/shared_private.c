/*
 * shared_private.c -- OpenMP Lesson 2: shared vs private variables.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 4 (shared and private clauses), Ch. 7 (data races).
 *
 * Compile (no -O flag, so the compiler does not hide the race):
 *     gcc -Wall -fopenmp -o shared_private shared_private.c
 * Run:
 *     OMP_NUM_THREADS=4 ./shared_private
 */

#include <stdio.h>
#include <omp.h>

#define N 1000000    /* increments done by EACH thread */

int main(void)
{
    int counter  = 0;   /* declared BEFORE the region -> SHARED by default:
                           all threads read and write the same memory cell. */
    int my_count = 0;   /* also declared before, but we will make it
                           PRIVATE with a clause below.                     */
    int nthreads = 1;

    #pragma omp parallel private(my_count)
    {
        /* private(my_count): each thread gets its OWN my_count.
           The private copy starts UNINITIALISED, so set it here.          */
        my_count = 0;

        for (int i = 0; i < N; i++) {
            counter++;      /* SHARED: read, add 1, write back.
                               Two threads can read the same old value
                               and one increment is lost -> DATA RACE.    */
            my_count++;     /* PRIVATE: no other thread touches it -> safe */
        }

        printf("  thread %d: my_count = %d\n", omp_get_thread_num(), my_count);

        if (omp_get_thread_num() == 0)
            nthreads = omp_get_num_threads();   /* only one thread writes it */
    }

    printf("expected counter = %d\n", nthreads * N);
    printf("actual   counter = %d\n", counter);
    return 0;
}
