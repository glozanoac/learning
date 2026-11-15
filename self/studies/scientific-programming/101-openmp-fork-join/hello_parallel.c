/*
 * hello_parallel.c -- OpenMP Lesson 1: the parallel region (fork-join).
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 2 (fork-join execution model) and Ch. 3 (first OpenMP program).
 *
 * Compile:   gcc -Wall -fopenmp -o hello_parallel hello_parallel.c
 * Run:       ./hello_parallel
 *            OMP_NUM_THREADS=4 ./hello_parallel
 */

#include <stdio.h>
#include <omp.h>    /* OpenMP runtime library: omp_get_thread_num(), ... */

int main(void)
{
    /* ---- SERIAL PART: only one thread exists (the "master", id 0) ---- */
    printf("before: %d thread(s) running\n", omp_get_num_threads());

    /* ---- FORK: a team of threads is created here ----
       Every thread executes the whole block { ... } below.            */
    #pragma omp parallel
    {
        /* Declared INSIDE the region -> each thread has its own copy. */
        int id = omp_get_thread_num();     /* 0, 1, ..., n-1          */
        int n  = omp_get_num_threads();    /* size of the team        */

        printf("  hello from thread %d of %d\n", id, n);
    }
    /* ---- JOIN: implicit barrier. The master waits here until every
       thread has finished the block; then the team is dissolved.     */

    /* ---- SERIAL PART again: back to one thread ---- */
    printf("after:  %d thread(s) running\n", omp_get_num_threads());

    return 0;
}
