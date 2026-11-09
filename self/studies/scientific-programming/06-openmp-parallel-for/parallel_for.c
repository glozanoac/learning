/*
 * parallel_for.c -- OpenMP Lesson 4: sharing the work of a loop.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 4 (the loop work-sharing construct, combined parallel for).
 *
 * Compile:
 *     gcc -Wall -fopenmp -o parallel_for parallel_for.c
 * Run:
 *     OMP_NUM_THREADS=4 ./parallel_for
 */

#include <stdio.h>
#include <omp.h>

#define SHOW  16           /* small loop: to SEE who does what         */
#define STEPS 100000000    /* big loop: to MEASURE the speed-up        */

int main(void)
{
    /* ---- Part 1: who does which iteration? ----
       "parallel for" = create a team AND split the loop among it.
       Each iteration is done by exactly ONE thread (not by all, as in
       Lessons 2-3). The loop index i is private automatically.         */
    #pragma omp parallel for
    for (int i = 0; i < SHOW; i++)
        printf("  iteration %2d -> thread %d\n", i, omp_get_thread_num());
    /* implicit barrier here: all iterations are finished after the loop */

    /* ---- Part 2: a loop worth parallelising ----
       pi = integral from 0 to 1 of 4/(1+x^2) dx, by the midpoint rule.
       Iterations are independent; only sum is combined -> reduction.   */
    double step = 1.0 / STEPS;
    double sum  = 0.0;

    double t = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < STEPS; i++) {
        double x = (i + 0.5) * step;      /* declared inside -> private */
        sum += 4.0 / (1.0 + x * x);
    }
    t = omp_get_wtime() - t;

    printf("\nthreads = %d   pi = %.12f   time = %.4f s\n",
           omp_get_max_threads(), step * sum, t);
    return 0;
}
