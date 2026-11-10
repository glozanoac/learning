/*
 * single_sections.c -- OpenMP Lesson 6: barrier, single and sections.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 4 (single and sections work-sharing constructs,
 *         barrier synchronization construct).
 *
 * Compile:
 *     gcc -Wall -fopenmp -o single_sections single_sections.c
 * Run:
 *     OMP_NUM_THREADS=4 ./single_sections
 */

#include <stdio.h>
#include <unistd.h>     /* usleep(): pause a thread for some microseconds */
#include <omp.h>

#define N 100

int main(void)
{
    int  a[N];
    long sum = 0;
    int  max = 0;

    /* ================= Part 1: explicit barrier ================= */
    printf("--- Part 1: barrier ---\n");
    #pragma omp parallel
    {
        int id = omp_get_thread_num();

        usleep(100000 * id);              /* phase 1: thread id works id*0.1 s */
        printf("  thread %d finished phase 1\n", id);

        #pragma omp barrier
        /* No thread passes this line until ALL threads have reached it. */

        printf("  thread %d started  phase 2\n", id);
    }

    /* ============== Part 2: single and sections ================= */
    printf("--- Part 2: single and sections ---\n");
    #pragma omp parallel
    {
        /* single: exactly ONE thread (whichever arrives first) runs the
           block; the others skip it and WAIT at its implicit barrier,
           so nobody reads a[] before it is filled.                     */
        #pragma omp single
        {
            printf("  thread %d fills the array (single)\n", omp_get_thread_num());
            for (int i = 0; i < N; i++)
                a[i] = (i * 37) % 101;    /* some numbers between 0 and 100 */
        }

        /* sections: DIFFERENT blocks of code, each run exactly once. Free
           threads take the sections; which thread gets which is up to the
           runtime (one thread may even run both if they are short).
           Here: two independent tasks on the same data. Threads without
           a section just wait at the implicit barrier.                     */
        #pragma omp sections
        {
            #pragma omp section
            {
                for (int i = 0; i < N; i++)
                    sum += a[i];          /* only this section writes sum */
                printf("  thread %d computed sum (section 1)\n", omp_get_thread_num());
            }
            #pragma omp section
            {
                for (int i = 0; i < N; i++)
                    if (a[i] > max) max = a[i];   /* only this section writes max */
                printf("  thread %d computed max (section 2)\n", omp_get_thread_num());
            }
        }   /* implicit barrier: both results are ready after this line */

        #pragma omp single
        printf("  thread %d prints: sum = %ld, max = %d (single)\n",
               omp_get_thread_num(), sum, max);
    }
    return 0;
}
