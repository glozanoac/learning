/*
 * fib_tasks.c -- OpenMP Lesson 12: tasks and taskwait.
 * Source: van der Pas, Stotzer & Terboven (2017), "Using OpenMP -- The Next
 *         Step", chapter on tasking.
 *
 * Fibonacci numbers by the (deliberately slow) recursion
 *     fib(n) = fib(n-1) + fib(n-2),  fib(0) = 0, fib(1) = 1.
 * The recursion is a tree of calls, not a loop: "omp for" cannot split it,
 * but tasks can.
 *
 * Compile:
 *     gcc -Wall -O2 -fopenmp -o fib_tasks fib_tasks.c
 * Run:   ./fib_tasks n cutoff
 *     OMP_NUM_THREADS=4 ./fib_tasks 40 20
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* plain recursion, no tasks */
static long fib_serial(int n)
{
    return n < 2 ? n : fib_serial(n - 1) + fib_serial(n - 2);
}

/* Each call creates two tasks, one per sub-problem, and waits for both.
   Below the cutoff the work is too small to be worth a task: compute it
   directly.                                                              */
static long fib(int n, int cutoff)
{
    if (n < cutoff)
        return fib_serial(n);

    long a, b;

    /* A task is a piece of work put in a queue. ANY thread of the team may
       execute it, now or later. Variables of this function are
       firstprivate in a task by default (a copy!), so a and b must be
       shared, or the results would be written into the task's copies.    */
    #pragma omp task shared(a)
    a = fib(n - 1, cutoff);

    #pragma omp task shared(b)
    b = fib(n - 2, cutoff);

    /* wait until the two child tasks above have finished */
    #pragma omp taskwait
    return a + b;
}

int main(int argc, char **argv)
{
    int n      = (argc > 1) ? atoi(argv[1]) : 40;
    int cutoff = (argc > 2) ? atoi(argv[2]) : 20;
    long r;

    double t = omp_get_wtime();
    r = fib_serial(n);
    double t_serial = omp_get_wtime() - t;
    printf("serial : fib(%d) = %ld   %.3f s\n", n, r, t_serial);

    t = omp_get_wtime();
    #pragma omp parallel        /* create the team of threads ...           */
    {
        #pragma omp single      /* ... but only ONE thread starts the recursion;
                                   the others pick up the tasks it creates   */
        r = fib(n, cutoff);
    }                           /* all tasks are finished at the barrier here */
    double t_tasks = omp_get_wtime() - t;

    printf("tasks  : fib(%d) = %ld   %.3f s   (threads = %d, cutoff = %d, speed-up = %.2f)\n",
           n, r, t_tasks, omp_get_max_threads(), cutoff, t_serial / t_tasks);
    return 0;
}
