/*
 * quicksort_tasks.c -- OpenMP Lesson 12 (second example): parallel quicksort
 * with tasks.
 * Source: van der Pas, Stotzer & Terboven (2017), "Using OpenMP -- The Next
 *         Step", chapter on tasking.
 *
 * Quicksort splits an array around a pivot into two parts of UNPREDICTABLE
 * size, then sorts each part. The parts can be sorted independently, so each
 * one becomes a task. No loop describes this work: its shape is only known
 * while the program runs.
 *
 * Compile:
 *     gcc -Wall -O2 -fopenmp -o quicksort_tasks quicksort_tasks.c
 * Run:   ./quicksort_tasks n cutoff
 *     OMP_NUM_THREADS=4 ./quicksort_tasks 20000000 10000
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define MAXT 64

/* Work done by each thread, in elements sorted in its serial pieces.
   Padded to one 64-byte cache line per thread: no false sharing (Lesson 8). */
typedef struct { _Alignas(64) long elements; long pieces; } counter_t;
static counter_t work[MAXT];

static void swap(double *a, long i, long j) { double t = a[i]; a[i] = a[j]; a[j] = t; }

/* Partition a[lo..hi] around the median of a[lo], a[mid], a[hi].
   Returns p such that a[lo..p-1] <= a[p] <= a[p+1..hi].                    */
static long partition(double *a, long lo, long hi)
{
    long mid = lo + (hi - lo) / 2;
    if (a[mid] < a[lo]) swap(a, mid, lo);
    if (a[hi]  < a[lo]) swap(a, hi, lo);
    if (a[hi]  < a[mid]) swap(a, hi, mid);
    swap(a, mid, hi);                       /* median goes to the end */
    double pivot = a[hi];
    long i = lo;
    for (long j = lo; j < hi; j++)
        if (a[j] < pivot) swap(a, i++, j);
    swap(a, i, hi);
    return i;
}

/* plain recursive quicksort, used for small pieces */
static void quicksort_serial(double *a, long lo, long hi)
{
    while (lo < hi) {
        long p = partition(a, lo, hi);
        if (p - lo < hi - p) { quicksort_serial(a, lo, p - 1); lo = p + 1; }
        else                 { quicksort_serial(a, p + 1, hi); hi = p - 1; }
    }
}

static void quicksort_tasks(double *a, long lo, long hi, long cutoff)
{
    if (hi - lo < cutoff) {                 /* small piece: not worth a task */
        quicksort_serial(a, lo, hi);
        int t = omp_get_thread_num();
        work[t].elements += hi - lo + 1;    /* only thread t writes work[t] */
        work[t].pieces++;
        return;
    }
    long p = partition(a, lo, hi);          /* split: sizes known only now */

    /* Two independent sub-problems -> two tasks. Variables are firstprivate
       in a task: each task gets a COPY of the pointer a, but the copy points
       to the same shared array, so the sorting happens in place.          */
    #pragma omp task
    quicksort_tasks(a, lo, p - 1, cutoff);
    #pragma omp task
    quicksort_tasks(a, p + 1, hi, cutoff);

    /* No taskwait: nothing below needs the children's results. All tasks
       are finished at the barrier at the end of the parallel region.     */
}

static int is_sorted(const double *a, long n)
{
    for (long i = 1; i < n; i++)
        if (a[i - 1] > a[i]) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    long n      = (argc > 1) ? atol(argv[1]) : 20000000;
    long cutoff = (argc > 2) ? atol(argv[2]) : 10000;
    int  T      = omp_get_max_threads();
    if (T > MAXT) { fprintf(stderr, "at most %d threads\n", MAXT); return 1; }

    double *a = malloc((size_t)n * sizeof(double));
    double *b = malloc((size_t)n * sizeof(double));
    if (!a || !b) { fprintf(stderr, "out of memory\n"); return 1; }
    srand(42);
    for (long i = 0; i < n; i++) a[i] = b[i] = (double)rand() / RAND_MAX;

    /* serial reference time */
    double t = omp_get_wtime();
    quicksort_serial(b, 0, n - 1);
    double t_serial = omp_get_wtime() - t;

    /* parallel sort of the same data */
    t = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single              /* one thread starts the recursion */
        quicksort_tasks(a, 0, n - 1, cutoff);
    }                                   /* barrier: every task is done */
    double t_tasks = omp_get_wtime() - t;

    printf("n = %ld, cutoff = %ld, threads = %d\n", n, cutoff, T);
    printf("serial : %.3f s\n", t_serial);
    printf("tasks  : %.3f s   speed-up %.2f   sorted: %s\n",
           t_tasks, t_serial / t_tasks, is_sorted(a, n) ? "yes" : "NO");
    printf("work per thread (elements sorted in serial pieces):\n");
    for (int i = 0; i < T; i++)
        printf("  thread %2d: %10ld elements in %6ld pieces (%5.1f%%)\n", i,
               work[i].elements, work[i].pieces, 100.0 * work[i].elements / n);

    free(a); free(b);
    return 0;
}
