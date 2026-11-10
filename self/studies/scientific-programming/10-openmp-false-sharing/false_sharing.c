/*
 * false_sharing.c -- OpenMP Lesson 8: false sharing.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 5 (How to Get Good Performance: false sharing).
 *
 * Compile (no -O flag, so every increment really goes to memory):
 *     gcc -Wall -fopenmp -o false_sharing false_sharing.c
 * Run (use as many threads as you have cores, see `nproc`):
 *     OMP_NUM_THREADS=2 ./false_sharing
 *
 * Check your cache line size with:   getconf LEVEL1_DCACHE_LINESIZE
 */

#include <stdio.h>
#include <omp.h>

#define ITERS      100000000L  /* increments done by EACH thread        */
#define MAXT       64          /* max threads                           */
#define CACHE_LINE 64          /* bytes per cache line on most x86 CPUs */

/* (1) PACKED: counters next to each other. 8 longs fit in ONE 64-byte
       cache line, so up to 8 threads write into the same line.         */
long packed[MAXT];

/* (2) PADDED: _Alignas (C11) puts each counter at the start of its own
       64-byte block, so sizeof(padded_t) == 64 and every thread writes
       to a DIFFERENT cache line.                                        */
typedef struct {
    _Alignas(CACHE_LINE) long v;
} padded_t;
padded_t padded[MAXT];

int main(void)
{
    int p = omp_get_max_threads();
    double t;

    printf("threads = %d, sizeof(padded_t) = %zu bytes\n\n", p, sizeof(padded_t));

    /* ---- (1) packed: no data race (each thread has its own slot),
       but the slots share a cache line -> FALSE SHARING.               */
    t = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long i = 0; i < ITERS; i++)
            packed[id]++;
    }
    printf("packed : %.3f s\n", omp_get_wtime() - t);

    /* ---- (2) padded: same work, one cache line per thread. ---- */
    t = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long i = 0; i < ITERS; i++)
            padded[id].v++;
    }
    printf("padded : %.3f s\n", omp_get_wtime() - t);

    /* ---- (3) local: accumulate in a private variable and write the
       shared array only ONCE at the end. Usually the best fix.          */
    t = omp_get_wtime();
    #pragma omp parallel
    {
        int  id  = omp_get_thread_num();
        long sum = 0;                       /* private, lives in a register
                                               or this thread's stack     */
        for (long i = 0; i < ITERS; i++)
            sum++;
        packed[id] += sum;                  /* one write per thread */
    }
    printf("local  : %.3f s\n", omp_get_wtime() - t);

    return 0;
}
