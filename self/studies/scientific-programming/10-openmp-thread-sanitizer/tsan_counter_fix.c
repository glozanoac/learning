/*
 * tsan_counter.c -- OpenMP Lesson 10: finding a data race with ThreadSanitizer.
 * Source: Chapman, Jost & van der Pas (2007), "Using OpenMP",
 *         Ch. 7 (Troubleshooting: data races, tools to find them).
 *
 * This is the buggy counter from Lesson 2, made smaller (the tool slows
 * the program down 5-15x).
 *
 * Arch Linux packages:  sudo pacman -S clang openmp compiler-rt
 *
 * Compile with ThreadSanitizer (-g gives file and line numbers in the report):
 *     clang -g -fopenmp -fsanitize=thread -o tsan_counter tsan_counter.c
 * Run, loading Archer (the helper that teaches the tool OpenMP's rules):
 *     export TSAN_OPTIONS='ignore_noninstrumented_modules=1'
 *     export OMP_TOOL_LIBRARIES=$(find /usr/lib -name libarcher.so | head -1)
 *     OMP_NUM_THREADS=2 ./tsan_counter
 */

#include <stdio.h>
#include <omp.h>

#define N 100000

int main(void)
{
    int counter = 0;             /* shared by default (Lesson 2) */

    #pragma omp parallel reduction(+:counter)         /* EXPERIMENT step 2: add  reduction(+:counter)  here */
    {
        for (int i = 0; i < N; i++)
            counter++;           /* DATA RACE: the tool should point at THIS line */
    }

    printf("expected = %d, counter = %d\n", omp_get_max_threads() * N, counter);
    return 0;
}
