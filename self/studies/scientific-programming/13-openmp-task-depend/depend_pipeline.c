/*
 * depend_pipeline.c -- OpenMP Lesson 13: task dependencies (depend).
 * Source: van der Pas, Stotzer & Terboven (2017), "Using OpenMP -- The Next
 *         Step", chapter on tasking (task dependences, OpenMP 4.0).
 *
 * Two stages on NB blocks:  stage 1 writes x[b],  stage 2 reads x[b] and
 * writes y[b] = 2 x[b].  Block 0 is slow in stage 1, the others in stage 2.
 * Work is simulated with usleep, so the times do not depend on your CPU.
 *
 * Compile:  gcc -Wall -O2 -fopenmp -o depend_pipeline depend_pipeline.c
 * Run:      OMP_NUM_THREADS=2 ./depend_pipeline
 */

#include <stdio.h>
#include <unistd.h>
#include <omp.h>

#define NB 8

static double x[NB], y[NB];

static int cost1(int b) { return b == 0 ? 100 : 10; }   /* ms */
static int cost2(int b) { return b == 0 ? 10 : 100; }   /* ms */

static void stage1(int b) { usleep(1000 * cost1(b)); x[b] = b + 1.0; }
static void stage2(int b) { usleep(1000 * cost2(b)); y[b] = 2.0 * x[b]; }

static void reset(void)
{
    for (int b = 0; b < NB; b++) x[b] = y[b] = 0.0;
}

static const char *check(void)
{
    for (int b = 0; b < NB; b++)
        if (y[b] != 2.0 * (b + 1)) return "NO";
    return "yes";
}

int main(void)
{
    double t;

    /* ---- 1) barrier: every stage-1 task must finish before stage 2 ---- */
    reset();
    t = omp_get_wtime();
    #pragma omp parallel
    #pragma omp single
    {
        for (int b = 0; b < NB; b++) {
            #pragma omp task
            stage1(b);
        }
        #pragma omp taskwait          /* waits for ALL stage-1 tasks */
        for (int b = 0; b < NB; b++) {
            #pragma omp task
            stage2(b);
        }
    }
    printf("barrier : %.3f s   correct: %s\n", omp_get_wtime() - t, check());

    /* ---- 2) depend: stage 2 of block b waits only for stage 1 of block b ---- */
    reset();
    t = omp_get_wtime();
    #pragma omp parallel
    #pragma omp single
    {
        for (int b = 0; b < NB; b++) {
            #pragma omp task depend(out: x[b])
            stage1(b);
            #pragma omp task depend(in: x[b]) depend(out: y[b])
            stage2(b);
        }
    }                                 /* all tasks are done at this barrier */
    printf("depend  : %.3f s   correct: %s\n", omp_get_wtime() - t, check());

    printf("threads = %d\n", omp_get_max_threads());
    return 0;
}
