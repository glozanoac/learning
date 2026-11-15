/* Test 2: OpenMP on the CPU (threads) */
#include <stdio.h>
#include <omp.h>

int main(void)
{
    long i, n = 100000000;
    double sum = 0.0, t;

    #pragma omp parallel
    {
        #pragma omp single
        printf("Threads: %d\n", omp_get_num_threads());
    }

    t = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for (i = 1; i <= n; i++)
        sum += 1.0 / ((double) i * i);
    t = omp_get_wtime() - t;

    printf("CPU OpenMP OK: sum = %.10f, time = %.3f s\n", sum, t);
    return 0;
}
