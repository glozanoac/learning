/* Test 1: plain C toolchain (K&R level) */
#include <stdio.h>

int main(void)
{
    long i, n = 1000000;
    double sum = 0.0;

    for (i = 1; i <= n; i++)
        sum += 1.0 / ((double) i * i);   /* -> pi^2/6 */

    printf("C OK: sum = %.10f (expected ~1.6449330668)\n", sum);
    return 0;
}
