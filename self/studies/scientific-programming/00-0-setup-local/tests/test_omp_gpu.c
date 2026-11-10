/* Test 3: OpenMP offload to the NVIDIA GPU */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(void)
{
    int i, n = 1 << 24;              /* ~16M elements */
    int on_host = 1;
    float a = 2.0f, *x, *y;
    double err = 0.0, t;

    printf("Devices visible: %d\n", omp_get_num_devices());

    x = malloc(n * sizeof *x);
    y = malloc(n * sizeof *y);
    for (i = 0; i < n; i++) { x[i] = 1.0f; y[i] = 2.0f; }

    /* Does code inside 'target' actually run on the GPU? */
    #pragma omp target map(from: on_host)
    on_host = omp_is_initial_device();

    /* SAXPY on the GPU: y = a*x + y */
    t = omp_get_wtime();
    #pragma omp target teams distribute parallel for map(to: x[0:n]) map(tofrom: y[0:n])
    for (i = 0; i < n; i++)
        y[i] = a * x[i] + y[i];
    t = omp_get_wtime() - t;

    for (i = 0; i < n; i++)
        err += (y[i] - 4.0f) * (y[i] - 4.0f);

    printf("Ran on: %s\n", on_host ? "HOST (CPU) -- offload NOT working" : "GPU");
    printf("SAXPY error = %g (expected 0), time = %.3f s\n", err, t);

    free(x); free(y);
    return on_host;   /* nonzero exit if it fell back to the CPU */
}
