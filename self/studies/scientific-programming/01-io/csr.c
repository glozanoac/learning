#include <stdio.h>
#include <string.h>
#include "sparse.h"

/*
 * Running example (section 2.3):
 *
 *     | 10  0  0  0  2  0 |        | 1 |
 *     |  0  3  0  0  0  0 |        | 2 |
 * A = |  0  0  0  0  0  0 |    x = | 3 |     M = N = 6, nnz = 8
 *     |  0  0  7  0  0  1 |        | 4 |
 *     |  4  0  0  0  0  0 |        | 5 |
 *     |  0  0  0  5  0  6 |        | 6 |
 *
 * Expected: y = (20, 6, 0, 27, 4, 56)
 */
int main(void){
	const int M = 6, N = 6, nnz = 8;

	const double val[]     = {10, 2, 3, 7, 1, 4, 5, 6};
	const int    col[]     = { 0, 4, 1, 2, 5, 0, 3, 5};
	const int    row_ptr[] = { 0, 2, 3, 3, 5, 6, 8};   /* M+1 entries */

	/* Allocate the CSR matrix and copy the arrays into it */
	csr* A = malloc_csr(M, nnz);
	memcpy(A->val,     val,     nnz*sizeof(double));
	memcpy(A->col,     col,     nnz*sizeof(int));
	memcpy(A->row_ptr, row_ptr, (M+1)*sizeof(int));

	/* Initialize x (length N) and y (length M) */
    double x[] = {1, 2, 3, 4, 5, 6};
	double* y = (double*)malloc(M*sizeof(double));

	csr_spmv(A, x, y);

	printf("y = (");
	for(int i=0; i<M; i++) printf(i ? ", %g" : "%g", y[i]);
	printf(")\n");

	free(y);
	free_csr(A);
	return 0;
}
