#ifndef SPARSE_H
#define SPARSE_H

#include <stdlib.h>

typedef struct {
	int M;
	int nnz;
	double* val;
	int* col;
	int* row_ptr;
}csr;

double* malloc_vector(int size);
csr* malloc_csr(int M, int nnz);
void free_csr(csr* mtx);
void csr_spmv(csr* mtx, double* x, double* y);

#endif
