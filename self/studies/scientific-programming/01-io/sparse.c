#include "sparse.h"

double* malloc_vector(int size){
    double*z = (double*)malloc(size*sizeof(double));

    return z;
}

csr* malloc_csr(int M, int nnz){
	csr* mtx = (csr*) malloc(sizeof(csr));

	mtx->M = M; mtx->nnz = nnz;
	mtx->val = (double*) malloc(nnz*sizeof(double));
	mtx->col = (int*) malloc(nnz*sizeof(int));
	mtx->row_ptr = (int*) malloc((M+1)*sizeof(int));

	return mtx;
}

void free_csr(csr* mtx){
	free(mtx->val);
	free(mtx->col);
	free(mtx->row_ptr);
	free(mtx);
}



void csr_spmv(csr* mtx, double* x, double* y){
	double s;
    for(int i=0; i<mtx->M; i++){
        s = 0;
        for(int k=mtx->row_ptr[i]; k < mtx->row_ptr[i+1]; k++){
            s += mtx->val[k]*x[mtx->col[k]];

        }
        y[i] = s;
	}
}
