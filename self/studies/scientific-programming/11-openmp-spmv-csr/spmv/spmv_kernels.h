/*
 * spmv_kernels.h -- four parallel versions of y = A x for CSR.
 */
#ifndef SPMV_KERNELS_H
#define SPMV_KERNELS_H

#include "csr.h"

void spmv_serial (const csr_t *A, const double *x, double *y);  /* reference      */
void spmv_static (const csr_t *A, const double *x, double *y);  /* rows, static   */
void spmv_dynamic(const csr_t *A, const double *x, double *y);  /* rows, dynamic  */
void spmv_nnz    (const csr_t *A, const double *x, double *y);  /* nnz-balanced   */
void spmv_merge  (const csr_t *A, const double *x, double *y);  /* merge path     */

#endif
