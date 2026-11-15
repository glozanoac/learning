/*
 * csr.h -- CSR matrix type and binary file I/O.
 *
 * Index widths (see design document, section 2):
 *   row_ptr : int64_t  -- nnz can exceed 2^31 (GAP, AGATHA matrices)
 *   col     : int32_t  -- every matrix of interest has fewer than 2^31 columns
 *   val     : double
 * Storage: 12*nnz + 8*(M+1) bytes.
 */
#ifndef CSR_H
#define CSR_H

#include <stdint.h>

typedef struct {
    int64_t  M, N, nnz;   /* rows, columns, stored nonzeros            */
    int64_t *row_ptr;     /* [M+1], row i is row_ptr[i] .. row_ptr[i+1]-1 */
    int32_t *col;         /* [nnz] column index of each value          */
    double  *val;         /* [nnz] values                              */
} csr_t;

/* Binary file layout: magic[8] = "CSRBIN1", M, N, nnz (int64),
   then row_ptr[M+1], col[nnz], val[nnz].                              */
#define CSR_MAGIC "CSRBIN1"

int  csr_write(const char *path, const csr_t *A);
int  csr_read(const char *path, csr_t *A);  /* first-touch placement, see csr.c */
void csr_free(csr_t *A);

#endif
