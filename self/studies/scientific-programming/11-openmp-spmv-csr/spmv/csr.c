/*
 * csr.c -- binary CSR file I/O.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "csr.h"

/* fwrite/fread in pieces of at most 1 GiB: some systems limit one call. */
#define IO_CHUNK ((size_t)1 << 30)

static int write_all(FILE *f, const void *p, size_t bytes)
{
    const char *c = p;
    while (bytes > 0) {
        size_t n = bytes < IO_CHUNK ? bytes : IO_CHUNK;
        if (fwrite(c, 1, n, f) != n) return -1;
        c += n; bytes -= n;
    }
    return 0;
}

static int read_all(FILE *f, void *p, size_t bytes)
{
    char *c = p;
    while (bytes > 0) {
        size_t n = bytes < IO_CHUNK ? bytes : IO_CHUNK;
        if (fread(c, 1, n, f) != n) return -1;
        c += n; bytes -= n;
    }
    return 0;
}

int csr_write(const char *path, const csr_t *A)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    char magic[8] = CSR_MAGIC;
    int64_t hdr[3] = { A->M, A->N, A->nnz };
    int err = write_all(f, magic, 8)
           || write_all(f, hdr, sizeof hdr)
           || write_all(f, A->row_ptr, (size_t)(A->M + 1) * sizeof(int64_t))
           || write_all(f, A->col,     (size_t)A->nnz * sizeof(int32_t))
           || write_all(f, A->val,     (size_t)A->nnz * sizeof(double));
    if (fclose(f) != 0) err = 1;
    if (err) { fprintf(stderr, "%s: write error\n", path); return -1; }
    return 0;
}

int csr_read(const char *path, csr_t *A)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    char magic[8];
    int64_t hdr[3];
    if (read_all(f, magic, 8) || memcmp(magic, CSR_MAGIC, 8) != 0
        || read_all(f, hdr, sizeof hdr)) {
        fprintf(stderr, "%s: not a CSRBIN1 file\n", path);
        fclose(f); return -1;
    }
    A->M = hdr[0]; A->N = hdr[1]; A->nnz = hdr[2];

    A->row_ptr = malloc((size_t)(A->M + 1) * sizeof(int64_t));
    A->col     = malloc((size_t)A->nnz * sizeof(int32_t));
    A->val     = malloc((size_t)A->nnz * sizeof(double));
    if (!A->row_ptr || !A->col || !A->val) {
        fprintf(stderr, "out of memory\n"); fclose(f); return -1;
    }

    /* First touch: memory pages are placed near the core that first
       writes them. The kernels give each thread an equal share of the
       nonzeros, so touch col/val with the same equal split here, before
       fread fills them. (Matters on multi-socket machines; harmless on a
       laptop.) See van der Pas et al. 2017, chapter on affinity.        */
    #pragma omp parallel for schedule(static)
    for (int64_t k = 0; k < A->nnz; k++) {
        A->col[k] = 0;
        A->val[k] = 0.0;
    }

    int err = read_all(f, A->row_ptr, (size_t)(A->M + 1) * sizeof(int64_t))
           || read_all(f, A->col,     (size_t)A->nnz * sizeof(int32_t))
           || read_all(f, A->val,     (size_t)A->nnz * sizeof(double));
    fclose(f);
    if (err) { fprintf(stderr, "%s: truncated file\n", path); return -1; }
    return 0;
}

void csr_free(csr_t *A)
{
    free(A->row_ptr); free(A->col); free(A->val);
    A->row_ptr = NULL; A->col = NULL; A->val = NULL;
}
