/*
 * mtx2csr.c -- convert a Matrix Market file to binary CSR (run once).
 *
 * Usage:   ./mtx2csr input.mtx output.csr
 *
 * Supports "matrix coordinate" files that are real, integer or pattern,
 * and general or symmetric. Symmetric files store only one triangle;
 * the mirror entry a_ji is added here for every off-diagonal a_ij.
 * Pattern files have no values; every stored entry gets the value 1.0.
 *
 * Memory: no temporary COO copy. The file is read TWICE:
 *   pass 1: count the entries of each row          -> row_ptr
 *   pass 2: put each entry directly in its place   -> col, val
 * so the peak memory is just the final CSR: 12*nnz + 8*(M+1) bytes.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdint.h>
#include "csr.h"

typedef struct {
    int symmetric;   /* 1: store only one triangle, mirror it      */
    int pattern;     /* 1: no values in the file                   */
} mm_kind_t;

/* Read the banner line and skip comments. Leaves f at the first entry.
   Returns the number of entries stored in the file.                  */
static int64_t read_header(FILE *f, mm_kind_t *kind, int64_t *M, int64_t *N)
{
    char line[1024], obj[64], fmt[64], field[64], sym[64];

    if (!fgets(line, sizeof line, f)
        || sscanf(line, "%%%%MatrixMarket %63s %63s %63s %63s", obj, fmt, field, sym) != 4
        || strcasecmp(obj, "matrix") != 0 || strcasecmp(fmt, "coordinate") != 0) {
        fprintf(stderr, "not a Matrix Market coordinate matrix\n");
        exit(1);
    }
    if (strcasecmp(field, "complex") == 0) {
        fprintf(stderr, "complex matrices are not supported\n");
        exit(1);
    }
    kind->pattern   = (strcasecmp(field, "pattern") == 0);
    kind->symmetric = (strcasecmp(sym, "symmetric") == 0);
    if (!kind->symmetric && strcasecmp(sym, "general") != 0) {
        fprintf(stderr, "symmetry '%s' is not supported\n", sym);
        exit(1);
    }

    do {                                   /* skip comment lines */
        if (!fgets(line, sizeof line, f)) { fprintf(stderr, "no size line\n"); exit(1); }
    } while (line[0] == '%');

    long long m, n, e;
    if (sscanf(line, "%lld %lld %lld", &m, &n, &e) != 3) {
        fprintf(stderr, "bad size line\n"); exit(1);
    }
    if (m >= INT32_MAX || n >= INT32_MAX) {
        fprintf(stderr, "more than 2^31 rows/columns: col must be int64\n"); exit(1);
    }
    *M = m; *N = n;
    return e;
}

/* Parse one entry "i j [value]" (1-based). Returns 0 at end of file.
   Hand-written integer parsing: much faster than sscanf on 10^8 lines. */
static int read_entry(FILE *f, char **buf, size_t *cap, int pattern,
                      int64_t *i, int64_t *j, double *v)
{
    ssize_t len;
    do {
        len = getline(buf, cap, f);
        if (len < 0) return 0;
    } while ((*buf)[0] == '%' || (*buf)[0] == '\n');

    char *p = *buf;
    int64_t a = 0, b = 0;
    while (*p == ' ' || *p == '\t') p++;
    while (*p >= '0' && *p <= '9') a = a * 10 + (*p++ - '0');
    while (*p == ' ' || *p == '\t') p++;
    while (*p >= '0' && *p <= '9') b = b * 10 + (*p++ - '0');
    *i = a - 1;                        /* Matrix Market is 1-based */
    *j = b - 1;
    *v = pattern ? 1.0 : strtod(p, NULL);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s input.mtx output.csr\n", argv[0]);
        return 1;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) { perror(argv[1]); return 1; }

    mm_kind_t kind;
    csr_t A;
    int64_t entries = read_header(f, &kind, &A.M, &A.N);
    long data_start = ftell(f);
    printf("%s: %lld x %lld, %lld stored entries, %s, %s\n", argv[1],
           (long long)A.M, (long long)A.N, (long long)entries,
           kind.pattern ? "pattern" : "real", kind.symmetric ? "symmetric" : "general");

    char *buf = NULL; size_t cap = 0;
    int64_t i, j; double v;

    /* ---- pass 1: count entries per row, in row_ptr[i+1] ---- */
    A.row_ptr = calloc((size_t)(A.M + 1), sizeof(int64_t));
    if (!A.row_ptr) { fprintf(stderr, "out of memory\n"); return 1; }
    int64_t seen = 0;
    while (read_entry(f, &buf, &cap, 1, &i, &j, &v)) {
        if (i < 0 || i >= A.M || j < 0 || j >= A.N) {
            fprintf(stderr, "entry %lld out of range\n", (long long)seen + 1); return 1;
        }
        A.row_ptr[i + 1]++;
        if (kind.symmetric && i != j) A.row_ptr[j + 1]++;   /* mirror a_ji */
        seen++;
    }
    if (seen != entries) {
        fprintf(stderr, "expected %lld entries, found %lld\n",
                (long long)entries, (long long)seen);
        return 1;
    }

    /* prefix sum: row_ptr[i] = start of row i, row_ptr[M] = nnz */
    for (int64_t r = 0; r < A.M; r++) A.row_ptr[r + 1] += A.row_ptr[r];
    A.nnz = A.row_ptr[A.M];
    printf("nnz after symmetric expansion: %lld  (CSR size %.2f GB)\n",
           (long long)A.nnz, (12.0 * A.nnz + 8.0 * (A.M + 1)) / 1e9);

    A.col = malloc((size_t)A.nnz * sizeof(int32_t));
    A.val = malloc((size_t)A.nnz * sizeof(double));
    if (!A.col || !A.val) { fprintf(stderr, "out of memory\n"); return 1; }

    /* ---- pass 2: fill col/val ----
       row_ptr[i] is used as the "next free slot" of row i. After the pass
       it points to the END of row i, so it is shifted back afterwards.  */
    fseek(f, data_start, SEEK_SET);
    while (read_entry(f, &buf, &cap, kind.pattern, &i, &j, &v)) {
        int64_t p = A.row_ptr[i]++;
        A.col[p] = (int32_t)j; A.val[p] = v;
        if (kind.symmetric && i != j) {
            int64_t q = A.row_ptr[j]++;
            A.col[q] = (int32_t)i; A.val[q] = v;
        }
    }
    for (int64_t r = A.M; r > 0; r--) A.row_ptr[r] = A.row_ptr[r - 1];
    A.row_ptr[0] = 0;

    free(buf);
    fclose(f);

    if (csr_write(argv[2], &A) != 0) return 1;
    printf("wrote %s\n", argv[2]);
    csr_free(&A);
    return 0;
}
