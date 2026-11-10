CC      = gcc
CFLAGS  = -O3 -march=native -fopenmp -Wall -Wextra
LDLIBS  = -lm

all: mtx2csr spmv_bench gen_graph

mtx2csr: mtx2csr.c csr.c csr.h
	$(CC) $(CFLAGS) -o $@ mtx2csr.c csr.c

spmv_bench: spmv_bench.c spmv_kernels.c csr.c spmv_kernels.h csr.h
	$(CC) $(CFLAGS) -o $@ spmv_bench.c spmv_kernels.c csr.c $(LDLIBS)

gen_graph: gen_graph.c
	$(CC) $(CFLAGS) -o $@ gen_graph.c

# Correctness check on the 6x6 example of the notes: y must be 20 6 0 27 4 56
test: mtx2csr spmv_bench
	./mtx2csr example.mtx example.csr
	OMP_NUM_THREADS=4 ./spmv_bench example.csr 1

clean:
	rm -f mtx2csr spmv_bench gen_graph example.csr

.PHONY: all test clean
