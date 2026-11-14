#!/bin/sh
# run_cutoff.sh -- time quicksort_tasks for several cutoffs and write cutoff.dat
#
# Usage:   sh run_cutoff.sh [n] [threads]
#          sh run_cutoff.sh 20000000 $(nproc)
#
# cutoff.dat: plain text, one line per cutoff, columns separated by spaces
#   1 cutoff   2 serial time (s)   3 tasks time (s)   4 speed-up
# Lines starting with # are comments (ignored by gnuplot, numpy.loadtxt, ...).

N=${1:-20000000}
T=${2:-$(nproc)}

echo "# n = $N, threads = $T"           > cutoff.dat
echo "# cutoff serial_s tasks_s speedup" >> cutoff.dat
for c in 100 300 1000 3000 10000 30000 100000 300000 1000000 3000000 10000000 $N; do
    OMP_NUM_THREADS=$T ./quicksort_tasks $N $c |
        awk -v c=$c '/^serial/ {s = $3} /^tasks/ {print c, s, $3, $6}'
done >> cutoff.dat
cat cutoff.dat
