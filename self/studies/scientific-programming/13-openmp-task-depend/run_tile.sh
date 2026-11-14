#!/bin/sh
# run_tile.sh -- time trisolve_tasks for several tile sizes and write tile.dat
#
# Usage:   sh run_tile.sh [m] [threads]
#          sh run_tile.sh 4000 $(nproc)
#
# tile.dat columns: 1 tile size B   2 serial (s)   3 levels (s)   4 tasks (s)
# Lines starting with # are comments.

M=${1:-4000}
T=${2:-$(nproc)}

echo "# grid = $M x $M, threads = $T"   > tile.dat
echo "# B serial_s levels_s tasks_s"    >> tile.dat
for B in 16 32 64 128 256 512 1000 2000; do
    OMP_NUM_THREADS=$T ./trisolve_tasks $M $B |
        awk -v B=$B '/^serial/ {s = $3} /^levels/ {l = $3} /^tasks/ {print B, s, l, $3}'
done >> tile.dat
cat tile.dat
