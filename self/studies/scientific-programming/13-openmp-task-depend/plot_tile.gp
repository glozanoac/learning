# plot_tile.gp -- plot tile.dat (written by run_tile.sh)
#
# Usage:   gnuplot plot_tile.gp      -> tile.png
# Columns of tile.dat: 1 B, 2 serial, 3 levels, 4 tasks (seconds)

set terminal pngcairo size 800,450 font ",11"
set output 'tile.png'

set logscale x 2
set grid
set title system("sed -n '1s/^# //p' tile.dat")
set xlabel 'tile size B (grid points per side)'
set ylabel 'speed-up over serial'
set yrange [0:*]
set key top left
plot 'tile.dat' using 1:($2/$4) with linespoints pt 7 title 'tasks (depend)', \
     ''         using 1:($2/$3) with linespoints pt 6 dt 2 title 'levels (barrier)', \
     1 with lines lc 'black' dt 3 notitle
