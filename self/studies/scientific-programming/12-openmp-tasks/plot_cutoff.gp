# plot_cutoff.gp -- plot cutoff.dat (written by run_cutoff.sh)
#
# Usage:   gnuplot plot_cutoff.gp      -> cutoff.png
# Columns of cutoff.dat: 1 cutoff, 2 serial time, 3 tasks time, 4 speed-up

set terminal pngcairo size 800,700 font ",11"
set output 'cutoff.png'

set logscale x
set format x "10^{%L}"
set xrange [50:*]
set grid
# first line of cutoff.dat is "# n = ..., threads = ...": use it as the title
info = system("sed -n '1s/^# //p' cutoff.dat")
set multiplot layout 2,1 title info

set ylabel 'time (s)'
set yrange [0:*]
set key bottom left
plot 'cutoff.dat' using 1:3 with linespoints pt 7 title 'tasks', \
     ''           using 1:2 with linespoints pt 6 dt 2 title 'serial'

set xlabel 'cutoff (elements)'
set ylabel 'speed-up'
set yrange [0:*]
set key bottom left
plot 'cutoff.dat' using 1:4 with linespoints pt 5 title 'serial / tasks'

unset multiplot
