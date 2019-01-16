## GNUPLOT command file
set samples 1000
set key 30.000000,90.000000
set xrange [0:1]
set yrange [0:100]
set ylabel '% Hypothesis Correct'
set xlabel 'Confidence Scores'
set title  'Scaled Binned Confidence scores for sastt-case2.sys.rttm.filt'
set nogrid
set size 0.78,1
set nolabel
plot 'sastt-case2.sys.rttm.filt.sbhist.dat'  title 'True' with boxes, x*100 title 'Expected'
set size 1.0, 1.0
set key
