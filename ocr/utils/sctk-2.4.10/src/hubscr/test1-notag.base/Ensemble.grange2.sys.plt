set yrange [3:0]
set xrange [0:100]
set title ""
set key
set ylabel "Systems"
set xlabel "Speaker Word Error Rate (%)"
set ytics ("lvc_hyp.notag.ctm" 1,"lvc_hyp2.notag.ctm" 2)
plot "Ensemble.grange2.sys.mean" using 2:1 title "Mean Speaker Word Error Rate (%)" with lines,\
     "Ensemble.grange2.sys.median" using 2:1 title "Median Speaker Word Error Rate (%)" with lines,\
     "Ensemble.grange2.sys.dat" using 2:1 "%f%f" title "inter_segment_gap",\
     "Ensemble.grange2.sys.dat" using 2:1 "%f%*s%f" title "2347-a",\
     "Ensemble.grange2.sys.dat" using 2:1 "%f%*s%*s%f" title "2347-b",\
     "Ensemble.grange2.sys.dat" using 2:1 "%f%*s%*s%*s%f" title "3129-a",\
     "Ensemble.grange2.sys.dat" using 2:1 "%f%*s%*s%*s%*s%f" title "3129-b"
