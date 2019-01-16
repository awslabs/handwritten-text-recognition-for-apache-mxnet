set yrange [6:0]
set xrange [0:100]
set title ""
set key
set ylabel "Speaker ID"
set xlabel "Speaker Word Error Rate (%)"
set ytics ("INTER_SEGMENT_GAP" 1,"2347-B" 2,"2347-A" 3,"3129-B" 4,"3129-A" 5)
plot "Ensemble.grange2.spk.mean" using 2:1 title "Mean Speaker Word Error Rate (%)" with lines,\
     "Ensemble.grange2.spk.median" using 2:1 title "Median Speaker Word Error Rate (%)" with lines,\
     "Ensemble.grange2.spk.dat" using 2:1 "%f%f" title "lvc_hyp.notag.ctm",\
     "Ensemble.grange2.spk.dat" using 2:1 "%f%*s%f" title "lvc_hyp2.notag.ctm"
