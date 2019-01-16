#!/bin/sh
 
SLM=../slm_v2/bin
 
perl -pe 's/\([^()]+\)\s*$/\n/; s/{//; s:/[^}]+}::; s/@//;' csrnab.ref | tr A-Z a-z | \
    $SLM/text2wfreq | $SLM/wfreq2vocab > csrnab_r.vocab
perl -pe 's/\([^()]+\)\s*$/\n/; s/{//; s:/[^}]+}::; s/@//;' csrnab.ref | tr A-Z a-z | \
    $SLM/text2idngram -vocab csrnab_r.vocab -n 2 > csrnab_r.idngram
$SLM/idngram2lm -idngram csrnab_r.idngram -n 2 -vocab csrnab_r.vocab -binary csrnab_r.blm
#rm -f csrnab_r.idngram csrnab_r.voc csrnab_r.vocab

