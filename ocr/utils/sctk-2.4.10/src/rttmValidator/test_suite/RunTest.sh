#!/bin/sh

usage="$0 <rttm validation script location> [-v]"

if [ "$#" -lt '1' ]; then
    echo Script missing
    echo $usage
    exit 1
fi

rttm=$1
if [ ! -x "$rttm" ]; then
	echo Script not executable
    echo $usage
    exit 1
fi

if [ "$2" = '-v' ] ; then 
    verbose=true
else
    verbose=false
fi

for file in test*.rttm ; do
  if [ -f "$file.toskip" ] ; then
     echo "(skipping) Testing $file..."
  else

    echo "Testing $file..."
    base=`echo $file | perl -pe 's/.rttm//'`
    log="$base.log.saved"
    tmp="$base.log.tmp"
    
    if [ ! -f "$log" ] ; then
		$rttm -S -t -i $file > $log
    fi

    $rttm -S -t -i $file > $tmp
    diff_status=`diff -I 'RTTMValidator' $log $tmp | wc -l`

    if [ $diff_status -ne 0 ] ; then
		echo "   Output log differs from saved log"
		
		if [ $verbose = true ] ; then
			diff -I 'RTTMValidator' $log $tmp | sed 's/^/   /'
		fi
                # exit with error status
                exit 1
	else
		rm $tmp
    fi
  fi
done

# exit with ok status
exit 0
