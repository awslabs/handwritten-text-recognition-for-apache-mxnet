#!/bin/sh

usage="$0 <stm validation script location>"

if [ "$#" != '1' ]; then
    echo Script missing
    echo $usage
    exit 1
fi

stm=$1
if [ ! -x "$stm" ]; then
	echo Script not executable
    echo $usage
    exit 1
fi

for file in test*.stm ; do
  if [ -f "$file.toskip" ] ; then
     echo "(skipping) $base ..."
  else
    echo "Testing $file..."
    base=`echo $file | perl -pe 's/.stm//'`
    log="$base.log.saved"
    tmp="$base.log.tmp"
    
    if [ ! -f "$log" ] ; then
		$stm -i $file > $log
    fi

    $stm -i $file > $tmp
    diff_status=`diff $log $tmp | wc -l`

    if [ $diff_status -ne 0 ] ; then
		echo "   Output log differs from saved log"
                exit 1
	else
		rm $tmp
    fi
  fi
done

exit 0
