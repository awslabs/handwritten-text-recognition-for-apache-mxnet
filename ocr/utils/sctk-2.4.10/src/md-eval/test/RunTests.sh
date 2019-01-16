#!/bin/sh

usage="$0 [-v] <mdeval pgm path>"

if [ "$1" = "-v" ] ; then
    verbose=true
    shift;
else
    verbose=false
fi

if [ $# -ne 1 ]; then
    echo $usage
    exit
fi

mdeval_pgm=$1

for file in *.ref.rttm sd_test*.ref.rttm ; do
#    head -1 $file
    base=`echo $file | perl -pe 's/\..*//'`
  if [ -f "$file.toskip" ] ; then
     echo "(skipping) $base ..."
  else
    base=`echo $file | perl -pe 's/\..*//'`

    if [ ! "`echo $file | grep 'md_test'`" = "" ] ; then
        # Structural metadata extraction tests
	com="$mdeval_pgm -af -e -D -d -W -w -t 1.0 -l 2 -u $base.uem -r $base.ref.rttm -s $base.sys.rttm"
    elif [ ! "`echo $file | grep 'sd_test'`" = "" ] ; then
        # Speaker diarization tests
	com="$mdeval_pgm -1 -v -e -d -D -m -af -c 0.0 -T 0.0 -u $base.uem -r $base.ref.rttm -s $base.sys.rttm"
    else 
	echo "Hey!!!! I can't run $file"
	exit 1;
    fi

    echo -n "Running $base: "
    if [ ! -f "$base.output.saved" ] ; then
	echo "   Generating Saved file, not Testing"
	$com 1> $base.output.saved 2> $base.output.saved.log
    else 
	$com 1> $base.output.tmp 2> $base.output.tmp.log

	if [ -f $base.output.tmp ] ; then
	    diff_status=`diff -b -B -I ' Performance analysis for ' -I 'md-eval run on ' -I 'command line ' $base.output.saved $base.output.tmp | wc -l`
	    if [ $diff_status -ne 0 ] ; then
		echo "*******  report output differs  *******"
		if [ "$verbose" = "true" ] ; then
		    diff -b -B -I ' Performance analysis for ' -I 'md-eval run on ' -I 'command line ' $base.output.saved $base.output.tmp | sed 's/^/     /'
		fi
	    else
#            echo "    newly created output is identical to saved output"
		rm $base.output.tmp
	    fi
	    diff_status=`diff -b -B -I ' Performance analysis for ' -I 'md-eval run on ' -I 'command line ' $base.output.saved.log $base.output.tmp.log | wc -l`
	    if [ $diff_status -ne 0 ] ; then
		echo "*******  stderr log output differs  *******"
		if [ "$verbose" = "true" ] ; then
		    diff -b -B -I ' Performance analysis for ' -I 'md-eval run on ' -I 'command line ' $base.output.saved.log $base.output.tmp.log | sed 's/^/     /'
		fi
	    else
#            echo "    newly created output is identical to saved output"
		rm $base.output.tmp.log
	    fi
	    
	fi
    fi
    echo
    echo
  fi
done

exit 0
