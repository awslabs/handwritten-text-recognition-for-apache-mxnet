#! /bin/sh

base_dir=base_sc_stats
OUT=out
exe_dir=..
exe_name=sc_stats
clean=TRUE
DATA=.
SCLFLAGS="  "

###
### File: tsc_stats.sh
### Usage: tsc_stats.sh [ -en exe_name | -ed exe_dir | -nc | -clean ]
###
###
for i in $*
do
	case $i in
        	-nc) clean="FALSE";;
                -en) exe_name=$2;;
                -ed) exe_dir=$2;;
		-clean) echo "Cleaning out tsc_stats.sh's directory"
			rm -rf $OUT Failed.log *.pvc *.pure *.pv; rclean-up; exit;;
                *) break;;
        esac
	shift;
done

echo "tsc_stats.sh -- Version 1.0"
echo "Variables:"
echo "    reference directory  =" $base_dir
echo "    executable directory =" $exe_dir
echo "    sc_stats executable    =" $exe_name
echo "" 

if [ -d $OUT ] ; then
	echo "Shall I delete the output directory \"$OUT\"'s contents [y]"
	read ans
	if [ "$ans" = "n" -o "$ans" = "N" ] ; then
		echo "    OK, leaving files inplace"
	else
		echo "    Erasing the output directory"
		rm -rf $OUT
		mkdir $OUT
	fi
	echo ""
else
	mkdir $OUT
fi

# TEST Number 1
echo "Test 1a+1b:     Symmetric tests on MAPSSWE"
TN=1a
TEST=test$TN
cat file1.sgml file2.sgml | $exe_dir/$exe_name -p -t mapsswe -v -n $OUT/$TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

TN=1b
TEST=test$TN
cat file2.sgml file1.sgml | $exe_dir/$exe_name -p -t mapsswe -v -n $OUT/$TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err




echo ""
echo "Executions complete: Comparing output"
filter="diff -r $base_dir $OUT | grep -v CVS"
vfilter="diff -c -r $base_dir $OUT | grep -v CVS"
echo ""

if test "`eval $filter`" = "" ; then
    echo "ALL TESTS SUCCESSFULLY COMPLETED"
    if [ $clean = "TRUE" ] ; then
	rm -r $OUT
    fi
    exit 0
else
    echo "     !!!!!  TESTS HAVE FAILED  !!!!!"
    echo ""
    echo "Read Failed.log"
    eval $vfilter > Failed.log
    exit 1
fi

