#! /bin/sh

base_dir=base
OUT=out
exe_dir=..
exe_name=sclite
clean=TRUE
DATA=.

###
### File: tsclite.sh
### Usage: tsclite.sh [ -en exe_name | -nc | -clean ]
###
###
for i in $*
do
	case $i in
        	-nc) clean="FALSE";;
                -en) exe_name=$2;;
		-clean) echo "Cleaning out tsclite.sh's directory"
			rm -rf $OUT Failed.log *.pvc *.pure *.pv; rclean-up; exit;;
                *) break;;
        esac
	shift;
done

echo "tsclite.sh -- Version 1.2"
echo "Variables:"
echo "    reference directory  =" $base_dir
echo "    executable directory =" $exe_dir
echo "    sclite executable    =" $exe_name
if test "`grep DIFF_EXE ../makefile`" = "" ; then
	DIFF_ENABLED=0
	echo "    Diff Alignments Disabled"
else
	DIFF_ENABLED=1
	echo "    Diff Alignments Enabled"
fi
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
TN=1
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Align Both Ref and Hyp transcripts"
echo "            (one transcript to a line followed by an utterance id in parens)"
$exe_dir/${exe_name} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj \
	-o all snt spk dtl -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
### This is a kludge to rename the *spk* and *snt* files to fit into 8.3 names
for f in out/test1.spk* ; do
	mv $f `echo $f |sed 's/test1.spk/test1_sp/'`
done
for f in out/test1.snt* ; do
	mv $f `echo $f |sed 's/test1.snt/test1_st/'`
done

# TEST Number 2
TN=2
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Same as Test 1, but use Diff instead of DP alignments"
if test $DIFF_ENABLED = 1 ; then
    $exe_dir/${exe_name} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj \
	-o all -O $OUT -f 0 -n $TEST -d \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
else
    echo "            **** Diff alignments have been disabled, not testing ***"
fi


# TEST Number 3
TN=3
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Align Segmental Time marks (STM) to "
echo "            Conversation time marks (CTM)"
$exe_dir/${exe_name} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hyp.ctm ctm \
	-o all lur -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 4
TN=4
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Same as test 3, but using diff for alignment"
if test $DIFF_ENABLED = 1 ; then 
    $exe_dir/${exe_name} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -d \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
else
    echo "            **** Diff alignments have been disabled, not testing ***"
fi

# TEST Number 5
TN=5
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Align STM to free formatted text (TXT)"
if test $DIFF_ENABLED = 1 ; then
    $exe_dir/${exe_name} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hyp.txt txt \
	-o all -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
else
    echo "            **** Diff alignments have been disabled, not testing ***"
fi

# TEST Number 6
TN=6
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Align Mandarin Chinese words using DIFF"
if test $DIFF_ENABLED = 1 ; then
    $exe_dir/${exe_name} -e gb -r $DATA/mand_ref.stm stm \
	-h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -d \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
else
    echo "            **** Diff alignments have been disabled, not testing ***"
fi

# TEST Number 7	
TN=7
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Run some test cases through"
$exe_dir/${exe_name} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_r
TN=7_r
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:   Run some test cases through (reversing ref and hyp)"
$exe_dir/${exe_name} -h $DATA/tests.ref -r $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_1
TN=7_1
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            not changing ASCII words"
$exe_dir/${exe_name} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_2
TN=7_2
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            changing ASCII words"
$exe_dir/${exe_name} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex ASCIITOO \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_3
TN=7_3
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            not changing ASCII words and correct Fragments"
$exe_dir/${exe_name} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex -F \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_4
TN=7_4
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            changing ASCII words and correct Fragments"
$exe_dir/${exe_name} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex ASCIITOO -F \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_5
TN=7_5
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:   Run some test cases through, character aligning them and"
echo "            removing hyphens"
$exe_dir/${exe_name} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -c DH \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 8
TN=8
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Align transcripts as character alignments"
$exe_dir/${exe_name} -r $DATA/csrnab1.ref -h $DATA/csrnab1.hyp -i wsj \
	-o all -O $OUT -f 0 -n $TEST -c \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 9
TN=9
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:     Run the Mandarin, doing a character alignment"
$exe_dir/${exe_name} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all dtl -O $OUT -f 0 -n $TEST -c \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 9_1
TN=9_1
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:   Run the Mandarin, doing a character alignment, removing hyphens"
$exe_dir/${exe_name} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -c DH \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 10
TN=10
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:    Run the Mandarin, doing a character alignment, not effecting ASCII WORDS"
$exe_dir/${exe_name} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -c NOASCII \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 10_1
TN=10_1
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:  Run the Mandarin, doing a character alignment, not effecting ASCII WORDS"
echo "            Removing hyphens."
$exe_dir/${exe_name} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -c NOASCII DH \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 11
TN=11
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:    Run the Mandarin, doing the inferred word segmentation alignments"
$exe_dir/${exe_name} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -S algo1 mand.lex \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 12
TN=12
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:    Run the Mandarin, doing the inferred word segmentation alignments"
echo "            Scoring fragments as correct"
$exe_dir/${exe_name} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -S algo1 mand.lex -F \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 13
TN=13
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:    Run alignments on two CTM files, using DP Word alignments"
$exe_dir/${exe_name} -r $DATA/tima_ref.ctm ctm -h $DATA/tima_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err


# TEST Number 13_a
TN=13_a
TEST=test$TN
PURECOVOPTIONS="-counts-file=`pwd`/$TEST.sclite.pvc"; export PURECOVOPTIONS
PURIFYOPTIONS="-log-file=`pwd`/$TEST.sclite.pure -view-file=`pwd`/$TEST.sclite.pv";
export PURIFYOPTIONS
echo "Test $TN:  Run alignments on two CTM files, using Time-Mediated DP alignments"
$exe_dir/${exe_name} -r $DATA/tima_ref.ctm ctm -h $DATA/tima_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -T \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

echo ""
echo "Executions complete: Comparing output"
echo ""

if test $DIFF_ENABLED = 1 ; then
	if test "`diff -r $base_dir $OUT`" = "" ; then
		echo "ALL TESTS SUCCESSFULLY COMPLETED"
		if [ $clean = "TRUE" ] ; then
			rm -r $OUT
		fi
		exit 0
	else
		echo "     !!!!!  TESTS HAVE FAILED  !!!!!"
		echo ""
		echo "Read Failed.log"
		diff -c -r $base_dir $OUT > Failed.log
		exit 1
	fi
else
	if test "`diff -r $base_dir $OUT | grep -ve 'test[2456]'`" = "" ; then
		echo "ALL TESTS SUCCESSFULLY COMPLETED"
		if [ $clean = "TRUE" ] ; then
			rm -r $OUT
		fi
		exit 0
	else
		echo "     !!!!!  TESTS HAVE FAILED  !!!!!"
		echo ""
		echo "Read Failed.log"
		diff -c -r $base_dir $OUT | grep -ve 'test[2456]' > Failed.log
		exit 1
	fi
fi

exit 0
