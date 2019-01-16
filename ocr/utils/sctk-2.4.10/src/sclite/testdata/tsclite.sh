#!/bin/sh

base_dir=base
OUT=out
exe_dir=..
exe_name=sclite
clean=TRUE
DATA=.
SCLFLAGS="  "

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
			rm -rf $OUT Failed.log ; exit;;
		-purify_status)
			echo "Checking Purify Output"
			echo "    " `ls *.pure|wc -l` "Purify files"
			grep '^[A-Z][A-Z][A-Z]:' *pure | egrep -v 'FIU.*(stdin|stdout|stderr|reserved for Purif)' > /tmp/tsc.log
			for t in `awk -F: '{print $2}' /tmp/tsc.log | sort -u` ; do
				echo "   Type: $t  Count:" `grep $t /tmp/tsc.log|wc -l` " Tests:" `grep $t /tmp/tsc.log|awk -F. '{print $1}' | sort -u`
			done
			exit;;
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
if test "`grep "SLM_TARGETS =" ../makefile| sed 's/.*= *//'`" = "" ; then
	SLM_ENABLED=0
	echo "    SLM-Toolkit Disabled"
else
	SLM_ENABLED=1
	echo "    SLM-Toolkit Enabled"
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


scliteCom="$exe_dir/${exe_name}"
#scliteCom="valgrind --dsymutil=yes --tool=exp-sgcheck --tool=memcheck $exe_dir/${exe_name}"

doit(){
   testid="$1"
   desc="$2"  
   com="$3"
   pipeInput="$4";
   prereq="$5"

   echo "$testid: $desc"
   if [ "$prereq" = "SLM" ] ; then
        if test $SLM_ENABLED = 1 ; then
            $exe_dir/${exe_name} $com 1> $OUT/$testid.out 2> $OUT/$testid.err
        else
            echo "            **** SLM weighted alignment is disabled, not testing ***"
        fi   
   elif [ "$prereq" = "DIFF" ] ; then
        if test $DIFF_ENABLED = 1 ; then
            $exe_dir/${exe_name} $com 1> $OUT/$testid.out 2> $OUT/$testid.err
        else
            echo "            **** Diff alignments have been disabled, not testing ***"
        fi   
   elif [ "$pipeInput" = "" ] ; then
        $exe_dir/${exe_name} $com 1> $OUT/$testid.out 2> $OUT/$testid.err
   else 
        $exe_dir/${exe_name} $com < $pipeInput 1> $OUT/$testid.out 2> $OUT/$testid.err
   fi 

   if [ -f $OUT/$TEST.prf ] ; then
       grep -v 'Creation date:' < $OUT/$TEST.prf > x ; mv x $OUT/$TEST.prf
   fi
   if [ -f $OUT/$TEST.sgml ] ; then
       sed 's/creation_date="[^"]*"//' < $OUT/$TEST.sgml > x ; mv x $OUT/$TEST.sgml
   fi
   if [ -f $OUT/$TEST.nl.sgml ] ; then
       sed 's/creation_date="[^"]*"//' < $OUT/$TEST.nl.sgml > x ; mv x $OUT/$TEST.nl.sgml
   fi
}

TEST=test1
doit $TEST \
        "Align Both Ref and Hyp transcripts. (one transcript to a line followed by an utterance id in parens)" \
        "${SCLFLAGS} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj -o all snt spk dtl prf sgml nl.sgml -O $OUT -f 0 -n $TEST" \
        "" \
        ""
        
TEST=test1a
doit $TEST \
        "Same as test1, but generating an sgml file, then piping to sclite for reports" \
        "${SCLFLAGS} -P -o dtl prf -O $OUT -f 0 -n $TEST" \
        test1.sgml \
	""
	
TEST=test1b
doit $TEST \
        "Same as test1, but using a language model for weights" \
        "${SCLFLAGS} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj -L $DATA/csrnab_r.blm -o sum wws prf -O $OUT -f 0 -n $TEST" \
        "" \
        "SLM"
	
TEST=test1c
doit $TEST \
        "Same as test1, but using a language model for weights" \
        "${SCLFLAGS} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj -w $DATA/csrnab_r.wwl -o wws prf -O $OUT -f 0 -n $TEST" \
        "" \
        ""

TEST=test1d
doit $TEST \
        "Same as test1, but producing a nl.sgml file" \
        "${SCLFLAGS} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj -o nl.sgml -O $OUT -f 0 -n $TEST" \
        "" \
        ""

TEST=test1e
doit $TEST \
        "Same as test1 but with utf-8 1 bytes per char" \
        "${SCLFLAGS} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj -o all snt spk dtl prf sgml nl.sgml -O $OUT -f 0 -n $TEST -e utf-8" \
        "" \
        ""

TEST=test2
doit $TEST \
        "Same as Test 1, but use Diff instead of DP alignments" \
        "${SCLFLAGS} -r $DATA/csrnab.ref -h $DATA/csrnab.hyp -i wsj -o all -O $OUT -f 0 -n $TEST -d" \
        "" \
        "DIFF"

TEST=test3
doit $TEST \
        "Align Segmental Time marks (STM) to Conversation time marks (CTM)" \
        "${SCLFLAGS} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hyp.ctm ctm -o all lur prf -O $OUT -f 0 -n $TEST" \
        "" \
        ""

TEST=test3a
doit $TEST \
        "Align Segmental Time marks (STM) to Conversation time marks (CTM) using the stm tag IGNORE_TIME_SEGMENT_IN_SCORING" \
        "${SCLFLAGS} -r $DATA/lvc_refe.stm stm -h $DATA/lvc_hyp.ctm ctm -o all lur prf -O $OUT -f 0 -n $TEST" \
        "" \
        ""

TEST=test3b
doit $TEST \
        "Align Segmental Time marks (STM) to Conversation time marks (CTM) with confidence scores" \
        "${SCLFLAGS} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hypc.ctm ctm -o sum -O $OUT -f 0 -n $TEST" \
        "" \
        ""

TEST=test3c
doit $TEST \
        "Test the output generated in lur when ther is no reference data" \
        "${SCLFLAGS} -r $DATA/lvc_refm.stm stm -h $DATA/lvc_hypm.ctm ctm -o lur -O $OUT -f 0 -n $TEST" \
        "" \
        ""

# TEST Number 4
TN=4
TEST=test$TN
echo "Test $TN:     Same as test 3, but using diff for alignment"
if test $DIFF_ENABLED = 1 ; then 
    $scliteCom ${SCLFLAGS} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -d \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
else
    echo "            **** Diff alignments have been disabled, not testing ***"
fi

# TEST Number 5
TN=5
TEST=test$TN
echo "Test $TN:     Align STM to free formatted text (TXT)"
if test $DIFF_ENABLED = 1 ; then
    $scliteCom ${SCLFLAGS} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hyp.txt txt \
	-o all prf -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
    grep -v 'Creation date:' < out/$TEST.prf > x ; mv x out/$TEST.prf
else
    echo "            **** Diff alignments have been disabled, not testing ***"
fi

# TEST Number 6
TN=6
TEST=test$TN
echo "Test $TN:     Align Mandarin Chinese words using DIFF"
if test $DIFF_ENABLED = 1 ; then
    $scliteCom ${SCLFLAGS} -e gb -r $DATA/mand_ref.stm stm \
	-h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -d \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
else
    echo "            **** Diff alignments have been disabled, not testing ***"
fi

# TEST Number 7	
TN=7
TEST=test$TN
echo "Test $TN:     Run some test cases through"
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -F -D \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_r
TN=7_r
TEST=test$TN
echo "Test $TN:   Run some test cases through (reversing ref and hyp)"
$scliteCom ${SCLFLAGS} -h $DATA/tests.ref -r $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -F -D \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_1
TN=7_1
TEST=test$TN
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            not changing ASCII words"
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_2
TN=7_2
TEST=test$TN
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            changing ASCII words"
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex ASCIITOO \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_2a
TN=7_2a
TEST=test$TN
echo "Test $TN:  Run some test cases through using infered word boundaries,"
echo "            changing ASCII words, using algo2."
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo2 tests.lex ASCIITOO \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_3
TN=7_3
TEST=test$TN
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            not changing ASCII words and correct Fragments"
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex -F \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_4
TN=7_4
TEST=test$TN
echo "Test $TN:   Run some test cases through using infered word boundaries,"
echo "            changing ASCII words and correct Fragments"
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -S algo1 tests.lex ASCIITOO -F \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_5
TN=7_5
TEST=test$TN
echo "Test $TN:   Run some test cases through, character aligning them and"
echo "            removing hyphens"
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all -O $OUT -f 0 -n $TEST -c DH \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 7_6
TN=7_6
TEST=test$TN
echo "Test $TN:     Run some test cases through with utf-8 encoding"
$scliteCom ${SCLFLAGS} -r $DATA/tests.ref -h $DATA/tests.hyp -i spu_id \
	-o all sgml nl.sgml -O $OUT -f 0 -n $TEST -F -D -e utf-8 \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
sed 's/creation_date="[^"]*"//' < out/$TEST.sgml > x ; mv x out/$TEST.sgml
sed 's/creation_date="[^"]*"//' < out/$TEST.nl.sgml > x ; mv x out/$TEST.nl.sgml


for utf in utf8-2bytes utf8-3bytes utf8-4bytes ; do
    TEST=test7-$utf
    doit $TEST \
        "Same as test 1 but with $utf" \
        "${SCLFLAGS} -r $DATA/tests.ref.$utf -h $DATA/tests.hyp.$utf -i spu_id -o all sgml -O $OUT -f 0 -n $TEST -F -D -e utf-8" \
        "" \
        ""        
done

# TEST Number 8
TN=8
TEST=test$TN
echo "Test $TN:     Align transcripts as character alignments"
$scliteCom ${SCLFLAGS} -r $DATA/csrnab1.ref -h $DATA/csrnab1.hyp -i wsj \
	-o all -O $OUT -f 0 -n $TEST -c \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 9
TN=9
TEST=test$TN
echo "Test $TN:     Run the Mandarin, doing a character alignment"
$scliteCom ${SCLFLAGS} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all dtl prf -O $OUT -f 0 -n $TEST -c \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
grep -v 'Creation date:' < out/$TEST.prf > x ; mv x out/$TEST.prf

# TEST Number 9_1
TN=9_1
TEST=test$TN
echo "Test $TN:   Run the Mandarin, doing a character alignment, removing hyphens"
$scliteCom ${SCLFLAGS} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -c DH \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 10
TN=10
TEST=test$TN
echo "Test $TN:    Run the Mandarin, doing a character alignment, not effecting ASCII WORDS"
$scliteCom ${SCLFLAGS} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all prf -O $OUT -f 0 -n $TEST -c NOASCII \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
grep -v 'Creation date:' < out/$TEST.prf > x ; mv x out/$TEST.prf

# TEST Number 10_1
TN=10_1
TEST=test$TN
echo "Test $TN:  Run the Mandarin, doing a character alignment, not effecting ASCII WORDS"
echo "            Removing hyphens."
$scliteCom ${SCLFLAGS} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -c NOASCII DH \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 11
TN=11
TEST=test$TN
echo "Test $TN:    Run the Mandarin, doing the inferred word segmentation alignments, algo1"
$scliteCom ${SCLFLAGS} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -S algo1 mand.lex \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 12
TN=12
TEST=test$TN
echo "Test $TN:    Run the Mandarin, doing the inferred word segmentation alignments, algo1"
echo "            Scoring fragments as correct"
$scliteCom ${SCLFLAGS} -e gb -r $DATA/mand_ref.stm stm -h $DATA/mand_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -S algo1 mand.lex -F \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 13
TN=13
TEST=test$TN
echo "Test $TN:    Run alignments on two CTM files, using DP Word alignments"
$scliteCom ${SCLFLAGS} -r $DATA/tima_ref.ctm ctm -h $DATA/tima_hyp.ctm ctm \
	-o all prf -O $OUT -f 0 -n $TEST \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err
grep -v 'Creation date:' < out/$TEST.prf > x ; mv x out/$TEST.prf

# TEST Number 13_a
TN=13_a
TEST=test$TN
echo "Test $TN:  Run alignments on two CTM files, using Time-Mediated DP alignments"
$scliteCom ${SCLFLAGS} -r $DATA/tima_ref.ctm ctm -h $DATA/tima_hyp.ctm ctm \
	-o all -O $OUT -f 0 -n $TEST -T \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 14_a
TN=14_a
TEST=test$TN
echo "Test $TN:  Reduce the ref and hyp input files to the intersection of the inputs"
$scliteCom ${SCLFLAGS} -r $DATA/lvc_refr.stm stm -h $DATA/lvc_hypr.ctm ctm \
	-o all lur -O $OUT -f 0 -n $TEST -m ref hyp\
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 14_b
TN=14_b
TEST=test$TN
echo "Test $TN:  Reduce the ref and hyp input files to the intersection of the inputs"
echo "            Using a reduced size hyp file"
$scliteCom ${SCLFLAGS} -r $DATA/lvc_ref.stm stm -h $DATA/lvc_hypr.ctm ctm \
	-o all lur -O $OUT -f 0 -n $TEST -m ref hyp\
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 14_c
TN=14_c
TEST=test$TN
echo "Test $TN:  Reduce the ref and hyp input files to the intersection of the inputs"
echo "            Using a reduced size hyp file"
$scliteCom ${SCLFLAGS} -r $DATA/lvc_refr.stm stm -h $DATA/lvc_hyp.ctm ctm \
	-o all lur -O $OUT -f 0 -n $TEST -m ref hyp \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 14_d
TN=14_d
TEST=test$TN
echo "Test $TN:  Reduce the ref and hyp input files to the intersection of the inputs"
echo "            Using a reduced size hyp and ref file"
$scliteCom ${SCLFLAGS} -r $DATA/lvc_refr.stm stm -h $DATA/lvc_hypr.ctm ctm \
	-o all lur -O $OUT -f 0 -n $TEST -m hyp \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

# TEST Number 14_e
TN=14_e
TEST=test$TN
echo "Test $TN:  Reduce the ref and hyp input files to the intersection of the inputs"
echo "            Using a reduced size hyp and ref file"
$scliteCom ${SCLFLAGS} -r $DATA/lvc_refr.stm stm -h $DATA/lvc_hypr.ctm ctm \
	-o all lur -O $OUT -f 0 -n $TEST -m ref \
	1> $OUT/$TEST.out 2> $OUT/$TEST.err

TEST=test15_a
doit $TEST \
        "UTF-8 test - Cantonese no options" \
        "${SCLFLAGS} -r $DATA/test.cantonese.stm stm -h $DATA/test.cantonese.ctm ctm -o all prf -O $OUT -f 0 -n $TEST -e utf-8" \
        "" \
        ""

TEST=test15_b
doit $TEST \
        "UTF-8 test - Cantonese no options - Character scoring" \
        "${SCLFLAGS} -r $DATA/test.cantonese.stm stm -h $DATA/test.cantonese.ctm ctm -o all prf -O $OUT -f 0 -n $TEST -e utf-8 -c NOASCII DH" \
        "" \
        ""

TEST=test15_c
doit $TEST \
        "UTF-8 test - UTF-8 Turkish" \
        "${SCLFLAGS} -r $DATA/test.turkish.ref trn -h $DATA/test.turkish.hyp -o all prf -O $OUT -f 0 -n $TEST -e utf-8 babel_turkish -i spu_id" \
        "" \
        ""

# TEST Number 16_X
n=1
for hyp in stm2ctm_missing.hyp-extra.ctm stm2ctm_missing.hyp-missall.ctm stm2ctm_missing.hyp-missfile1.ctm stm2ctm_missing.hyp-missfile1chanA.ctm stm2ctm_missing.hyp-missfile1chanb.ctm stm2ctm_missing.hyp-missfile2.ctm stm2ctm_missing.hyp-missfile2chanA.ctm stm2ctm_missing.hyp-missfile2chanB.ctm stm2ctm_missing.hyp.ctm ; do
    TEST=test16_$n
    doit $TEST \
        "Allow incomplete hyp CTM files - $hyp" \
        "${SCLFLAGS} -r $DATA/stm2ctm_missing.ref.stm stm -h $DATA/$hyp ctm -o all prf -O $OUT -f 0 -n $TEST " \
        "" \
        ""
    n=`expr $n + 1`
done

# TEST Number 17
TEST=test17
doit $TEST \
    "Vietnamese case conversion" \
    "${SCLFLAGS} -r $DATA/test.vietnamese.ref.trn trn -h test.vietnamese.hyp.trn trn -i spu_id -o all prf -O $OUT -f 0 -n $TEST -e utf-8 babel_vietnamese " \
    "" \
    ""

echo ""
echo "Executions complete: Comparing output"
filter="diff -r $base_dir $OUT | grep -v CVS"
vfilter="diff -c -r $base_dir $OUT | grep -v CVS"
if test $DIFF_ENABLED = 0 ; then
    echo "   Removing DIFF tests"
    filter="$filter | grep -ve 'test[2456]\.'"
    vfilter="$vfilter | grep -ve 'test[2456]\.'"
fi
if test $SLM_ENABLED = 0 ; then
    echo "   Removing SLM tests"
    filter="$filter | grep -ve 'test1b\.'"
    vfilter="$vfilter | grep -ve 'test1b\.'"
fi
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
    grep 'diff -c -r' Failed.log | awk '{print $5}'
    echo ""
    echo "Read Failed.log"
    eval $vfilter > Failed.log
    exit 1
fi

