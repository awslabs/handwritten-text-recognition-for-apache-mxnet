#!/usr/bin/perl -w

use strict;
my $operation = (defined($ARGV[0]) ? $ARGV[0] : "test");

####################
sub error_exit { exit(1); }
sub error_quit { print('[ERROR] ', join(' ', @_), "\n"); &error_exit(); }
sub ok_exit { exit(0); }
sub ok_quit { print(join(' ', @_), "\n"); &ok_exit(); }
####################

sub runIt{
    my ($op, $testId, $options, $glm, $hub, $lang, $ref, $systems) = @_;
    my $baseDir = $testId.".base";
    my $outDir = $testId.($op eq "setTests" ? ".base" : ".test");
    print "   Running test '$testId', operation '$op', options '$options', directory '$outDir'\n";
    system ("mkdir -p $outDir");
    system ("rm -fr $outDir/test* $outDir/lvc*");
    ### Copy files
    foreach my $file($glm, $ref, split(/\s+/,$systems)){
	system("cp $file $outDir");
    }
    ### make all file names relative to the Outdir
    my ($refRoot) = $ref; $refRoot =~ s:.*/::;
    my ($glmRoot) = $glm; $glmRoot =~ s:.*/::;
    my ($systemsRoot) = "";
    foreach $_(split(/\s+/, $systems)){
	$_ =~ s:.*/::;
	$systemsRoot .= " ".$_;
    }
    print "      Executing command\n";
    my $com = "(cd $outDir; ../hubscr.pl $options -p ../../rfilter1:../../asclite/core:../../csrfilt:../../def_art:../../acomp:../../hamzaNorm:../../tanweenFilt:../../sclite:../../md-eval:../../rttmSort:../../align2html:../../stm2rttm:../../ctmValidator:../../stmValidator:../../rttmValidator ".
	"-l $lang -g $glmRoot -h $hub -r $refRoot $systemsRoot > log)";
#    print  "$com";
    my $ret = system "$com";
    &error_quit("Execution failed") if ($ret != 0);
    if ($op ne "setTests"){
	print "      Comparing output\n";
	my $diffoption = "";
			
	if($options eq "-a") { $diffoption = "-i"; }
			
	my $diffCom = "diff -i -x CVS -x .DS_Store -x log -x \*lur -I '[cC]reation[ _]date' -I 'md-eval' -r $outDir $baseDir";
	open (DIFF, "$diffCom |") || &error_quit("Diff command '$diffCom' Failed");
	my @diff = <DIFF>;
	close DIFF;
	&error_quit("Test $testId has failed.  Diff output is :\n@diff\n") if (@diff > 0);
	print "      Successful Test.  Removing $outDir\n";
	system "rm -rf $outDir";
    }
}

runIt($operation, "test1-sastt", "-G -f rttm -F rttm -a", "../test_suite/example.glm", "sastt", "english",
      "../test_suite/sastt-case1.ref.rttm", "../test_suite/sastt-case1.sys.rttm");
runIt($operation, "test2-sastt", "-G -f rttm -F rttm -a", "../test_suite/example.glm", "sastt", "english",
      "../test_suite/sastt-case2.ref.rttm", "../test_suite/sastt-case2.sys.rttm");
runIt($operation, "test1-notag", "", "../test_suite/example.glm", "hub5", "english",
      "../test_suite/lvc_refe.notag.noat.stm", 
      "../test_suite/lvc_hyp.notag.ctm ../test_suite/lvc_hyp2.notag.ctm");
runIt($operation, "test1-notag-a", "-a", "../test_suite/example.glm", "hub5", "english",
      "../test_suite/lvc_refe.notag.noat.stm", 
      "../test_suite/lvc_hyp.notag.ctm ../test_suite/lvc_hyp2.notag.ctm");
runIt($operation, "test1", "-V", "../test_suite/example.glm", "hub5", "english",
      "../test_suite/lvc_refe.stm", 
      "../test_suite/lvc_hyp.ctm ../test_suite/lvc_hyp2.ctm");
runIt($operation, "testArb", "-V -H -T -d", "../test_suite/test.arb2004.glm", "hub5", "arabic",
      "../test_suite/test.arb2004.txt.stm", 
      "../test_suite/test.arb2004.txt.ctm");

&ok_quit();

#	rm -rf testBase
#	mkdir -p testBase
#	cp ../test_suite/lvc_hyp.ctm ../test_suite/lvc_refe.stm testBase
#	cp ../test_suite/lvc_hyp.ctm testBase/lvc_hyp2.ctm
#	(cd testBase; ../hubscr.pl -p ../../csrfilt:../../def_art:../../acomp:../../hamzaNorm -l english -g ../../test_suite/example.glm -h hub5 -r lvc_refe.stm lvc_hyp.ctm lvc_hyp2.ctm)
