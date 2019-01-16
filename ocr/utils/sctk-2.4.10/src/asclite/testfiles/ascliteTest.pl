#!/usr/bin/perl -w 

# asclite
# Author: Jerome Ajot, Nicolas Radde, Chris Laprun
#
# This software was developed at the National Institute of Standards and Technology by
# employees of the Federal Government in the course of their official duties.  Pursuant to
# Title 17 Section 105 of the United States Code this software is not subject to copyright
# protection within the United States and is in the public domain. asclite is
# an experimental system.  NIST assumes no responsibility whatsoever for its use by any party.
#
# THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO EXPRESS
# OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.
 
use strict;

my $scliteCom = "../../sclite/sclite";
my $ascliteCom = "../core/asclite";
my $compatOutDir = "scliteCompatTestOutDir";
my $ascliteTestOutDir = "ascliteTestOutDir";
my $failure = 0;
use Getopt::Long;

####################
sub error_exit { exit(1); }
sub error_quit { print('[ERROR] ', join(' ', @_), "\n"); &error_exit(); }
sub ok_exit { exit(0); }
sub ok_quit { print(join(' ', @_), "\n"); &ok_exit(); }
####################

my $Usage = "Usage: ascliteTest.pl -s (all|sastt|std|mdm04|mdm04ByFile|cts04|mdmVariations|passed|notpassed) [ -m ]\n";
my $suite = "std";
my $bigMem = 0;
my $result = GetOptions("s=s" => \$suite, "m" => \$bigMem);
&error_quit("Aborting:\n$Usage\n") if (!$result);


if ($suite =~ /^(std|all|passed)$/)
{
### Nulls not fully implmented so the next test is commented out JGF
### RunCompatTest("CT-basic",             "-i spu_id", "-r basic.trn trn","-h basic.trn trn CT-basic");
    RunCompatTest("CT-stmctm-i-x-x",               "", "-r stmctm.stm stm", "-h stmctm.ctm ctm CT-stmctm-i-x-x");
    RunCompatTest("CT-rttmstm-i-x-x",              "", "-r rttmstm.stm stm","-h rttmstm.rttm rttm CT-rttmstm-i-x-x");
    RunCompatTest("CT-trn-i-x-x","         -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-i-x-x");
    RunCompatTest("CT-trn-i-F-x","    -F   -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-i-F-x");
    RunCompatTest("CT-trn-i-x-D","      -D -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-i-x-D");
    RunCompatTest("CT-trn-i-F-D","   -F -D -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-i-F-D");
    RunCompatTest("CT-trn-s-x-x","-s    -D -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-s-x-x");
    RunCompatTest("CT-trn-s-F-x","-s -F    -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-s-F-x");
    RunCompatTest("CT-trn-s-x-D","-s    -D -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-s-x-D");
    RunCompatTest("CT-trn-s-F-D","-s -F -D -i spu_id", "-r trn.ref trn",    "-h trn.hyp trn CT-trn-s-F-D");

    RunAscliteTest("CT-overlap.correct-i-x-x", "", "-r overlap.correct.stm stm", "-h overlap.correct.ctm ctm CT-overlap.correct-i-x-x");
    RunAscliteTest("CT-overlap.correct-i-x-x", "", "-r overlap.correct.stm stm", "-h overlap.correct.ctm ctm CT-overlap.correct-i-x-x -force-memory-compression");
    RunAscliteTest("CT-overlap-i-x-x",         "", "-r overlap.stm stm",         "-h overlap.ctm ctm CT-overlap-i-x-x");
    RunAscliteTest("CT-overlap-i-x-x",         "", "-r overlap.stm stm",         "-h overlap.ctm ctm CT-overlap-i-x-x -force-memory-compression");
}
    
if ($suite =~ /^(mdm04|all|passed)$/)
{
    RunAscliteTest("CT-mdm-full-i-F-D-ov1", "-noisg -F -D -overlap-limit 1",                           "-r rt04s.040420.mdm.overlap.stm.filt stm", "-h  rt04s.040810.mdm.overlap.ctm.filt ctm CT-mdm-full-i-F-D-ov1");
    RunAscliteTest("CT-mdm-full-i-F-D-ov1", "-noisg -F -D -overlap-limit 1 -force-memory-compression -memory-limit 0.0005", "-r rt04s.040420.mdm.overlap.stm.filt stm", "-h  rt04s.040810.mdm.overlap.ctm.filt ctm CT-mdm-full-i-F-D-ov1");
    RunAscliteTest("CT-mdm-full-i-F-D-ov2", "-noisg -F -D -overlap-limit 2",                           "-r rt04s.040420.mdm.overlap.stm.filt stm", "-h  rt04s.040810.mdm.overlap.ctm.filt ctm CT-mdm-full-i-F-D-ov2");
    RunAscliteTest("CT-mdm-full-i-F-D-ov2", "-noisg -F -D -overlap-limit 2 -force-memory-compression -memory-limit 0.0005", "-r rt04s.040420.mdm.overlap.stm.filt stm", "-h  rt04s.040810.mdm.overlap.ctm.filt ctm CT-mdm-full-i-F-D-ov2");
}

if ($suite =~ /^(uem|isg|all|passed)$/)
{
    RunAscliteTest("isg-0"     , "-F -D",                    "-r isg.ref.rttm rttm", "-h isg.sys.rttm rttm isg-0");
    RunAscliteTest("isg-1"     , "-F -D -uem isg1.uem ref" , "-r isg.ref.rttm rttm", "-h isg.sys.rttm rttm isg-1");
    RunAscliteTest("isg-2"     , "-F -D -uem isg2.uem ref" , "-r isg.ref.rttm rttm", "-h isg.sys.rttm rttm isg-2");
    RunAscliteTest("isg-3"     , "-F -D -uem isg3.uem ref" , "-r isg.ref.rttm rttm", "-h isg.sys.rttm rttm isg-3");
    RunAscliteTest("isg-4-ref" , "-F -D -uem isg4.uem ref" , "-r isg.ref.rttm rttm", "-h isg.sys.rttm rttm isg-4-ref");
    RunAscliteTest("isg-4-both", "-F -D -uem isg4.uem both", "-r isg.ref.rttm rttm", "-h isg.sys.rttm rttm isg-4-both");
}

if ($suite =~ /^(timebasecost|isg|all|passed)$/)
{
    RunAscliteTest("timebasecost" , "-F -D -time-base-cost", "-r isg.ref.rttm rttm", "-h isg.sys.rttm rttm timebasecost");
}

if ($suite =~ /^(sastt|all|passed)$/)
{
    RunAscliteTest("sastt-1", "-F -D -spkr-align sastt.map.csv", "-r sastt.ref.rttm rttm", "-h sastt.sys.rttm rttm sastt-1");
}

if ($suite =~ /^(generic|all|passed)$/)
{
    RunAscliteTest("generic-1", "-F -D -generic-cost -noisg", "", "-h generic-1.rttm rttm generic-1");
}

&error_quit("Errors Occured.  Exiting with non-zero code") if ($failure);
&ok_quit();

sub RunCompatTest
{
    my ($testId, $opts, $refOpts, $hypOpts) = @_;
    
    if (! -f "$compatOutDir/$testId.sgml")
    {
        print "Building Authoritative SGML file: $opts, $refOpts, $hypOpts\n";
        system "$scliteCom $opts $refOpts $hypOpts -o sgml stdout -f 0 | perl -pe 's/(creation_date=\")[^\"]+/\$1/' > $compatOutDir/$testId.sgml";
    }
	
    print "Comparing asclite to Authoritative SGML file: $opts, $refOpts, $hypOpts\n";
    my $com = "$ascliteCom $opts $refOpts $hypOpts -o sgml stdout -f 0";
    my $ret = system "$com |  perl -pe 's/(creation_date=\")[^\"]+/\$1/' > $compatOutDir/$testId.sgml.asclite";
        
    if ($ret != 0)
    {
        print "Error: Execution of '$com' returned code != 0\n";
    } 
    else
    { 
        my $diffCom = "diff $compatOutDir/$testId.sgml $compatOutDir/$testId.sgml.asclite";
        open (DIFF, "$diffCom |") || &error_quit("Diff command '$diffCom' Failed");
        my @diff = <DIFF>;
        close DIFF;
    
        if (@diff > 0)
        {
            print "Error: Test $testId has failed.  Diff output is :\n@diff\n" ;
            $failure = 1;
        } 
        else 
        {
            print "      Successful Test.  Removing $testId.sgml.asclite\n";
            system "rm -rf $compatOutDir/$testId.sgml.asclite";
        }	
    }
}

sub RunAscliteTest
{
    my ($testId, $opts, $refOpts, $hypOpts) = @_;
    
    if (! -f "$ascliteTestOutDir/$testId.sgml")
    {
        print "Building Authoritative SGML file: $opts, $refOpts, $hypOpts\n";
        system "$ascliteCom $opts $refOpts $hypOpts -o sgml stdout -f 0 | perl -pe 's/(creation_date=\")[^\"]+/\$1/' > $ascliteTestOutDir/$testId.sgml";
    } 
    else
    {
        print "Comparing asclite to Authoritative SGML file: $opts, $refOpts, $hypOpts\n";
        my $com = "$ascliteCom $opts $refOpts $hypOpts -o sgml stdout -f 0";
        my $ret = system "$com |  perl -pe 's/(creation_date=\")[^\"]+/\$1/' > $ascliteTestOutDir/$testId.sgml.asclite";
	
        if ($ret != 0)
        {
            print "Error: Execution of '$com' returned code != 0\n";
        }
        else 
        { 
            my $diffCom = "diff $ascliteTestOutDir/$testId.sgml $ascliteTestOutDir/$testId.sgml.asclite";
            open (DIFF, "$diffCom |") || &error_quit("Diff command '$diffCom' Failed");
            my @diff = <DIFF>;
            close DIFF;
            
            if (@diff > 0)
            {
                print "Error: Test $testId has failed.  Diff output is :\n@diff\n" ;
                $failure = 1;
            }
            else
            {
                print "      Successful Test.  Removing $testId.sgml.asclite\n";
                system "rm -rf $ascliteTestOutDir/$testId.sgml.asclite";
            }	
        }
    }
}

sub RunAscliteTestLog
{
    my ($testId, $opts, $refOpts, $hypOpts) = @_;
    
    if (! -f "$ascliteTestOutDir/$testId.log")
    {
        print "Building Authoritative SGML file: $opts, $refOpts, $hypOpts\n";
        system "$ascliteCom $opts $refOpts $hypOpts -f 6 2> $ascliteTestOutDir/$testId.log > /dev/null";
    } 
    else
    {
        print "Comparing asclite to Authoritative SGML file: $opts, $refOpts, $hypOpts\n";
        my $com = "$ascliteCom $opts $refOpts $hypOpts -f 6";
        my $ret = system "$com 2> $ascliteTestOutDir/$testId.log.asclite > /dev/null";
	
        if ($ret != 0)
        {
            print "Error: Execution of '$com' returned code != 0\n";
        }
        else 
        { 
            my $diffCom = "diff $ascliteTestOutDir/$testId.log $ascliteTestOutDir/$testId.log.asclite";
            open (DIFF, "$diffCom |") || &error_quit("Diff command '$diffCom' Failed");
            my @diff = <DIFF>;
            close DIFF;
            
            if (@diff > 0)
            {
                print "Error: Test $testId has failed.  Diff output is :\n@diff\n" ;
                $failure = 1;
            }
            else
            {
                print "      Successful Test.  Removing $testId.sgml.asclite\n";
                system "rm -rf $ascliteTestOutDir/$testId.log.asclite";
            }	
        }
    }
}
