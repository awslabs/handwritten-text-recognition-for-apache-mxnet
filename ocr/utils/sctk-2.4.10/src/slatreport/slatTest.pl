#!/usr/bin/env perl
#
# $Id: slatTest.pl,v 1.3 2009/03/04 16:49:16 ajot Exp $

use warnings;
use strict;

system("./slatreport.pl -i ../test_suite/slat.rttm -o ../test_suite/slat.rttm.out.test -t LEXEME -s lex | grep -v 'PNG:' > ../test_suite/slat.rttm.out.test");

unlink("../test_suite/slat.rttm.out.test.SPLbDistribution.10.png");
unlink("../test_suite/slat.rttm.out.test.SPLmDistribution.10.png");
unlink("../test_suite/slat.rttm.out.test.SPLeDistribution.10.png");

my $diff = `diff ../test_suite/slat.rttm.out ../test_suite/slat.rttm.out.test`;

if($diff ne "")
{
	print "Slat Test Failed.\n";
	print "$diff\n";
	exit(1);
}
else
{
	print "Slat Test OK.\n";
	unlink("../test_suite/slat.rttm.out.test");
	exit(0);
}
