#!/usr/bin/perl -w

# CTMVALIDATOR
# Author: Jerome Ajot
#
# This software was developed at the National Institute of Standards and Technology by 
# employees of the Federal Government in the course of their official duties. Pursuant
# to title 17 Section 105 of the United States Code this software is not subject to
# copyright protection and is in the public domain. CTMVALIDATOR is an experimental system.
# NIST assumes no responsibility whatsoever for its use by other parties, and makes no
# guarantees, expressed or implied, about its quality, reliability, or any other
# characteristic. We would appreciate acknowledgement if the software is used.
#
# THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO EXPRESS
# OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.

use strict;
use Getopt::Long;
use Data::Dumper;

my $VERSION = "3";

my $USAGE = "\n\n$0 [-l <language>] [-h] -i <CTM file>\n\n".
    "Version: $VERSION\n".
    "Description: This Perl program (version $VERSION) validates a given CTM file.\n".
    "\tNote that the program will exit after it finds all the syntax errors.\n".
    "Options:\n".
    "  -l <language> : check regarding the laguage (default: english)\n".
	"  -h            : print this help message\n".
    "Input:\n".
    "  -i <CTM file>: a CTM file\n\n";

# List the defined types
my %TYPES;
$TYPES{"lex"} = 1;
$TYPES{"frag"} = 1;
$TYPES{"fp"} = 1;
$TYPES{"un-lex"} = 1;
$TYPES{"for-lex"} = 1;
$TYPES{"non-lex"} = 1;
$TYPES{"misc"} = 1;
$TYPES{"noscore"} = 1;

# Number of fields
my $NUM_FIELDS = 8;
my $MIN_FIELDS = 5;

my $inputfile = "";
my $language = "english";

GetOptions
(
    'i=s' => \$inputfile,
    'l=s' => \$language,
    'h'   => sub { print $USAGE; exit },
);

if($inputfile eq "")
{
	print $USAGE;
	exit;
}

open(CTMFILE, $inputfile) or die "Unable to open for read ctm file '$inputfile'";

my $errors = 0;
my $line = 0;

while(<CTMFILE>)
{
	chomp;
	$line++;
	
	# Comments
	# you can have plenty of those
	next if($_ =~ /^;;/);
	
	my @ctm_record = split(/\s+/, $_);
#	print scalar(@ctm_record)." $_\n";
	if(scalar(@ctm_record) < $MIN_FIELDS){
		print "ERROR: [line $line] ctm record must have at least $MIN_FIELDS fields and no whitespace at the beginning\n";
		$errors++;
	} 
	elsif(scalar(@ctm_record) > $NUM_FIELDS)
	{
		print "ERROR: [line $line] ctm record must have no more than $NUM_FIELDS fields and no whitespace at the beginning\n";
		$errors++;
	}
	else
	{

		my $source   = $ctm_record[0];
		my $channel  = $ctm_record[1];
		my $beg_time = $ctm_record[2];
		my $duration = $ctm_record[3];
		my $token    = $ctm_record[4];
		my $conf     = (defined($ctm_record[5]) ? $ctm_record[5] : 0.0);
		my $type     = (defined($ctm_record[6]) ? $ctm_record[6] : "lex");
		my $speaker  = (defined($ctm_record[7]) ? $ctm_record[7] : "spkr");
		
		if($source !~ /^[A-Za-z0-9_-]+$/)
		{
			print "ERROR: [line $line] source '$source' must have alphanumeric, hyphens (-) and underscore (_) characters only\n";
			$errors++;
		}
		
		if($channel !~ /^(\d+|[AB])$/)
		{
			print "ERROR: [line $line] channel '$channel' must be numeric only\n";
			$errors++;
		}
		
		if ($token =~ /(<ALT>|<ALT_BEGIN>|<ALT_END>)/){
		    if($beg_time !~ /^\*$/)
		    {
			print "ERROR: [line $line] begin '$beg_time' time for ALT Tag must be an asterisk /*/\n";
			$errors++;
		    }
		    if($duration !~ /^\*$/)
		    {
			print "ERROR: [line $line] duration '$duration' time for ALT Tag must be an asterisk /*/\n";
			$errors++;
		    }
		} else {
		    if($beg_time !~ /^(\d+(\.\d+)?)$/)
		    {
			print "ERROR: [line $line] begin '$beg_time' time must floating point value\n";
			$errors++;
		    }
	
		    if($duration !~ /^(\d+(\.\d+)?)$/)
		    {
			print "ERROR: [line $line] duration '$duration' time must floating point value\n";
			$errors++;
		    }
		    if( (lc($language) eq "english") && ($type eq "lex") )
		    {
			if( ($token !~ /^[A-Za-z-\']+$/) && ($token !~ /^[A-Za-z]\./) && ($token !~ /@/) && ($token !~ /%(BCN?ACK|HESITATION)/i))
			{
				print "ERROR: [line $line] token '$token' must have alphabetic, hyphens (-) and apostrophes (') characters only\n";
				$errors++;
			}
		    }
		}

		if($conf !~ /^(\d+(\.\d+)?)$/)
		{
			print "ERROR: [line $line] confidence '$conf' time must floating point value\n";
			$errors++;
		}
		
		if(!defined($TYPES{$type}))
		{
			print "ERROR: [line $line] type '$type' is not a known type\n";
			$errors++;
		}
		
		if($speaker !~ /^[A-Za-z0-9_-]+$/)
		{
			print "ERROR: [line $line] speaker '$speaker' must have alphanumeric, hyphens (-) and underscore (_) characters only\n";
			$errors++;
		}
	}
}

close(CTMFILE);

if($errors > 0){
    print "FATAL: Validation '$inputfile' failed due to the $errors previous error(s)\n";
    exit 1;
} else {
    print "Validated $inputfile\n";
    exit 0;
}
