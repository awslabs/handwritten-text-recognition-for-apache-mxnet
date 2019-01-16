#!/usr/bin/env perl

use strict;

my $Version="1.0";

#####
#  Version 1.0  Released August 25, 2004
#        - Initial Release

my $Usage="Usage: hamzaNorm.pl [ -i fmt ] Infile|- OutFile|-\n".
"Version: $Version\n".
"Desc: hamzaNorm normalizes the hamza in the initial position of the word to be a\n".
"      consistent form.  In the Buckwalter normalization scheme, the initial letters\n".
"      'A', '<', '>' and '|' are all translated to 'A'".
"Options:\n".
"      -i fmt   Set the input file formant to 'fmt'.  The possible choices are:\n".
"                  txt -> plain text, the default\n".
"                  ctm -> CTM format, ignores all but the 5th column, and if\n".
"                         a division occurs and a confidence score is present,\n".
"                         the confidence score is copied to all parts.\n".
"                  stm -> STM format, change only the text field of the stm record\n".
"\n";

use Getopt::Long;
my ($InFmt) = undef;
my $result = GetOptions ("i:s" => \$InFmt);
die "Aborting:\n$Usage\n:" if (!$result);

if (defined($InFmt)) {
    die("$Usage\n\nError: Undefined input format '$InFmt'") 
	if ($InFmt !~ /^(txt|ctm|stm)$/);
} else {
    $InFmt = "txt";
} 

#### The main functions arguements:
die "$Usage\nToo many arguements" if ($#ARGV > 1);
die "$Usage\nOutput Not Specified" if ($#ARGV == 0);
die "$Usage\nInput and Output Not Specified" if ($#ARGV == -1);

my $InFile=$ARGV[0];
my $OutFile=$ARGV[1];
die("$Usage\nError: Input file $InFile does not exist\n")
    if ($InFile ne "-" && ! -r $InFile);

open(IN, "$InFile") || die "Unable to open trans-file $InFile";
open(OUT, ">$OutFile") || die "Unable to open new-trans-file $OutFile";

while (<IN>){
    chomp;
    if ($InFmt eq "txt"){
	print OUT normalize($_)."\n";
    } elsif ($InFmt eq "ctm"){
	if ($_ =~ /^(\;\;|\#)/){
	    print OUT $_."\n";
	    next;
	}	     
	s/^(\s+)//;
	my $prefix = (defined($1) ? $1 : "");
	my @ctm = split(/(\s+)/,$_);
	$ctm[8] = normalize($ctm[8]);
	print OUT $prefix.join("", @ctm)."\n";
    } elsif ($InFmt eq "stm"){
	if ($_ =~ /^(\;\;|\#)/){
	    print OUT $_."\n";
	    next;
	}	     
	s/^(\s+)//;
	my $prefix = (defined($1) ? $1 : "");
	my @stm = split(/(\s+)/,$_, 7);
	if ($stm[10] =~ /^<[^<>]*>$/){
	    $stm[12] = normalize($stm[12]);
	} else {
	    $stm[10] .= join("",splice(@stm,11,2));
	    $stm[10] = normalize($stm[10]);
	}
	print OUT $prefix.join("", @stm)."\n";
    } else {
	die "Error: unknown input format '$InFmt'\n$Usage\n";
    }    
}

close IN; close OUT;
exit 0;

sub normalize{
    my ($text) = @_;
    $text = " ".$text;    
    $text =~ s/ (\330\242|\330\243|\330\245|\330\247)/ \330\247/g;
    $text =~ s/^ //;
    $text;
}


