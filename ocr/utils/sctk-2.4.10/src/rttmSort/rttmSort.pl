#!/usr/bin/perl -w

use strict;
use Data::Dumper;

my %sort_order = ("SPKRINFO"       => 0,
		  "NOSCORE"         =>  1,
		  "NORTMETADATA"  =>  2,
		  "SEGMENT"         =>  3,
		  "SPEAKER"         =>  4,
		  "SU"              =>  5,
		  "A/P"             =>  6,
		  "CB"              =>  7,
		  "IP"              =>  8,
		  "EDIT"            =>  9,
		  "FILLER"          =>  10,
		  "NON-SPEECH"      => 11,
		  "NON-LEX"         => 12,
		  "LEXEME"          => 13,
		  "SUboundary"      => 14);
my %spkrinfo;
my %stm_file;
 while (<>) {
     next if ($_ =~ /;;/);
     my $wrdExp = '[\]\[\w%\{'."\\'".'\<\>.-]+';
     $wrdExp = '[^\s]+';
     my $txtExp = "$wrdExp|\\($wrdExp\\)|<NA>";
     if (/(SPKR-INFO|SEGMENT|LEXEME|NON-LEX|CB|SU|EDIT|FILLER|IP|NOSCORE|SPEAKER|NORTMETADATA|NON-SPEECH|A\/P)\s+(\S+)\s+(\d+)\s+(\d*\.?\d+|<NA>)\s+(\d*\.?\d+|<NA>)\s+($txtExp)\s+([\w&-]+|<NA>)\s+(\S+|<NA>)\s+(\d*\.?\d+|<NA>)/) {
	 if ($1 eq "SPKR-INFO") {
#	     print "--$2  $8--\n";
	     die "Error: spkrinfo exists for '$2 $8'" if (exists($spkrinfo{$2." ".$8}));
	     $spkrinfo{$2." ".$8}{file} = $2;
	     $spkrinfo{$2." ".$8}{chan} = $3;
	     $spkrinfo{$2." ".$8}{gender} = $7;
	     $spkrinfo{$2." ".$8}{spkr} = $8;
	     $spkrinfo{$2." ".$8}{conf} = $9;
	     $spkrinfo{$2." ".$8}{line} = $. . " $_";
	 } else {
	     $stm_file{$2}{$3}{$4}{$.}{type} = $1;
	     $stm_file{$2}{$3}{$4}{$.}{beg_time} = $4;
	     $stm_file{$2}{$3}{$4}{$.}{end_time} = $5;
	     $stm_file{$2}{$3}{$4}{$.}{token} = $6;
	     $stm_file{$2}{$3}{$4}{$.}{subtype} = $7;
	     $stm_file{$2}{$3}{$4}{$.}{speaker} = $8;
	     $stm_file{$2}{$3}{$4}{$.}{conf} = $9;
	     $stm_file{$2}{$3}{$4}{$.}{line} = $. . " $_";
	 }
     } elsif (/^;;/) {
	 #nothing
     } else {
	 die "malformed line $.\n--> $_\n";
     }
 }

sub cmp_float {
    return 0 if ($a eq "<NA>" && $b eq "<NA>");
    return 1 if ($b eq "<NA>");
    return -1 if ($a eq "<NA>");
    return $a <=> $b;
}


foreach my $spkr (sort keys %spkrinfo) {
    print "SPKR-INFO $spkrinfo{$spkr}{file} $spkrinfo{$spkr}{chan} <NA> <NA> <NA> $spkrinfo{$spkr}{gender} $spkrinfo{$spkr}{spkr} $spkrinfo{$spkr}{conf}\n";
}

#print Dumper(\%stm_file);
foreach my $key_filename (sort keys %stm_file) {
    foreach my $key_channel (sort keys %{$stm_file{$key_filename}}) {
	foreach my $key_begtime (sort cmp_float keys %{$stm_file{$key_filename}{$key_channel}}) {
	    foreach my $line (sort {$sort_order{$stm_file{$key_filename}{$key_channel}{$key_begtime}{$a}{type}} <=>
					$sort_order{$stm_file{$key_filename}{$key_channel}{$key_begtime}{$b}{type}}} 
			      keys %{$stm_file{$key_filename}{$key_channel}{$key_begtime}}) {
		print "$stm_file{$key_filename}{$key_channel}{$key_begtime}{$line}{type} ";
		print "$key_filename $key_channel ";
		print "$stm_file{$key_filename}{$key_channel}{$key_begtime}{$line}{beg_time} ";
		print "$stm_file{$key_filename}{$key_channel}{$key_begtime}{$line}{end_time} ";
		print "$stm_file{$key_filename}{$key_channel}{$key_begtime}{$line}{token} ";
		print "$stm_file{$key_filename}{$key_channel}{$key_begtime}{$line}{subtype} ";
		print "$stm_file{$key_filename}{$key_channel}{$key_begtime}{$line}{speaker} ";
		print "$stm_file{$key_filename}{$key_channel}{$key_begtime}{$line}{conf}\n";
	    }
	}
    }
}

