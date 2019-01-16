#!/usr/bin/perl -w

# History: 
#  v0 - initial version
#  v1 - Smooth IFF the gap is <= the threshold
#  v2 - echo out all lines
#     - IF $argv[0] is defined, it's the smooth time

use strict;
use Data::Dumper;
use Getopt::Long;
my $time = 0.3;
my $result = GetOptions ("s=s"   => \$time);
die "Error " if (! $result);

my %data = ();
while (<STDIN>){
    if ($_ =~ /^;/){
	print;
	next;
    }
    my @a = split;
    push @{ $data{$a[1]." ".$a[7]}{$a[0]} }, \@a;
}
#print Dumper(\%data);
#exit;
my $max;
foreach my $file(sort (keys %data)){
#    print "PRocess $file\n";
    my @d;
    foreach my $key(keys %{ $data{$file} }){
	next if ($key eq "SPEAKER");
	@d = @{ $data{$file}{$key} };
	for (my $i=0; $i<@d; $i++){
	    print join(" ",@{ $d[$i] })."\n";
	}
    }
    next if (! exists($data{$file}{"SPEAKER"} ));
    @d = sort {$a->[3] <=> $b->[3]} @{ $data{$file}{"SPEAKER"} };
    for (my $i=0; $i<@d - 1; $i++){
#	print "  ".join(" ",@{ @d[$i] })."\n";
#	die "Error: Segments from the same speaker overlap. \n   ".
#	    join(" $i: ", @{$d[$i]}). "\n   ".($i+1)." " join("  ".@{ $d[$i]})
#	    if ($d[$i]->[3] + $d[$i]->[4] > $d[$i+1]->[3]);
	if ($d[$i]->[3] + $d[$i]->[4] >= $d[$i+1]->[3] - $time){
#	    print "   Smooth with next ".($d[$i]->[3] + $d[$i]->[4])." with ",( $d[$i+1]->[3] - $time)."\n";
	    $max = $d[$i]->[3] + $d[$i]->[4];
	    $max = $d[$i+1]->[3] + $d[$i+1]->[4] if ($max < $d[$i+1]->[3] + $d[$i+1]->[4]);
	    $d[$i]->[4] = sprintf("%.3f",$max - $d[$i]->[3]);
	    splice (@d, $i+1, 1);
#	    print "      Replaced with ".join(" ",@{ @d[$i] })."\n";
	    $i--;
	}
    }
    for (my $i=0; $i<@d; $i++){
	print join(" ",@{ $d[$i] })."\n";
    }
}
