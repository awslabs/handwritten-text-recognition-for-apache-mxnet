#!/usr/bin/perl -w

use strict;
use Data::Dumper;

my $time = 0.3;
my %data = ();
while (<>){
    my @a = split;
    if ($_ =~ /^;/){
	print;
	next;
    }
    push @{ $data{$a[1]}{$a[0]} }, \@a;
}
#print Dumper(\%data);
my $max;
foreach my $file(sort (keys %data)){
    if (exists($data{$file}{"SPKR-INFO"})){
	my @d = @{ $data{$file}{"SPKR-INFO"} };
	$d[0]->[7] = "speech";
	$d[0]->[6] = "unknown";
	print join(" ",@{ $d[0] })."\n";
    } else {
	die "Error: No SPKR-INFO and no SPEAKER Tags" if (! exists($data{$file}{"SPEAKER"}));
	# Copy a SPEAKER line
	my @a = ();
	foreach my $e( @{ $data{$file}{"SPEAKER"}[0] }){
	    push @a, $e;
	}
	$a[0] = "SPKR-INFO";
	$a[3] = "<NA>";
	$a[4] = "<NA>";
	$a[5] = "<NA>";
	$a[6] = "unknown";
	$a[7] = "speech";
	print join(" ",@a)."\n";
    }
    if (exists($data{$file}{"SPEAKER"})){
	my @d = sort {$a->[3] <=> $b->[3]} @{ $data{$file}{"SPEAKER"} };
	for (my $i=0; $i<@d ; $i++){
	    $d[$i]->[7] = "speech";
	    print join(" ",@{ $d[$i] })."\n";
	}
    }
}
