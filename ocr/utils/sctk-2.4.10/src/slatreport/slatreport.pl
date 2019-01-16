#!/usr/bin/env perl
#
# $Id: slatreport.pl,v 1.13 2009/03/03 19:55:25 ajot Exp $

use warnings;
use strict;
use Getopt::Long;
use Data::Dumper;
use Pod::Usage;

my $GNUPLOT = `which gnuplot`;
$GNUPLOT = "" if($GNUPLOT !~ /^\//);

my $version = "$1" if('$Revision: 1.13 $' =~ /(\d+\.\d+)/);

my $man = 0;
my $help = 0;
my $rttmfile = "";
my $outputfile = "";
my $typeslist = "";
my $stypeslist = "";
my $filelist = "";
my $chnllist = "";
my $namelist = "";
my $histogrampartitions = 10;

Getopt::Long::Configure("no_ignore_case");

GetOptions
(
	'help|h'        => \$help,
	'man|m'         => \$man,
	'version'       => sub { my $name = $0; $name =~ s/.*\/(.+)/$1/; print "$name version $version\n"; exit(0); },
	'i=s'           => \$rttmfile,
	'o=s'           => \$outputfile,
	'type|t=s'      => \$typeslist,
	'stype|s=s'     => \$stypeslist,
	'file|f=s'      => \$filelist,
	'chnl|c=s'      => \$chnllist,
	'name|n=s'      => \$namelist,
	'Histogram|H=i' => \$histogrampartitions,
) or pod2usage(1);

# Docs
pod2usage(1) if $help;
pod2usage(-exitvalue => 0, -verbose => 2) if $man;

# Checking the inputs
pod2usage("Error: RTTM file must be specified.\n") if ($rttmfile eq "");
pod2usage("Error: Output filename must be specified.\n") if($outputfile eq "");
pod2usage("Error: Histogram partitions 'H' must be bigger than 1.\n") if($histogrampartitions < 2);

my @Files = split(/,/, $filelist);
my @Chnl = split(/,/, $chnllist);
my @Names = split(/,/, $namelist);
my @Types = split(/,/, $typeslist);
my @STypes = split(/,/, $stypeslist);

print "Conditions:\n";

print "  FILE: " .  ($filelist eq "" ? "ALL" : $filelist) . "\n";
print "  CHANNEL: " .  ($chnllist eq "" ? "ALL" : $chnllist) . "\n";
print "  NAME: " .  ($namelist eq "" ? "ALL" : $namelist) . "\n";
print "  TYPE: " .  ($typeslist eq "" ? "ALL" : $typeslist) . "\n";
print "  SUBTYPE: " .  ($stypeslist eq "" ? "ALL" : $stypeslist) . "\n";

print "\n";

my $data = LoadRTTM($rttmfile);
my ($LatencyBegMean, $LatencyBegStddev, $LatencyBegDistribution, $LatencyMidMean, $LatencyMidStddev, $LatencyMidDistribution, $LatencyEndMean, $LatencyEndStddev, $LatencyEndDistribution) = Compute($data, $histogrampartitions, \@Files, \@Chnl, \@Names, \@Types, \@STypes);

print "Sample Processing Latency Begin Time Based (SPLb):\n";
print "  Mean: " . sprintf("%.3f", $LatencyBegMean) . "\n";
print "  Standard Deviation: " . sprintf("%.3f", $LatencyBegStddev) . "\n" if(defined($LatencyBegStddev));

print "  Distribution:\n";

if(defined($LatencyBegDistribution))
{
	for(sort {$a <=> $b} keys %$LatencyBegDistribution)
	{
		my $beginrange = sprintf("%6.3f", $LatencyBegDistribution->{$_}{RANGE_MIN});
		my $endrange = sprintf("%6.3f", $LatencyBegDistribution->{$_}{RANGE_MAX});
		my $count = $LatencyBegDistribution->{$_}{COUNT};
		
		print "    $beginrange - $endrange: $count\n";
	}
	
	BuildPNG($LatencyBegDistribution, "$outputfile.SPLbDistribution.$histogrampartitions", "Sample Processing Latency Begin Time Based Distribution - $histogrampartitions partitions") if($outputfile ne "/dev/null");
}
else
{
	print "No distribution with one data point.\n";
}

print "\n";
print "Sample Processing Latency Mid-point Time Based (SPLm):\n";
print "  Mean: " . sprintf("%.3f", $LatencyMidMean) . "\n";
print "  Standard Deviation: " . sprintf("%.3f", $LatencyMidStddev) . "\n" if(defined($LatencyMidStddev));
print "  Distribution:\n";

if(defined($LatencyMidDistribution))
{
	for(sort {$a <=> $b} keys %$LatencyMidDistribution)
	{
		my $beginrange = sprintf("%6.3f", $LatencyMidDistribution->{$_}{RANGE_MIN});
		my $endrange = sprintf("%6.3f", $LatencyMidDistribution->{$_}{RANGE_MAX});
		my $count = $LatencyMidDistribution->{$_}{COUNT};
		
		print "    $beginrange - $endrange: $count\n";
	}
	
	BuildPNG($LatencyMidDistribution, "$outputfile.SPLmDistribution.$histogrampartitions", "Sample Processing Latency Mid-point Time Based Distribution - $histogrampartitions partitions") if($outputfile ne "/dev/null");
}
else
{
	print "No distribution with one data point.\n";
}

print "\n";
print "Sample Processing Latency End Time Based (SPLe):\n";
print "  Mean: " . sprintf("%.3f", $LatencyEndMean) . "\n";
print "  Standard Deviation: " . sprintf("%.3f", $LatencyEndStddev) . "\n" if(defined($LatencyEndStddev));
print "  Distribution:\n";

if(defined($LatencyEndDistribution))
{
	for(sort {$a <=> $b} keys %$LatencyEndDistribution)
	{
		my $beginrange = sprintf("%6.3f", $LatencyEndDistribution->{$_}{RANGE_MIN});
		my $endrange = sprintf("%6.3f", $LatencyEndDistribution->{$_}{RANGE_MAX});
		my $count = $LatencyEndDistribution->{$_}{COUNT};
		
		print "    $beginrange - $endrange: $count\n";
	}
	
	BuildPNG($LatencyEndDistribution, "$outputfile.SPLeDistribution.$histogrampartitions", "Sample Processing Latency End Time Based Distribution - $histogrampartitions partitions") if($outputfile ne "/dev/null");
}
else
{
	print "No distribution with one data point.\n";
}

################
### Funtions ###
################
sub BuildPNG
{
	my ($data, $filename, $title) = @_;
	
	if($GNUPLOT eq "")
	{
		print "PNG: 'gnuplot not found' - no PNG generated. Install 'gnuplot' to have the PNGs.\n";
		return;
	}
	
	# Create the dat file
	my $rectstr = "";
	
	my $maxx = -1;
	my $maxy = -1;
	
	for(sort {$a <=> $b} keys %$data)
	{
		my $xbl = $data->{$_}{RANGE_MIN};
		my $ybl = 0;
		my $xtr = $data->{$_}{RANGE_MAX};
		my $ytr = $data->{$_}{COUNT};
		
		$maxx = $data->{$_}{RANGE_MAX} if($data->{$_}{RANGE_MAX} > $maxx);
		$maxy = $data->{$_}{COUNT} if($data->{$_}{COUNT} > $maxy);
		
		$rectstr .= "set object rect from $xbl,$ybl to $xtr,$ytr fc lt 1\n";
	}
	
	open(FILE, ">", "$filename.plt")
		or die "Cannot open file '$filename.plt'";
	
	print FILE "set terminal png medium\n";
	print FILE "set nokey\n";
	print FILE "set title \"$title\"\n";
	print FILE "set xrange [0:$maxx]\n";
	print FILE "set yrange [0:$maxy]\n";
	print FILE $rectstr;	
	print FILE "plot -1\n";

	close(FILE);
	
	system("cat $filename.plt | gnuplot > $filename.png");
	
	unlink("$filename.plt");
	
	print "PNG: $filename.png\n";
}

sub Compute
{
	my ($data, $HistPar, $Fil, $Chn, $Nam, $Typ, $STy) = @_;

	my @statLacencyBeg = ();
	my @statLacencyMid = ();
	my @statLacencyEnd = ();
	
	foreach my $t (keys %{ $data })
	{
		next if(! IsElement($t, $Typ));
		
		foreach my $f (keys %{ $data->{$t} })
		{
			next if(! IsElement($f, $Fil));
			
			foreach my $c (keys %{ $data->{$t}{$f} })
			{
				next if(! IsElement($c, $Chn));
				
				foreach my $s (keys %{ $data->{$t}{$f}{$c} })
				{
					next if(! IsElement($s, $STy));
					
					foreach my $n (keys %{ $data->{$t}{$f}{$c}{$s} })
					{
						next if(! IsElement($n, $Nam));
						
						foreach my $TBEG (keys %{ $data->{$t}{$f}{$c}{$s}{$n} })
						{
							my $TMID = $data->{$t}{$f}{$c}{$s}{$n}{$TBEG}{TMID};
							my $TEND = $data->{$t}{$f}{$c}{$s}{$n}{$TBEG}{TEND};
							my $TSLAT = $data->{$t}{$f}{$c}{$s}{$n}{$TBEG}{TSLAT};
							
							push(@statLacencyBeg, $TSLAT-$TBEG);
							push(@statLacencyMid, $TSLAT-$TMID);
							push(@statLacencyEnd, $TSLAT-$TEND);
						}
					}
				}
			}
		}
	}

	if(scalar(@statLacencyBeg) == 0)
	{
		print "No data found in files regarding criteria for Latency.\n";
		exit;
	}
	
	if(scalar(@statLacencyBeg) == 1)
	{
		return( mean(\@statLacencyBeg), undef, undef,
	            mean(\@statLacencyMid), undef, undef,
	            mean(\@statLacencyEnd), undef, undef );
	}
	
	my $lbd = frequency_distribution(\@statLacencyBeg, $HistPar);
	my $lmd = frequency_distribution(\@statLacencyMid, $HistPar);
	my $led = frequency_distribution(\@statLacencyEnd, $HistPar);
	
	return( mean(\@statLacencyBeg), stddev(\@statLacencyBeg), $lbd,
			mean(\@statLacencyMid), stddev(\@statLacencyMid), $lmd,
			mean(\@statLacencyEnd), stddev(\@statLacencyEnd), $led );
}

sub IsElement
{
	my ($e, $l) = @_;
	
	return 1
		if(scalar(@$l) == 0);
	
	for(my $i=0; $i<scalar(@$l); $i++)
	{
		return 1
			if($l->[$i] eq $e);
	}
	
	return 0;
}

sub LoadRTTM
{
	my ($file) = @_;
	
	open(FILE, "<", $file)
		or die "Cannot open file '$file'";
	
	my %h;
	
	while ( <FILE> )
	{
		chomp;
		next if($_ =~ /^\;;/);
		next if($_ =~ /^\s*$/);
		my @a = split(/\s+/, $_);
		
		die "This file '$file' is not properly formated: SLAT information missing"
			if(scalar(@a) != 10);
		
		next if($a[9] =~ /<NA>/);
		
		my $tdur = 0;
		$tdur = $a[4] if($a[4] !~ /<NA>/);
		
		$h{$a[0]}{$a[1]}{$a[2]}{$a[6]}{$a[7]}{$a[3]}{TMID} = $a[3]+($tdur/2);
		$h{$a[0]}{$a[1]}{$a[2]}{$a[6]}{$a[7]}{$a[3]}{TEND} = $a[3]+$tdur;
		$h{$a[0]}{$a[1]}{$a[2]}{$a[6]}{$a[7]}{$a[3]}{TSLAT} = $a[9];
	}
		
	close(FILE);
	
	return \%h;
}

sub mean
{
	my ($list) = @_;
	
	my $sum = 0;
	
	for(my $i=0; $i<scalar(@$list); $i++)
	{
		$sum += $list->[$i];
	}
	
	return($sum/scalar(@$list));
}

sub stddev
{
	my ($list) = @_;
	
	my $mean = mean($list);
	my $stddev = 0;
	
	for(my $i=0; $i<scalar(@$list); $i++)
	{
		$stddev += ($list->[$i] - $mean)*($list->[$i] - $mean);
	}
	
	return(sqrt($stddev/scalar(@$list)));
}

sub frequency_distribution
{
	my ($list, $partitions) = @_;
	
	my %hash;
	
	my @sorted_list = sort {$a <=> $b} @$list;
	
	my $min = $sorted_list[0];
	my $max = $sorted_list[scalar(@sorted_list)-1];
	
	my $range_iter = ($max-$min)/$partitions;
	
	for(my $i=0; $i<$partitions; $i++)
	{
		my $min_range = $min+$i*$range_iter;
		my $max_range = $min_range+$range_iter;
		my $key_range = ($min_range+$max_range)/2;
		
		my $range_max_use = $max_range;
		
		if($i == $partitions-1)
		{
			#last one, make sure we have the max elements
			$range_max_use += $range_iter
		}
		
		for (my $j=0; $j<scalar(@sorted_list); $j++)
		{
			if( ($sorted_list[$j] >= $min_range) && ($sorted_list[$j] < $range_max_use) )
			{
				$hash{$i}{RANGE_MIN} = $min_range;
				$hash{$i}{RANGE_MAX} = $max_range;
				$hash{$i}{COUNT}++;
			}
		}
	}
	
	return(\%hash);
}

__END__

=head1 NAME

slatreport.pl -- Create reports for SLAT information in RTTM files.

=head1 SYNOPSIS

B<slatreport.pl> B<-i> F<FILE> B<-o> F<FILE> [B<-H> F<NUMBER>] [OPTIONS]

=head1 DESCRIPTION

The script analyses and generates a report and bar charts based on the SLAT information from the RTTM file (RT09 specifications). It generates Sample Processing Latency based on the begin, mid-point and end time of all RTTM Objects with SLAT information.

=head1 OPTIONS

=over 25

=item B<-i> F<FILE>

RTTM input file.

=item B<-o> F<FILE>

Base filename for the PNG bar charts. If '/dev/null' is passed then it will not generate the charts.

=item B<-H> F<NUMBER>

Generates reports and bar charts with F<NUMBER> partitions (default: 10).

=item B<-f> F<FILENAME>[,F<FILENAME>[,...]]

Analyses the data only the F<FILENAME>s.

=item B<-c> F<CHANNEL>[,F<CHANNEL>[,...]]

Analyses the data only the F<CHANNEL>s.

=item  B<-n> F<NAME>[,F<NAME>[,...]]

Analyses the data for only the F<NAME> objects.  Default is all objects.

=item B<-t> F<TYPE>[,F<TYPE>[,...]]

Analyses the data for only the objects of type F<TYPE>s.  The default is all types. 

=item B<-s> F<SUBTYPE>[,F<SUBTYPE>[,...]]

Analyses the data for only the objects of type F<SUBTYPE>s.  The default is all subtypes. 

=item B<-h>, B<--help>

Print the help.

=item B<-m>, B<--man>

Print the manual.

=item B<--version>

Print the version number.

=back

=head1 BUGS

No known bugs.

=head1 NOTES

A report that shows the Sample Processing Latency for LEXEMEs and lex (as the sub-type):

$ slatreport.pl -i file.rttm -o slatreport -t LEXEME -s lex

If only one element is in the data pool, the standard deviation and distributions are not calculated.

See RTTM and SLAT specifications on the Rich Transcription 2009 Evaluation website (http://nist.gov/speech/tests/rt/2009/index.html).

Sample Processing Latency begin time based (SPLb) is calculated from the begin time to the SLAT.

Sample Processing Latency mid-point time based (SPLm) is calculated from the  mid-point time to the SLAT.

Sample Processing Latency end time based (SPLe) is calculated from the end time to the SLAT.

=head1 AUTHOR

Jerome Ajot <jerome.ajot@nist.gov>

=head1 VERSION

slatreport.pl $Revision: 1.13 $

=head1 COPYRIGHT 

This software was developed at the National Institute of Standards and Technology by employees of the Federal Government in the course of their official duties.  Pursuant to Title 17 Section 105 of the United States Code this software is not subject to copyright protection within the United States and is in the public domain. It is an experimental system.  NIST assumes no responsibility whatsoever for its use by any party.

THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO EXPRESS OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
