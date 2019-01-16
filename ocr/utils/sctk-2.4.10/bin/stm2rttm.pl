#!/usr/bin/perl -w

# Converts an STM file into a corresponding RTTM file
# Authors: Chris Laprun, Audrey Tong, Jon Fiscus
#
# v4:
#       - Added a check to make sure speakers do not have segments that overlap with themselves
# v2: 
# 	- Added constants for better legibility
#	- Added smoothing capability
#	- Now takes an evaluation code as parameter to support multiple evaluations

use Getopt::Long;

my $SUPPORTED = "Supported evaluations are:\n\t- Rich Transcription 02 (code: rt02)\n\t- Rich Transcription 05s (code: rt05s)\n\t- Rich Transcription 04 Spring (code: rt04s)\n";
my $ret    = GetOptions ("e|evaluation=s");
my $evaluation;
if (defined($opt_e)){
    $evaluation  = $opt_e;
} else {
  die "Usage: stm2rttm.pl -e [rt02|rt04s|rt05s]\nVersion: 0.1\n\nError: You must specify an evaluation code with the -e option!\n" . $SUPPORTED;
}
my %STM = (); # STM entries
my $FILE = 0; # index of the file id in the STM entry
my $CHANNEL = 1; # index of the channel number in the STM entry
my $SPEAKER = 2; # index of the speaker name in the STM entry
my $START_TIME = 3; # index of the utterance start time in the STM entry
my $END_TIME = 4; # index of the utterance end time in the STM entry
my $CATEGORIES = 5; # index of the categories block in the STM entry
my $TEXT = 6; # index of the utterance text in the STM entry

my $SEX_INDEX = -1; # index of the sex category in the categories block
my $SMOOTHING_TIME = -1.0; # number of seconds between utterances for smoothing

if ($evaluation eq "rt04s"){
	$SEX_INDEX = 1;
	$SMOOTHING_TIME = 0;
	$WITH_SEX = 1;
} elsif ($evaluation eq "rt02") {
	$SEX_INDEX = 2;
	$SMOOTHING_TIME = 0.5;
	$WITH_SEX = 1;
} elsif ($evaluation eq "rt05s") {
	$SMOOTHING_TIME = 0.3;
	$WITH_SEX = 0;
} else {
    die "Unknown target evaluation code!\n" . $SUPPORTED;
}

# Get the STM data from standard input
while (<>){
	next if ($_ =~ /^\s*$/);
	next if ($_ =~ /(^;;|inter_segment_gap|intersegment_gap)/i);
	my @d = split(/\s+/, $_, 7);
	if (! defined ($d[$FILE]) || ! defined ($d[$CHANNEL]) || ! defined ($d[$SPEAKER]) ) {
		die "No file, channel, or speaker defined";
	}
	
	push (@{ $STM{$d[$SPEAKER]}{$d[$FILE]}{$d[$CHANNEL]} }, [ @d ]);
	@info = split (",", $d[$CATEGORIES]);
	if ($WITH_SEX){
	    ($sex = $info[$SEX_INDEX]) =~ s/>//;
	
	    if ($sex eq "male") { $sex = "adult_male"; }
	    elsif ($sex eq "female") { $sex = "adult_female"; }
	    elsif ($sex =~ /unk/) { $sex = "unknown"; }
	    else { die "Unknown sex $sex"; }
	    
	    if (defined($spkrInfo{$d[$SPEAKER]})){
		die "Error: ambiguous spkr info $d[$SPEAKER]=>$sex but had $spkrInfo{$d[$SPEAKER]}" if ($spkrInfo{$d[$SPEAKER]} ne $sex && $d[$SPEAKER] !~ /excluded_region/i);
	    } else {
		$spkrInfo{$d[$SPEAKER]} = $sex;
	    }
	} else {
		$spkrInfo{$d[$SPEAKER]} = "unknown";
	}
    }



# Sort STM entries for smoothing
foreach $spkr (keys %STM) {
	for $file (keys %{ $STM{$spkr} }){
		for $chan (keys %{ $STM{$spkr}{$file} }){
			@ { $STM{$spkr}{$file}{$chan} } = sort numerically (@ { $STM{$spkr}{$file}{$chan} });
		}
	}
}

# Perform smoothing
foreach $spkr (keys %STM) {
	for $file (keys %{ $STM{$spkr} }){
		for $chan (keys %{ $STM{$spkr}{$file} }){
	    	$first = 1;
	    	for ($i=0; $i<@{ $STM{$spkr}{$file}{$chan} }; $i++){
				$seg = $STM{$spkr}{$file}{$chan}[$i];
				if ($first) {
				    $prev_seg = $seg;
			    	$first = 0;
				} else {
				    if ($seg->[$START_TIME] < $prev_seg->[$END_TIME]) {
					die "Error: segments from the same speaker overlap\n  ".
					    join(" ",@{$prev_seg})."\n  ".
					    join(" ",@{$seg});
				    }
				    if ($seg->[$START_TIME] - $prev_seg->[$END_TIME] <= $SMOOTHING_TIME) {
						$prev_seg->[$END_TIME] = $seg->[$END_TIME];
						$prev_seg->[$TEXT] = $prev_seg->[$TEXT] . " " . $seg->[$TEXT];
						splice (@{ $STM{$spkr}{$file}{$chan} }, $i, 1);
						$i++;
				    } else {
						$prev_seg = $seg;
			    	}
				}
	    	}
		}
	}
}

# Output speaker info metadata
foreach $spkr(keys %STM){
	for $file (keys %{ $STM{$spkr} }){
		for $chan (keys %{ $STM{$spkr}{$file} }){
	    	if ($spkr !~ /excluded_region/i) {
				print "SPKR-INFO $file $chan <NA> <NA> <NA> $spkrInfo{$spkr} $spkr <NA>\n";
	    	}
		}
	}
}

# Output speaker turns
foreach $spkr(keys %STM){
	for $file (keys %{ $STM{$spkr} }){
		for $chan (keys %{ $STM{$spkr}{$file} }){
	    	for $seg (@{ $STM{$spkr}{$file}{$chan} }){
				$beg = $seg->[$START_TIME];
				$end = $seg->[$END_TIME];
				$dur = sprintf("%.3f", $end - $beg);
				if ($spkr =~ /excluded_region/i) {
		    		print "NOSCORE $file $chan $beg $dur <NA> <NA> <NA> <NA>\n";
				} else {
				    print "SPEAKER $file $chan $beg $dur <NA> <NA> $spkr <NA>\n";
				}
			}
		}
	}
}

sub numerically {
	$a->[$START_TIME] <=> $b->[$START_TIME];
}
