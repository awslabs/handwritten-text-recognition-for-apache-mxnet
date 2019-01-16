#!/usr/bin/env perl

# RTTMVALIDATOR
# Author: Jon Fiscus, Jerome Ajot
#
# This software was developed at the National Institute of Standards and Technology by 
# employees of the Federal Government in the course of their official duties. Pursuant
# to title 17 Section 105 of the United States Code this software is not subject to
# copyright protection and is in the public domain. RTTMVALIDATOR is an experimental system.
# NIST assumes no responsibility whatsoever for its use by other parties, and makes no
# guarantees, expressed or implied, about its quality, reliability, or any other
# characteristic. We would appreciate acknowledgement if the software is used.
#
# THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO EXPRESS
# OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.

use warnings;
use strict;
use Getopt::Std;
use Data::Dumper;

my $debug = 0;

my $VERSION = "v13";

my $USAGE = "\n\n$0 [-useh] -i <RTTM file>\n\n".
    "Version: $VERSION\n".
    "Description: This Perl program (version $VERSION) validates a given RTTM file.\n".
    "\tNote that the program will exit after it finds all the syntax errors or\n".
    "\twhen it finds the first logic error.\n".
    "Options:\n".
    "  -u            : disable check that ensures all LEXEMEs belong to some SU object\n".
    "  -s            : disable check that ensures all LEXEMEs belong to some SPEAKER object\n".
    "  -p            : disable check that ensures all speakers defined in SPKR-INFO object\n".
    "  -e            : disable check that ensure there is an IP for each EDIT and each FILLER object\n".
    "  -f            : disable check that ensures the file columc matches the filename\n".
    "  -S            : allow SCTK enhancements to allow multiple itme notations in LEXEME tags\n".
	"  -h            : print this help message\n".
    "Input:\n".
    "  -i <RTTM file>: an RTTM file\n\n";

#my $NUM_FIELDS = 9; # count from 1
#number of fields can be 9 or 10 (with the SLAT)

my $ROUNDING_THRESHOLD = 0.000999999;

my $checkFileName = 1;   ### Check the filename against the file column

my %SORT_ORDER = ("NOSCORE"         =>  0,
		  "NO_RT_METADATA"  =>  1,
		  "SEGMENT"         =>  2,
		  "SPEAKER"         =>  3,
		  "SU"              =>  4,
		  "A/P"             =>  5,
		  "CB"              =>  6,
		  "IP"              =>  7,
		  "EDIT"            =>  8,
		  "FILLER"          =>  9,
		  "NON-SPEECH"      => 10,
		  "NON-LEX"         => 11,
		  "LEXEME"          => 12,
		  "SPKR-INFO"       => 13);

########################################################
# Main
#
########################################################
{
    my ($date, $time) = date_time_stamp();
    my $commandline = join(" ", @ARGV);

    use vars qw ($opt_i $opt_u $opt_s $opt_e $opt_h $opt_f $opt_t $opt_p $opt_S);
    getopts('i:usevhftpS');
    die ("$USAGE") if ($opt_h) || (! $opt_i);

    my @mde_types = ();
    push (@mde_types, "SU") if ! $opt_u;
    push (@mde_types, "SPEAKER") if ! $opt_s;
    push (@mde_types, "EDIT") if ! $opt_e;
    $checkFileName = ! $opt_f if $opt_f;
    my $allowSCTKExtensions = 1 if $opt_S;

	if (! $opt_t)
    {
    	print "$0 (version $VERSION) run on $date at $time\n";
   		print "command line:  ", $0, " (version ", $VERSION, ") ", $commandline, "\n";
	}
	
    my (%rttm_data, $data_domain);
    get_rttm_data(\%rttm_data, \$data_domain, $opt_i, $allowSCTKExtensions);

    # debug
    #
    my $count;
    if ($debug) {
#	print Dumper(%rttm_data);
	foreach my $src (keys %rttm_data) {
	    foreach my $chnl (keys %{$rttm_data{$src}}) {
		foreach my $spkr (keys %{$rttm_data{$src}{$chnl}}) {
		    foreach my $type (keys %{$rttm_data{$src}{$chnl}{$spkr}}) {
			$count = 0;
			foreach my $obj (@{$rttm_data{$src}{$chnl}{$spkr}{$type}}) {
			    print "[$src $chnl $spkr $type $count]\t$obj->{TYPE}\t$obj->{SRC}\t$obj->{CHNL}\t$obj->{TBEG}\t$obj->{TDUR}\t$obj->{ORTHO}\t$obj->{STYPE}\t$obj->{SPKR}\t$obj->{CONF}\n";
			    $count++;
			}
		    }
		}
	    }
	}
    }

    if (check_syntax_errors(\%rttm_data, $data_domain, $opt_i, $allowSCTKExtensions) &&
	check_logic_errors(\%rttm_data, \@mde_types)){
	exit 0;
    }
    exit 1;
}

########################################################
# Subroutines
#
########################################################
sub get_rttm_data {

    # read the RTTM file
    #
    my ($data, $domain, $file) = @_;

    my (@fields, $obj);

    my $line_num = 0;

    ${$domain} = "";

    open DATA, "$file" or die ("ERROR: Unable to open file $file\n");
    while (<DATA>) {
	$line_num++;
	chomp;
	if (/^\s*;;\s*EXP-ID:\s*(.*)/) {
	    my $exp_id = $1;
	    if ($exp_id =~ /bnews/i) {
		${$domain} = "bn";
	    } elsif ($exp_id =~ /cts/i) {
		${$domain} = "cts";
	    }
	} elsif (! /^\s*;;/ && ! /^\s*$/) {
	    @fields = split;
	    undef $obj;
	    $obj->{FIELD_COUNT} = scalar (@fields);
            $obj->{LOC} = "line $line_num";
	    $obj->{TYPE} = shift @fields;
	    $obj->{SRC} = shift @fields;
	    $obj->{CHNL} = shift @fields;
	    $obj->{TBEG} = shift @fields;
	    $obj->{TDUR} = shift @fields;
	    $obj->{ORTHO} = shift @fields;
	    $obj->{STYPE} = shift @fields;
	    $obj->{SPKR} = shift @fields;
	    $obj->{CONF} = shift @fields;
	    $obj->{SLAT} = shift @fields;
	    push (@{$data->{$obj->{SRC}}{$obj->{CHNL}}{$obj->{SPKR}}{$obj->{TYPE}}}, $obj);
	}
    }
    close DATA;
}

sub check_syntax_errors {
    # check for syntax errors such as missing fields,
    # invalid field values, etc.
    # 
    my ($data, $domain, $file, $allowSCTKExtensions) = @_;

    my $pass = 1;

    $file =~ s/\.[^\.]*$//;
    $file =~ s/.*\///;

    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach my $type (keys %{$data->{$src}{$chnl}{$spkr}}) {
		    foreach my $obj (@{$data->{$src}{$chnl}{$spkr}{$type}}) {
			# make sure that we have the correct number of fields
			#
#			if ($obj->{FIELD_COUNT} != $NUM_FIELDS) {
			if( ($obj->{FIELD_COUNT} != 9) && ($obj->{FIELD_COUNT} != 10) ){
			    print "ERROR: This record has $obj->{FIELD_COUNT} fields instead of the required 9 or 10 fields; see $obj->{LOC}\n";
			    $pass = 0;
			}
			
			# for the following checks to work, all the fields must be defined
			#
			if (defined ($obj->{TYPE}) && defined ($obj->{SRC}) &&
			    defined ($obj->{CHNL}) && defined ($obj->{TBEG}) &&
			    defined ($obj->{TDUR}) && defined ($obj->{ORTHO}) &&
			    defined ($obj->{STYPE}) && defined ($obj->{SPKR}) &&
			    defined ($obj->{CONF})) {

			    # make sure the type field is a valid RTTM type
			    #
			    if ($obj->{TYPE} !~ /^(SEGMENT|NOSCORE|NO_RT_METADATA|LEXEME|NON-LEX|NON-SPEECH|FILLER|EDIT|SU|IP|CB|A\/P|SPEAKER|SPKR-INFO)$/i) {
				print "ERROR: Invalid RTTM type '$obj->{TYPE}'; see field (0) in $obj->{LOC}\n";
				$pass = 0;
			    }
			    
			    # make sure the source field matches the base filename.
			    # This won't work if people concat all the files into one.
			    # Make it a warning instead
			    #
			    if ($checkFileName && $obj->{SRC} !~ /^$file$/i) {
				print "WARNING: Source field '$obj->{SRC}' doesn't match the input file's base filename '$file'; see field (2) in $obj->{LOC}\n";
			    }
			    
			    # make sure the channel ID has a value of 1 or 2 but 1 for BN data
			    #
			    if ($obj->{CHNL} !~ /^(1|2)$/) {
				print "ERROR: Invalid channel ID; see field (3) in $obj->{LOC}\n";
				$pass = 0;
			    } elsif ($domain =~ /bn/i && $obj->{CHNL} != 1) {
				print "ERROR: Expected channel ID to be 1 for BN data; see field (3) in $obj->{LOC}\n";
				$pass = 0;
			    }
			    
			    # make sure that if it's a SPKR-INFO the start time is <NA>
			    # else the start time must be a number and is non-negative
			    #
			    if ($obj->{TYPE} =~ /SPKR-INFO/i) {
				if ($obj->{TBEG} !~ /<NA>/i) {
				    print "ERROR: $obj->{TYPE} should not have any start time; see field (4) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } else {
				if ($obj->{TBEG} !~ /^(\-{0,1}\d+\.?\d*|\-{0,1}\.\d+)\**$/) {
				    print "ERROR: Expected start time to be a number; see field (4) in $obj->{LOC}\n";
				    $pass = 0;
				} else {
				    my $tbeg = $obj->{TBEG};
				    $tbeg =~ s/\*//;
				    if ($tbeg < 0) {
					print "ERROR: Negative start time; see field (4) in $obj->{LOC}\n";
					$pass = 0;
				    }
				}
			    }
			    
			    # make sure that if it's a SPKR-INFO, an IP, or a CB the duration is <NA>
			    # else the duration must be a number and is non-negative
			    #
			    if ($obj->{TYPE} =~ /(SPKR-INFO|IP|CB)/i) {
				if ($obj->{TDUR} !~ /^<NA>$/i) {
				    print "ERROR: $obj->{TYPE} should not have any duration; see field (5) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } else {
				if ($obj->{TDUR} !~ /^(\-{0,1}\d+\.?\d*|\-{0,1}\.\d+)\**$/) {
				    print "ERROR: Expected duration to be a number; see field (5) in $obj->{LOC}\n";
				    $pass = 0;
				} else {
				    my $tdur = $obj->{TDUR};
				    $tdur =~ s/\*//;
				    if ($tdur < 0) {
					print "ERROR: Negative duration; see field (5) in $obj->{LOC}\n";
					$pass = 0;
				    }
				}
			    }
			    
			    # make sure that if it's not a LEXEME or a NON-LEX it doesn't have any orthography
			    #
			    if ($obj->{TYPE} =~ /(SEGMENT|NOSCORE|NO_RT_METADATA|NON-SPEECH|FILLER|EDIT|SU|IP|CB|A\/P|SPEAKER|SPKR-INFO)/i) {
				if ($obj->{ORTHO} !~ /^<NA>$/i) {
				    print "ERROR: Value for the orthography field for $obj->{TYPE} should be <NA>; see field (6) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    }
			    
			    # make sure the subtype for various type is valid
			    #
			    if ($obj->{TYPE} =~ /SEGMENT/i) {
				if ($obj->{STYPE} !~ /^(eval|<NA>)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /(NOSCORE|NO_RT_METADATA|SPEAKER)/i) {
				if ($obj->{STYPE} !~ /^<NA>$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /A\/P/i) {
				if ($obj->{STYPE} !~ /^<NA>$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /LEXEME/i) {
				if ($obj->{STYPE} !~ /^(lex|fp|frag|un-lex|for-lex|alpha|acronym|interjection|propernoun|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
				if ($obj->{STYPE} =~ /alpha/i && $obj->{ORTHO} !~ /^([A-Z]\.|[A-Z]\.\'*s|[\x80-\xff]{2}\.|<NA>)$/i) {
				    print "ERROR: Invalid orthography for alpha $obj->{TYPE}; see field (6) '$obj->{ORTHO}' in $obj->{LOC}\n";
				    $pass = 0;
			        } 
				my $ww = '[\[\]A-Z\.\-\'\x80-\xff]+';
				if (! $allowSCTKExtensions){
				    if ($obj->{ORTHO} !~ /^(${ww}|<NA>|%?[A-Z\.\x80-\xff]+-?)$/i) {
					print "WARNING: Invalid orthography for $obj->{TYPE}; see field (6) '$obj->{ORTHO}' in $obj->{LOC}\n";
				    }
				} else {
				    ### Destructively handle recursive alternations
				    my $_t = $obj->{ORTHO};
				    my $altExp = "\\{((_${ww})+_\\/)*(_${ww})+_\\}";
				    while ($_t =~ s/_${altExp}_/_/g){ ; }
				    if ($_t !~ /^(${ww}|(${ww}_)*$ww|$altExp|<NA>|%?[A-Z\.\x80-\xff]+-?)$/i) {
					print "WARNING: Invalid orthography with SCTK Extensions for $obj->{TYPE}; see field (6) '$obj->{ORTHO}' in $obj->{LOC}\n";
				    }
				}
			    } elsif ($obj->{TYPE} =~ /NON-LEX/i) {
				#### fix lipsmack!!!!
				if ($obj->{STYPE} !~ /^(laugh|breath|lipsmack|cough|sneeze|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype $obj->{STYPE}; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /NON-SPEECH/i) {
				if ($obj->{STYPE} !~ /^(noise|music|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /FILLER/i) {
				if ($obj->{STYPE} !~ /^(filled_pause|discourse_marker|explicit_editing_term|discourse_response|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /EDIT/i) {
				if ($obj->{STYPE} !~ /^(repetition|restart|revision|simple|complex|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /SU/i) {
				if ($obj->{STYPE} !~ /^(statement|question|incomplete|backchannel|unannotated|discourse_response|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /IP/i) {
				if ($obj->{STYPE} !~ /^(edit|filler|edit\&filler|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /CB/i) {
				if ($obj->{STYPE} !~ /^(coordinating|clausal|other)$/i) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    } elsif ($obj->{TYPE} =~ /SPKR-INFO/i) {
				if ($obj->{STYPE} !~ /^(adult_male|adult_female|child|unknown)$/) {
				    print "ERROR: Invalid $obj->{TYPE} subtype; see field (7) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    }
			    
			    # make sure that the confidence field is a number it's from 0-1 else it's <NA>
			    #
			    if ($obj->{CONF} !~ /<NA>/) {
				if ($obj->{CONF} !~ /^(\-{0,1}\d+\.?\d*|\-{0,1}\.\d+)$/) {
				    print "ERROR: Expected confidence value to be a number or <NA>; see field (9) in $obj->{LOC}\n";
				    $pass = 0;
				} elsif ($obj->{CONF} < 0 || $obj->{CONF} > 1) {
				    print "ERROR: Expected confidence value to be [0,1]; see field (9) in $obj->{LOC}\n";
				    $pass = 0;
				}
			    }
			    
			    # make sure that the speaker id field for certain types match the speaker id
			    # given in the SPKR-INFO object
			    #
			    if( ! $opt_p )
			    {
					if ($type !~ /(SPKR-INFO|NOSCORE|NONSPEECH)/i && ! find_speaker($obj->{SPKR}, $data) ) {
					print "ERROR: Speaker $obj->{SPKR} doesn't match any of the speaker IDs in SPKR-INFO objects; see $obj->{LOC}\n";
					$pass = 0;
				}
			    }
			}
                    }
                }
	    }
	}
    }
    if (! $pass) {
	print "Validation aborted\n";
    }
    return $pass;
}

sub check_logic_errors {
    # check for logic errors such as overlapping words from the same speaker, etc.
    #
    my ($data, $mde_types) = @_;

    # sort the data 
    #
    foreach my $src (keys %{$data}) {
        foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach  my $type (keys %{$data->{$src}{$chnl}{$spkr}}) {
		    if ($type !~ /SPKR-INFO/i) {
			@{$data->{$src}{$chnl}{$spkr}{$type}} = 
			    sort {$a->{TBEG} <=> $b->{TBEG}} @{$data->{$src}{$chnl}{$spkr}{$type}};
		    }
		}
	    }
	}
    }

    if (check_metadata_overlap($data) &&
	check_metadata_content($data) &&
	check_partial_word_coverage($data) &&
	check_noscore_overlap($data)) {
	foreach my $type (@$mde_types) {
	    if ($type =~ /(SU|SPEAKER)/i) {
		if (! ensure_word_covered_by_metadata_of_type($data, $type)) {
		    return 0;
		}
	    }
	    if ($type =~ /(EDIT|FILLER)/i) {
		if (! ensure_ip_existed_for_edit_and_filler($data)) {
		    return 0;
		}
	    }
	}
	return 1;
    } else {
	return 0;
    }
}

sub check_metadata_overlap {
    # check if two metadata objects of the same type
    # from the same speaker overlap
    #
    my ($data) = @_;

    my $pass = 1;

    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach my $type (keys %{$data->{$src}{$chnl}{$spkr}}) {
		    if ($type !~ /SPKR-INFO/i) {
			my $prev_etime = 0;
			foreach my $obj (@{$data->{$src}{$chnl}{$spkr}{$type}}) {
			    if (($prev_etime - $obj->{TBEG}) > $ROUNDING_THRESHOLD) {
				print "ERROR: Speaker $spkr has ${type}s ending at $prev_etime and starting at $obj->{TBEG}; ${type}s overlap; see $obj->{LOC}\n";
				$pass = 0;
			    }
			    if ($type =~ /(CB|IP)/i) {
				$prev_etime = $obj->{TBEG};
			    } else {
				$prev_etime = $obj->{TBEG} + $obj->{TDUR};
			    }
			}
		    }
		}
	    }
	}
    }
    if (! $pass) {
	print "Validation aborted\n";
    }
    return $pass;
}

sub check_metadata_content {
    # check if a metadata object of type SU, EDIT, or FILLER contains any words.
    # The words could be missing or simply didn't get transcribed
    #
    my ($data) = @_;

    my $pass = 1;

    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach my $type (keys %{$data->{$src}{$chnl}{$spkr}}) {
		    if ($type =~ /(SU|EDIT|FILLER)/i) {
			foreach my $obj (@{$data->{$src}{$chnl}{$spkr}{$type}}) {
			    if (! find_word($data->{$src}{$chnl}{$spkr}{LEXEME}, $obj->{TBEG}, $obj->{TBEG} + $obj->{TDUR})) {
				print "ERROR: $type at $obj->{TBEG} contains no words; see $obj->{LOC}\n";
				$pass = 0;
			    }
			}
		    }
		}
	    }
	}
    }
    if (! $pass) {
	print "Validation aborted\n";
    }
    return $pass;
}

sub check_partial_word_coverage {
    # check if a given word partially overlaps with
    # any metadata object of type SU, EDIT, or FILLER
    #
    my ($data) = @_;
    
    my $pass = 1;
    
    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach my $type (keys %{$data->{$src}{$chnl}{$spkr}}) {
		    if ($type =~ /LEXEME/i) {
			foreach my $obj (@{$data->{$src}{$chnl}{$spkr}{$type}}) {
			    foreach my $mde_type (keys %{$data->{$src}{$chnl}{$spkr}}) {
				if ($mde_type =~ /(SU|EDIT|FILLER)/i) {
				    if (find_partial_coverage($data->{$src}{$chnl}{$spkr}{$mde_type}, $obj->{TBEG}, $obj->{TBEG} + $obj->{TDUR})) {
					print "ERROR: Word at $obj->{TBEG} is partially covered by $mde_type object; see $obj->{LOC}\n";
					$pass = 0;
				    }
				}
			    }
			}
		    }
		}
	    }
	}
    }
    if (! $pass) {
	print "Validation aborted\n";
    }
    return $pass;
}

sub check_noscore_overlap {
    # check if a metadata object of type
    # SU, EDIT, FILLER, SPEAKER, LEXEME, NONLEX, NONSPEECH
    # partially overlap with noscore regions
    #
    my ($data) = @_;

    my $pass = 1;

    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    next if (! defined ($data->{$src}{$chnl}{"<NA>"}{NOSCORE}));
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach my $type (keys %{$data->{$src}{$chnl}{$spkr}}) {
		    if ($type =~ /(SU|EDIT|FILLER|SPEAKER|LEXEME|NON-LEX|NON-SPEECH)/i) {
			foreach my $obj (@{$data->{$src}{$chnl}{$spkr}{$type}}) {
			    if (find_partial_coverage($data->{$src}{$chnl}{"<NA>"}{NOSCORE}, $obj->{TBEG}, $obj->{TBEG} + $obj->{TDUR})) {
				print "ERROR: Speaker $spkr has $type partially overlapping with NOSCORE tag at $obj->{TBEG}; see $obj->{LOC}\n";
				$pass = 0;
			    }
			}
		    }
		}
	    }
	}
    }
    if (! $pass) {
	print "Validation aborted\n";
    }
    return $pass;
}

sub ensure_word_covered_by_metadata_of_type {
    # make sure that all words belong to some metadata object of the given type.
    #
    my ($data, $mde_type) = @_;

    my $pass = 1;

    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach my $type (keys %{$data->{$src}{$chnl}{$spkr}}) {
		    if ($type =~ /LEXEME/i) {
			foreach my $obj (@{$data->{$src}{$chnl}{$spkr}{$type}}) {
			    if (! find_type($data->{$src}{$chnl}{$spkr}{$mde_type}, $obj->{TBEG}, $obj->{TBEG} + $obj->{TDUR})) {
				print "ERROR: Word at $obj->{TBEG} doesn't belong to any $mde_type object; see $obj->{LOC}\n";
				$pass = 0;
			    }
			}
		    }
		}
	    }
	}
    }
    if (! $pass) {
	print "Validation aborted\n";
    }
    return $pass;
}

sub ensure_ip_existed_for_edit_and_filler {
    # make sure that for each EDIT or FILLER there is
    # a valid IP that follows or precedes it
    #
    my ($data) = @_;

    my $pass = 1;

    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		if (exists($data->{$src}{$chnl}{$spkr}{SU})) {
		    my @sus = @{$data->{$src}{$chnl}{$spkr}{SU}};
		    for (my $i = 0; $i < @sus; $i++) {
#			print "SU $sus[$i]->{TBEG}\n";
			my @edits = ();
			my @fillers = ();
			my @ips = ();
			if (exists($data->{$src}{$chnl}{$spkr}{EDIT})) {
			    @edits = extract_mde_of_type($sus[$i]->{TBEG},
							 $sus[$i]->{TBEG} + $sus[$i]->{TDUR},
							 $i,
							 "EDIT",
							 $data->{$src}{$chnl}{$spkr}{EDIT},
							 \@sus);
			}
			if (exists($data->{$src}{$chnl}{$spkr}{FILLER})) {
			    @fillers = extract_mde_of_type($sus[$i]->{TBEG},
							   $sus[$i]->{TBEG} + $sus[$i]->{TDUR},
							   $i,
							   "FILLER",
							   $data->{$src}{$chnl}{$spkr}{FILLER},
							   \@sus);
			}
			if (exists($data->{$src}{$chnl}{$spkr}{IP})) {
			    @ips = extract_mde_of_type($sus[$i]->{TBEG},
						       $sus[$i]->{TBEG} + $sus[$i]->{TDUR},
						       $i,
						       "IP",
						       $data->{$src}{$chnl}{$spkr}{IP},
						       \@sus);
			}
#			print "edits\n";
#			print Dumper(@edits);
#			print "fillers\n";
#			print Dumper(@fillers);
#			print "ips\n";
#			print Dumper(@ips);
			
			foreach my $obj (@edits) {
			    my ($status, $ip_stype, $ip_pos) = find_object($obj->[0], $obj->[1], \@ips);
			    if ($status == 1) {
#				print "$status, edit_type=$obj->[1] ip_stype=$ip_stype $ip_pos\n";
				if ($ip_stype =~ /^EDIT$/i) {
#				    print "case 1\n";
				    $obj->[4] = 1;
				    $ips[$ip_pos]->[4] = 1;
				} elsif ($ip_stype =~ /^EDIT&FILLER$/i) {
				    foreach my $obj2 (@fillers) {
					my ($status2, $filler_type, $filler_pos) = find_object($obj->[0], "FILLER", \@fillers);
					if ($status2 == 1) {
#					    print "$status2, filler_type=FILLER filler_type=$filler_type $filler_pos\n";
#					    print "case 2\n";
					    $obj->[4] = 1;
					    $ips[$ip_pos]->[4] = 1;
                                            $fillers[$filler_pos]->[4] = 1;
					}
				    }
				}
			    }
			}
			
			foreach my $obj (@fillers) {
			    my ($status, $ip_stype, $ip_pos) = find_object($obj->[0], $obj->[1], \@ips);
			    if ($status == 1) {
#				print "$status, filler type=$obj->[1] ip stype=$ip_stype, $ip_pos\n";
#				print "case 3\n";

				if ($ip_stype =~ /^FILLER$/i) {
				    $obj->[4] = 1;
				    $ips[$ip_pos]->[4] = 1;
				}
			    }
			}

			foreach my $obj (@edits) {
			    if ($obj->[4] == 0) {
				print "ERROR: EDIT at $obj->[2] doesn't have a matching IP; see $obj->[3]\n";
				$pass = 0;
			    }
			}
			foreach my $obj (@fillers) {
			    if ($obj->[4] == 0) {
				print "ERROR: FILLER at $obj->[2] doesn't have a matching IP; see $obj->[3]\n";
				$pass = 0;
			    }
			}
			foreach my $obj (@ips) {
			    if ($obj->[4] == 0) {
				print "ERROR: IP at $obj->[2] doesn't have a matching EDIT and/or FILLER; see line $obj->[3]\n";
				$pass = 0;
			    }
			}
		    }
		}
	    }
	}
    }
    if (! $pass) {
	print "Validation aborted\n";
    }
    return $pass;
}

sub find_object {
    my ($time, $type, $data1) = @_;
#    print "find object ------------\n";
#    print "  $time $type\n";
    for (my $i=0; $i < @$data1; $i++) {
#	print "      ${$data1}[$i]->[0]\n";
	if (abs(${$data1}[$i]->[0] - $time) < $ROUNDING_THRESHOLD &&
	    ${$data1}[$i]->[4] != 1) {
#            print "      found\n";
	    return (1, ${$data1}[$i]->[1], $i);
	}
    }
    return (0, undef, undef);
}

sub find_partial_coverage {
    # determine if a given object (indicated by its start and end times)
    # partially overlap with any object of the given type
    #
    my ($data, $start, $end) = @_;

    my $obj;

    foreach $obj (@{$data}) {
	my $curr_beg = $obj->{TBEG};
	my $curr_end = $obj->{TBEG} + $obj->{TDUR};

	# case 1       #########
	#        ####
        #
	# case 2 ########
	#                 ####
	#
	# case 3 ########
	#          ####
	#
	# everything else is partially covered
	#
	if (less_than($end, $curr_beg) || equal_to($end, $curr_beg)) {
	    # case 1
	    next;
	} elsif (greater_than($start, $curr_end) || equal_to($start, $curr_end)) {
	    # case 2
	    next;
	} elsif ( ( greater_than($start, $curr_beg) || equal_to($start, $curr_beg) ) &&
		  ( less_than($end, $curr_end) || equal_to($end, $curr_end) ) ) {
	    # case 3
	    next;
	} else {
	    # others
	    return 1;
	}
    }
    return 0;
}

sub find_type {
    # determine if a given word (indicated by its start and end times)
    # belongs to any metadata object of the given type
    #
    my ($data, $token_start, $token_end) = @_;

    foreach my $mde_obj (@{$data}) {
	if ($token_start + $ROUNDING_THRESHOLD >= $mde_obj->{TBEG} &&
	    $token_end - $ROUNDING_THRESHOLD <= $mde_obj->{TBEG} + $mde_obj->{TDUR}) {
	    return 1;
	}
    }
    return 0;
}

sub find_word {
    # determine if a given metadata object (indicated by its start and end times)
    # contains any word
    #
    my ($data, $mde_obj_start, $mde_obj_end) = @_;

    foreach my $token (@{$data}) {
	if ($token->{TBEG} + $ROUNDING_THRESHOLD >= $mde_obj_start &&
	    $token->{TBEG} + $token->{TDUR} - $ROUNDING_THRESHOLD <= $mde_obj_end) {
	    return 1;
	}
    }
    return 0;
}

sub find_ip {
    # determine if an IP is at the time given
    #
    my ($data, $tbeg, $tend, $target) = @_;
    if ($tbeg && $tend && $target) {
	foreach my $ip (@{$data}) {
	    if ( $tbeg <= $ip->{TBEG} &&
		 $ip->{TBEG} <= $tend &&
		 abs($ip->{TBEG} - $target) < $ROUNDING_THRESHOLD ) {
		return $ip;
	    }
	}
    }
    return undef;
}

sub find_speaker {
    # determine if a given speaker is in the data
    #
    my ($src_spkr, $data) = @_;

    my ($curr_spkr);
    foreach my $src (keys %{$data}) {
	foreach my $chnl (keys %{$data->{$src}}) {
	    foreach my $spkr (keys %{$data->{$src}{$chnl}}) {
		foreach my $obj (@{$data->{$src}{$chnl}{$spkr}{'SPKR-INFO'}}) {
		    $curr_spkr = $obj->{SPKR};
		    if ($src_spkr =~ /^$curr_spkr$/i) {
			return 1;
		    }
		}
	    }
	}
    }
    return 0;
}

sub extract_mde_of_type {
    # for a given SU (indicated by its begin and end times)
    # return all the MDE objects of the given type in that SU
    #
    my ($su_tbeg, $su_tend, $su_pos, $mde_type, $data, $sus) = @_;
    my @ip_time = ();
    if ($mde_type =~ /EDIT/i) {
	foreach my $obj (@$data) {
	    if ( (greater_than($obj->{TBEG}, $su_tbeg) || equal_to ($obj->{TBEG}, $su_tbeg)) &&
		 (less_than($obj->{TBEG} + $obj->{TDUR}, $su_tend) || equal_to ($obj->{TBEG} + $obj->{TDUR}, $su_tend)) ) {
		push (@ip_time, [ $obj->{TBEG} + $obj->{TDUR}, uc($obj->{TYPE}), $obj->{TBEG}, $obj->{LOC}, 0 ]);
	    }
	}
    } elsif ($mde_type =~ /FILLER/i) {
	foreach my $obj (@$data) {
	    if ( (greater_than($obj->{TBEG}, $su_tbeg) || equal_to ($obj->{TBEG}, $su_tbeg)) &&
		 (less_than($obj->{TBEG} + $obj->{TDUR}, $su_tend) || equal_to ($obj->{TBEG} + $obj->{TDUR}, $su_tend)) ) {
		push (@ip_time, [ $obj->{TBEG}, uc($obj->{TYPE}), $obj->{TBEG}, $obj->{LOC}, 0 ]);
	    }
	}
    } elsif ($mde_type =~ /IP/i) {
	foreach my $obj (@$data) {
	    if ( (greater_than($obj->{TBEG}, $su_tbeg) && less_than($obj->{TBEG}, $su_tend) ) ||
		 equal_to($obj->{TBEG}, $su_tbeg) ) {
		push (@ip_time, [ $obj->{TBEG}, uc($obj->{STYPE}), $obj->{TBEG}, $obj->{LOC}, 0 ]);
	    } elsif ( equal_to($obj->{TBEG}, $su_tend) ) {
		if ($su_pos + 1 <= scalar($sus)) {
		    if ( equal_to($su_tend, $sus->[$su_pos + 1]->{TBEG}) &&
			 $obj->{STYPE} !~ /filler/i ) {
			push (@ip_time, [ $obj->{TBEG}, uc($obj->{STYPE}), $obj->{TBEG}, $obj->{LOC}, 0 ]);
		    }
		}
	    }
	}
    }
    return @ip_time;
}

sub sort_data {
   return ($a->{TBEG} < $b->{TBEG} - $ROUNDING_THRESHOLD ? -1 :
	   ($a->{TBEG} > $b->{TBEG} + $ROUNDING_THRESHOLD ?  1 :
	    $SORT_ORDER{$a->{TYPE}} <=> $SORT_ORDER{$b->{TYPE}}));
}

sub less_than {
    # if a < b return 1
    #
    my ($a, $b) = @_;
    if ($a + $ROUNDING_THRESHOLD < $b) {
	return 1;
    }
    return 0;
}

sub greater_than {
    # if a > b return 1
    #
    my ($a, $b) = @_;
    if ($a > $b + $ROUNDING_THRESHOLD) {
	return 1;
    }
    return 0;
}

sub equal_to {
    # if a == b return 1
    #
    my ($a, $b) = @_;
    if ( abs($a - $b) < $ROUNDING_THRESHOLD) {
	return 1;
    }
    return 0;
}

sub date_time_stamp {
    my ($sec, $min, $hour, $mday, $mon, $year, $wday, $yday, $isdst) = localtime();
    my @months = qw(Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec);
    my ($date, $time);

    $time = sprintf "%2.2d:%2.2d:%2.2d", $hour, $min, $sec;
    $date = sprintf "%4.4s %3.3s %s", 1900+$year, $months[$mon], $mday;
    return ($date, $time);
}
