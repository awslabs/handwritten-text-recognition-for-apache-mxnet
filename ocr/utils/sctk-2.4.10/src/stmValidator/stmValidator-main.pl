#!/usr/bin/perl -w

# STMVALIDATOR
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

use strict;
use Getopt::Std;
use Data::Dumper;

### Begin Local Modules
use STMRecord;
use STMList;
### End Local Modules

## This variable says if the program is installed as a standalone executable
my $Installed = 0;

my $debug = 0;

my $VERSION = "1";

my $USAGE = "\n\n$0 -i <STM file>\n\n".
    "Version: $VERSION\n".    
    "Description: This Perl program (version $VERSION) validates a given STM file.\n".
    "Options:\n".
    "  -l <language> : language\n".
    "  -s            : silent\n".
    "  -h            : print this help message\n".
    "Input:\n".
    "  -i <STM file>: an STM file\n\n";

use vars qw ($opt_i $opt_l $opt_s $opt_h);
getopts('i:l:sh');
die ("$USAGE") if( (! $opt_i) || ($opt_h) );

my $language = "english";
$language = $opt_l if($opt_l);

my $verbose = 2;
$verbose = 0 if($opt_s);

STMList::passFailValidate($opt_i, $language, $verbose);
