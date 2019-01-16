#!/usr/bin/env perl

# ALIGN2HTML
# Author: Jerome Ajot
#
# This software was developed at the National Institute of Standards and Technology by 
# employees of the Federal Government in the course of their official duties. Pursuant
# to title 17 Section 105 of the United States Code this software is not subject to
# copyright protection and is in the public domain. ALIGN2HTML is an experimental system.
# NIST assumes no responsibility whatsoever for its use by other parties, and makes no
# guarantees, expressed or implied, about its quality, reliability, or any other
# characteristic. We would appreciate acknowledgement if the software is used.
#
# THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO EXPRESS
# OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING MERCHANTABILITY,
# OR FITNESS FOR A PARTICULAR PURPOSE.

use strict;
use warnings;

my $str = "";

open TAR, "tar zcf - --exclude CVS --exclude .DS_Store ./js/wz_jsgraphics.js ./images/*gif |" || die "TAR/UUENCODE Fialed";
binmode TAR;
read(TAR, $_, 90000000);
$str .= pack("u", $_) . "\n";
close TAR;

print "$str";
