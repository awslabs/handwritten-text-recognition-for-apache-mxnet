#!/usr/bin/perl

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

my $exe = "stmValidator-main.pl";

my @modules = ();
my @file = ();

open EXE,"<$exe" || die "Failed to open executable $exe";

while (<EXE>)
{
    if ($_ =~ /my \$Installed = 0;/)
    {
		push @file, "my \$Installed = 1;\n";
    } 
    elsif ($_ =~ /\#\#\# Begin Local Modules/)
    {
		do
		{
	    	push(@modules, $1) if($_ =~ /use\s+(\S+)\s*;/);
	    	$_ = <EXE>;
		}
		while ($_ !~ /\#\#\# End Local Modules/);
		my $modExp = "use\\s+(".join("|",@modules).")";

		### Insert the modules 
		foreach my $mod(@modules)
		{
			open (MOD, "<$mod.pm") || die "Failed to open $mod.pm";
			
			while (<MOD>)
			{
				push(@file, $_) if ($_ !~ /$modExp/);
			}
			
			close MOD;
		}
		
		### Reset the package
		push @file, "package main;\n"
	} 
	else
	{
		push(@file, $_);
    }
}

print @file;
