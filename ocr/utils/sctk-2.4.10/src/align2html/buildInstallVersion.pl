#!/usr/bin/perl

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

my $exe = "align2html-main.pl";

my @modules = ();
my @file = ();

open EXE,"<$exe" || die "Failed to open executable $exe";
while (<EXE>){
    if ($_ =~ /my \$Installed = 0;/){
	push @file, "my \$Installed = 1;\n";
    } elsif ($_ =~ /\#\#\# Begin Local Modules/){
	do {
	    if ($_ =~ /use\s+(\S+)\s*;/){
		push @modules, $1;
	    }
	    $_ = <EXE>;
	} while ($_ !~ /\#\#\# End Local Modules/);
	my $modExp = "use\\s+(".join("|",@modules).")";
	#print "Modes = ".join(" ",@modules)." $modExp\n";
	### Insert the modules 
	foreach my $mod(@modules){
	    open (MOD, "<$mod.pm") || die "Failed to open $mod.pm";
	    while (<MOD>){
		push(@file, $_) if ($_ !~ /$modExp/);
	    }
	    close MOD;
	}
	### Reset the package
	push @file, "package main;\n"
    } elsif ($_ =~ /die "HERE DOCUMENT NOT BUILT"/){
        push @file, 'my $here = "";'."\n";
        push @file, "\$here = << 'EOTAR';\n";
	open TARPACK, "packImageTarFile" || die "Failed to open the packed Image tar file";
	while (<TARPACK>){
	    push @file, $_;
	}
	close TARPACK;
	push @file, pack("u", $_);
        push @file, "EOTAR\n";

	push @file, 'open UNTAR, "| (cd $outDir ; tar zxf -)" || die "Failed to UUDECODE/TAR"'.";\n";
	push @file, 'binmode(UNTAR);'."\n";
	push @file, 'print UNTAR unpack("u",$here);'."\n";
	push @file, 'close UNTAR'."\n";
    } else {
	push @file, $_;
    }
}
print @file;

#	cat align2html.pl | perl -ne 'if ($$_ =~ /die "HERE DOCUMENT NOT BUILT"/) { system "cat DATA.tgz.uu" } else { print }' > align2html-combined.pl
