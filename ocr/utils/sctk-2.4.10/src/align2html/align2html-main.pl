#!/usr/bin/perl -w

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
use Getopt::Long;
use Data::Dumper;
use File::Path;

### Begin Local Modules
use SegmentGroup;
use Segment;
use Token;
### End Local Modules

Getopt::Long::Configure(qw( auto_abbrev no_ignore_case ));

my $VERSION = "0.6";
my $AlignFile = "";
my $MapFile = "";
my $Outputdir = "";
my $NonLinearSegments = 0;
my $pixelpersec = 100;
my $maxscalewidth = 0;

## This variable says if the program is installed as a standalone executable
my $Installed = 0;

sub usage
{
	print "perl align2html.pl [OPTIONS] -i alignfile -o outputdir\n";
	print "Version: $VERSION\n";
	print "\n";
	print "Required file arguments:\n";
	print "  -a, --alignfile         Path to the Alignment file.\n";
	print "  -o, --outputdir         Path to the output directory.\n";
	print "\n";
	print "Options\n";
	print "  -m, --mapfile           Path to the Mapping file.\n";
	print "\n";
}

sub extractJavaImages
{
    my ($outDir) = @_;
    die "HERE DOCUMENT NOT BUILT";
}

sub unique
{
	my %saw;
	return grep(!$saw{$_}++, sort @_);
}

my %SegGroups;
my %Alignments;
my %FileChannelSG;
my %Overlap;
my %CumulOverlap;

sub addSGLine
{
	my ($inputline) = @_;
	
	my $SG;
	my $Segmnt;
	my $SYSREF = "";
	my $SegID;
	my $Tokn;
	my $SpkrID;
	
	my $SG_SegGrpID	  = $inputline->[1];
	my $SG_SegGrpFile = $inputline->[2];
	my $SG_SegGrpChan = $inputline->[3];
	my $SG_RefSegID	  = $inputline->[5];
	my $SG_HypSegID	  = $inputline->[16];
	
	my $ok = 0;
	
	if($SG_RefSegID ne "" && $inputline->[9] ne "")
	{
		$SYSREF = "REF";
		$SegID = $SG_RefSegID;
		$SpkrID = "ref:" . $inputline->[8];
		
		my $dur = 0;
		$dur = sprintf("%.3f", $inputline->[11]-$inputline->[10]) if( ($inputline->[10] ne "") || ($inputline->[11] ne "") );
		
		$Tokn = new Token($inputline->[9], $inputline->[10], $dur, $inputline->[12], $SYSREF, $inputline->[1], $inputline->[5], $SpkrID, $inputline->[13], $inputline->[14], $inputline->[15], $inputline->[6], $inputline->[7]);
		$ok = 1;
	}
	
	if($SG_HypSegID ne "" && $inputline->[20] ne "")
	{
		$SYSREF = "SYS";
		$SegID = $SG_HypSegID;
		$SpkrID = "hyp:" . $inputline->[19];
		
		my $dur = 0;
		$dur = sprintf("%.3f", $inputline->[22]-$inputline->[21]) if( ($inputline->[21] ne "") || ($inputline->[22] ne "") );
		
		$Tokn = new Token($inputline->[20], $inputline->[21], $dur, $inputline->[23], $SYSREF, $inputline->[1], $inputline->[16], $SpkrID, $inputline->[24], $inputline->[25], $inputline->[26], $inputline->[17], $inputline->[18]);
		$ok = 1;
	}
	
	if($ok)
	{
		$SegGroups{$SG_SegGrpID} = new SegmentGroup($SG_SegGrpID, $SG_SegGrpFile, $SG_SegGrpChan) if(!exists($SegGroups{$SG_SegGrpID}));
		$SG = $SegGroups{$SG_SegGrpID};
			
		if(!exists( $SG->{$SYSREF}{$SegID} ))
		{
			$Segmnt = new Segment($SegID, $SpkrID);
			
			$SG->addSysSegment($Segmnt) if($SYSREF eq "SYS");
			$SG->addRefSegment($Segmnt) if($SYSREF eq "REF");
		}
		else
		{
			$Segmnt = $SG->{$SYSREF}{$SegID};
		}
		
		$Tokn->DoDisplay();
		$Segmnt->AddToken($Tokn);
	}
}

sub addAlignments
{
	my ($inputline) = @_;
	
	my $SG_SegGrpID	  = $inputline->[1];
	my $SG_Eval		  = $inputline->[4];
	my $SG_RefTknID	  = $inputline->[9];
	my $SG_HypTknID	  = $inputline->[20];
	
	push( @{ $Alignments{$SG_SegGrpID} }, [($SG_Eval, $SG_RefTknID, $SG_HypTknID)]);
	$SegGroups{$SG_SegGrpID}->{ALIGNED} = 1;
	
	if($SG_RefTknID ne "")
	{
		$SegGroups{$SG_SegGrpID}->GetToken($SG_RefTknID)->SetWidthLine(2);
		$SegGroups{$SG_SegGrpID}->GetToken($SG_RefTknID)->DoDisplay();
	}
	
	if($SG_HypTknID ne "")
	{
		$SegGroups{$SG_SegGrpID}->GetToken($SG_HypTknID)->SetWidthLine(2);
		$SegGroups{$SG_SegGrpID}->GetToken($SG_HypTknID)->DoDisplay();
	}
}

sub loadAlignFile
{
	my ($alignfile) = @_;
	my $linenum = 0;
	
	open(ALIGNFILE, $alignfile) or die "Unable to open for read alignfile file '$alignfile'";
    
    while (<ALIGNFILE>)
    {
        chomp;
        my $fileline = $_;
        $linenum++;
        
        next if($fileline !~ /^\[ALIGNMENT\]/);
        next if($fileline =~ /^\[ALIGNMENT\]\s*Aligned/);
        
        $fileline =~ s/^\[ALIGNMENT\]//g;
    	$fileline =~ s/\s+//g;
    	$fileline =~ s/,/, /g;
    	
    	my @spliteedline = split(",", $fileline);

		foreach (@spliteedline) { $_ =~ s/\s+//g; }
    	       	
    	addSGLine(\@spliteedline)     if($spliteedline[0] eq "SG" );
    	addAlignments(\@spliteedline) if($spliteedline[0] eq "YES");
    }
    
    close(ALIGNFILE);
}

sub GetHTMLHeader
{
	my ($ID) = @_;
	
	my $out = "";
	
	$out .= "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">\n";
	$out .= "\n";
	$out .= "<html>\n";
	$out .= "<head>\n";
	$out .= "<meta content=\"text/html; charset=utf-8\" http-equiv=\"content-type\">\n";
	$out .= "<title>Segment Group ID $ID</title>\n";
	$out .= "<script type=\"text/javascript\" src=\"js/wz_jsgraphics.js\"></script>\n";
	$out .= "<script type=\"text/javascript\">\n";
	$out .= "<!--\n";
	$out .= "var jg;\n";
	$out .= "var scale = 1;\n";
	$out .= "\n";
	$out .= "function Init()\n";
	$out .= "{\n";
	$out .= "jg = new jsGraphics(\"Graph\");\n";
	$out .= "scale = 1;\n";
	$out .= "}\n";
	$out .= "\n";
	$out .= "function myDraw()\n";
	$out .= "{\n";
	$out .= "jg.clear();\n";
	
	return $out;	
}

sub GetHTMLFooter
{
	my ($end, $heigh, $warning, $SG) = @_;
	
	my $out = "";
	
	$out .= "jg.paint();\n";
	$out .= "}\n";
	$out .= "\n";
	$out .= "//-->\n";
	$out .= "</script>\n";
	$out .= "</head>\n";
	$out .= "\n";
	$out .= "<body bgcolor=\"white\" onload=\"Init(); myDraw()\">\n";
	$out .= "\n";
	$out .= sprintf("<div id=\"Graph\" style=\"background-color:white;height:%dpx;width:%dpx;border-style:solid;border-width:0;position:relative;top:0;left:0;\"></div>\n", $heigh, $end);
	$out .= "\n";
	$out .= "<font face=\"courier\">\n";
	$out .= "<a href=\"#\" onclick=\"scale -= 0.5; myDraw(); return false\">[-5]</a>\n";
	$out .= "<a href=\"#\" onclick=\"scale -= 0.1; myDraw(); return false\">[-]</a>\n";
	$out .= "<a href=\"#\" onclick=\"scale  = 1; myDraw(); return false\">[Reset]</a>\n";
	$out .= "<a href=\"#\" onclick=\"scale += 0.1; myDraw(); return false\">[+]</a>\n";
	$out .= "<a href=\"#\" onclick=\"scale += 0.5; myDraw(); return false\">[+5]</a>\n";
	
	
	
	$out .= "</font>\n";
	$out .= PrintMapping($SG);
	$out .= PrintWarning() if($warning == 1);
	$out .= "</body>\n";
	$out .= "</html>\n";

	return $out;	
}

sub CalculateMaxScale
{
	my ($start, $end, $stepsec, $heigh) = @_;
	my $beginxline = $pixelpersec*sprintf("%.0f", $start/$pixelpersec);
	my $endxline = $pixelpersec*sprintf("%.0f", $end/$pixelpersec);
	
	$beginxline -= $pixelpersec if( ($beginxline > $start) && ($beginxline > $pixelpersec) );
	$endxline += $pixelpersec if($endxline < $end);	
	$maxscalewidth = $endxline - $start + $pixelpersec;
}

sub GetDrawScale
{
	my ($start, $end, $stepsec, $heigh) = @_;
		
	my $output = "";
	$output .= sprintf("jg.setColor(\"orange\");\n");
	$output .= sprintf("jg.setStroke(Stroke.DOTTED);\n");
	$output .= sprintf("jg.setFont(\"arial\",\"8px\",Font.BOLD);\n");
	
	my $beginxline = $pixelpersec*sprintf("%.0f", $start/$pixelpersec);
	my $endxline = $pixelpersec*sprintf("%.0f", $end/$pixelpersec);
		
	$beginxline -= $pixelpersec if( ($beginxline > $start) && ($beginxline > $pixelpersec) );
	$endxline += $pixelpersec if($endxline < $end);
			
	for(my $i=$beginxline; $i<=$endxline; $ i += $stepsec*$pixelpersec)
	{
		$output .= sprintf("jg.drawLine(scale*%.0f, 0, scale*%.0f, %.0f);\n", $i - $start + $pixelpersec, $i - $start + $pixelpersec, $heigh);
		$output .= sprintf("jg.drawStringRect(\"%s\", scale*%.1f - 9, %.0f + 2, 20, \"center\");\n", $i/$pixelpersec, $i - $start + $pixelpersec, $heigh);
	}
		
	return $output;
}

sub LinkToken
{
	my ($tkn1x, $tkn1y, $tkn2x, $tkn2y, $decalx, $type) = @_;
	
	my $color = "black";
	$color = "blue" if($type eq "spkrerr");
	$color = "red" if($type eq "sub");
	$color = "green"if($type eq "corr");
	
	my $out = "";
	$out .= sprintf("jg.setColor(\"%s\");\n", $color);
	$out .= sprintf("jg.setStroke(2);\n");
	$out .= sprintf("jg.drawLine(%.0f*scale, %.0f, %.0f*scale, %.0f);\n", $tkn1x - $decalx, $tkn1y, $tkn2x - $decalx, $tkn2y);
	
	return $out;
}

sub InsSubToken
{
	my ($tknx, $tkny, $decalx, $type) = @_;
	
	my $multi = 1;
	$multi = -1 if($type eq "SYS");
	
	my $color = "black";
	
	my $out = "";
	$out .= sprintf("jg.setColor(\"%s\");\n", $color);
	$out .= sprintf("jg.setStroke(1);\n");
	$out .= sprintf("jg.drawLine(%.0f*scale, %.0f, %.0f*scale, %.0f);\n", $tknx - $decalx, $tkny, $tknx - $decalx, $tkny + $multi*13);
	$out .= sprintf("jg.drawLine(%.0f*scale - 8, %.0f, %.0f*scale + 8, %.0f);\n", $tknx - $decalx, $tkny + $multi*13 , $tknx - $decalx, $tkny + $multi*13);
	$out .= sprintf("jg.drawLine(%.0f*scale - 5, %.0f, %.0f*scale + 5, %.0f);\n", $tknx - $decalx, $tkny + $multi*15 , $tknx - $decalx, $tkny + $multi*15);
	$out .= sprintf("jg.drawLine(%.0f*scale - 2, %.0f, %.0f*scale + 2, %.0f);\n", $tknx - $decalx, $tkny + $multi*17 , $tknx - $decalx, $tkny + $multi*17);
	
	return $out;
}

sub InsOptSubToken
{
	my ($tknx, $tkny, $decalx, $type) = @_;
	
	my $multi = 1;
	$multi = -1 if($type eq "SYS");
	
	my $color = "green";
	
	my $out = "";
	$out .= sprintf("jg.setColor(\"%s\");\n", $color);
	$out .= sprintf("jg.setStroke(1);\n");
	$out .= sprintf("jg.drawLine(%.0f*scale, %.0f, %.0f*scale, %.0f);\n", $tknx - $decalx, $tkny, $tknx - $decalx, $tkny + $multi*13);
	$out .= sprintf("jg.drawLine(%.0f*scale - 8, %.0f, %.0f*scale + 8, %.0f);\n", $tknx - $decalx, $tkny + $multi*13 , $tknx - $decalx, $tkny + $multi*13);
	$out .= sprintf("jg.drawLine(%.0f*scale - 5, %.0f, %.0f*scale + 5, %.0f);\n", $tknx - $decalx, $tkny + $multi*15 , $tknx - $decalx, $tkny + $multi*15);
	$out .= sprintf("jg.drawLine(%.0f*scale - 2, %.0f, %.0f*scale + 2, %.0f);\n", $tknx - $decalx, $tkny + $multi*17 , $tknx - $decalx, $tkny + $multi*17);
	
	return $out;
}

sub DrawLinkToken
{
	my ($sg, $tknrefid, $tknhypid, $decalx, $type) = @_;
	
	my $tknref;
	my $tknhyp;
	
	if($type eq "I")
	{
		$sg->{NINS}++;
		$tknhyp = $sg->GetToken($tknhypid);
		return InsSubToken($tknhyp->{XMIDPOS}, $tknhyp->{YPOS}, $sg->{MINX} - $pixelpersec, $tknhyp->{REFORSYS});
	}
	
	if($type eq "D")
	{
		$sg->{NDEL}++;
		$sg->{NREF}++;
		$tknref = $sg->GetToken($tknrefid);
		return InsSubToken($tknref->{XMIDPOS}, $tknref->{YPOS}, $sg->{MINX} - $pixelpersec, $tknref->{REFORSYS});
	}
	
	if($type eq "C")
	{
		$sg->{NCORR}++;
		$tknref = $sg->GetToken($tknrefid);
		$tknhyp = $sg->GetToken($tknhypid);
		
		my $refdef = 0;
		$refdef = 1 if($tknrefid ne "");
		
		my $hypdef = 0;
		$hypdef = 1 if($tknhypid ne "");
		
		$sg->{NREF}++ if($refdef);
		
		return LinkToken($tknref->{XMIDPOS}, $tknref->{YPOS}, $tknhyp->{XMIDPOS}, $tknhyp->{YPOS}, $sg->{MINX} - $pixelpersec, "corr") if($refdef && $hypdef);
		return InsOptSubToken($tknref->{XMIDPOS}, $tknref->{YPOS}, $sg->{MINX} - $pixelpersec, $tknref->{REFORSYS}) if($refdef && !$hypdef);
		return InsOptSubToken($tknhyp->{XMIDPOS}, $tknhyp->{YPOS}, $sg->{MINX} - $pixelpersec, $tknhyp->{REFORSYS}) if(!$refdef && $hypdef);	
	}
	
	if($type eq "S")
	{
		$sg->{NSUB}++;
		$sg->{NREF}++;
		$tknref = $sg->GetToken($tknrefid);
		$tknhyp = $sg->GetToken($tknhypid);
		return LinkToken($tknref->{XMIDPOS}, $tknref->{YPOS}, $tknhyp->{XMIDPOS}, $tknhyp->{YPOS}, $sg->{MINX} - $pixelpersec, "sub");
	}
	
	if($type eq "P")
	{
		$sg->{NSPKRERR}++;
		$sg->{NREF}++;
		$tknref = $sg->GetToken($tknrefid);
		$tknhyp = $sg->GetToken($tknhypid);
		return LinkToken($tknref->{XMIDPOS}, $tknref->{YPOS}, $tknhyp->{XMIDPOS}, $tknhyp->{YPOS}, $sg->{MINX} - $pixelpersec, "spkrerr");
	}
	
	die "Unknown type $type";
}

sub PrintWarning
{
	return "<p>WARNING: Some words do not have times specified for them.  For these cases, word times <img src='images/white_lt.gif' border='0' /> are estimated from the segment times.</p>\n";
}

sub PrintMapping
{
	my($SG) = @_;
	my $out = "";
	my @listmapping;
		
	if($MapFile ne "")
	{
		$out .= "<br><br>MAPPING:<br>\n";
	
		open(MAPFILE, $MapFile) or die "$?";
		
		while (<MAPFILE>)
		{
			chomp;
			next if($_ =~ /^File,Channel,RefSpeaker,SysSpeaker,isMapped,timeOverlap$/);
			
			my @info = split(/,/, $_);
			
			if($info[4] eq "mapped")
			{
				my $tmp1 = "ref:" . uc($info[2]);
				my $tmp2 = "hyp:" . uc($info[3]);
				
				if($SG->isInSpeakers($tmp1) == 1 && $SG->isInSpeakers($tmp2) == 1)
				{
					$tmp1 =~ s/^ref://;
					$tmp1 =~ s/^hyp://;
					$tmp2 =~ s/^ref://;
					$tmp2 =~ s/^hyp://;
					push(@listmapping, "$tmp1 <=> $tmp2<br>\n");
				}
			}
		}
			
		close MAPFILE;
		
		my @listsorted = sort(@listmapping);
		
		foreach my $mapp (@listsorted)
		{
			$out .= "$mapp\n";
		}
	}
		
	return $out;
}

sub DrawKey
{
	my ($ystart) = @_;
	
	my $ystartpos = $ystart;
	
	my $out = "";
	$out .= sprintf("jg.setFont(\"verdana\",\"12px\",Font.PLAIN);\n");
	$out .= sprintf("jg.setStroke(1);\n");
	
	$out .= sprintf("jg.setColor(\"black\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 15, $ystartpos      , 15, $ystartpos + 13);
	$out .= sprintf("jg.drawLine(%.0f - 8, %.0f, %.0f + 8, %.0f);\n", 15, $ystartpos + 13 , 15, $ystartpos + 13);
	$out .= sprintf("jg.drawLine(%.0f - 5, %.0f, %.0f + 5, %.0f);\n", 15, $ystartpos + 15 , 15, $ystartpos + 15);
	$out .= sprintf("jg.drawLine(%.0f - 2, %.0f, %.0f + 2, %.0f);\n", 15, $ystartpos + 17 , 15, $ystartpos + 17);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "Deletion", 50, $ystartpos + 4);
	
	$out .= sprintf("jg.setColor(\"green\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 215, $ystartpos      , 215, $ystartpos + 13);
	$out .= sprintf("jg.drawLine(%.0f - 8, %.0f, %.0f + 8, %.0f);\n", 215, $ystartpos + 13 , 215, $ystartpos + 13);
	$out .= sprintf("jg.drawLine(%.0f - 5, %.0f, %.0f + 5, %.0f);\n", 215, $ystartpos + 15 , 215, $ystartpos + 15);
	$out .= sprintf("jg.drawLine(%.0f - 2, %.0f, %.0f + 2, %.0f);\n", 215, $ystartpos + 17 , 215, $ystartpos + 17);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 200, \"left\");\n", "Optionally Deletable", 250, $ystartpos + 4);
	
	$ystartpos += 50;
	
	$out .= sprintf("jg.setColor(\"black\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 15, $ystartpos      , 15, $ystartpos - 13);
	$out .= sprintf("jg.drawLine(%.0f - 8, %.0f, %.0f + 8, %.0f);\n", 15, $ystartpos - 13 , 15, $ystartpos - 13);
	$out .= sprintf("jg.drawLine(%.0f - 5, %.0f, %.0f + 5, %.0f);\n", 15, $ystartpos - 15 , 15, $ystartpos - 15);
	$out .= sprintf("jg.drawLine(%.0f - 2, %.0f, %.0f + 2, %.0f);\n", 15, $ystartpos - 17 , 15, $ystartpos - 17);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "Insertion", 50, $ystartpos - 17);
	
	$out .= sprintf("jg.setColor(\"green\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 215, $ystartpos      , 215, $ystartpos - 13);
	$out .= sprintf("jg.drawLine(%.0f - 8, %.0f, %.0f + 8, %.0f);\n", 215, $ystartpos - 13 , 215, $ystartpos - 13);
	$out .= sprintf("jg.drawLine(%.0f - 5, %.0f, %.0f + 5, %.0f);\n", 215, $ystartpos - 15 , 215, $ystartpos - 15);
	$out .= sprintf("jg.drawLine(%.0f - 2, %.0f, %.0f + 2, %.0f);\n", 215, $ystartpos - 17 , 215, $ystartpos - 17);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 200, \"left\");\n", "Optionally Insertable", 250, $ystartpos - 17);
	
	$ystartpos += 15;
	
	$out .= sprintf("jg.setStroke(2);\n");
	
	$out .= sprintf("jg.setColor(\"green\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 205, $ystartpos      , 225, $ystartpos);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "Correct", 250, $ystartpos - 7);
	
	$out .= sprintf("jg.setColor(\"red\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 5, $ystartpos      , 25, $ystartpos);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "Substitution", 50, $ystartpos - 7);
	
	$ystartpos += 25;
	
	$out .= sprintf("jg.setColor(\"blue\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 5, $ystartpos      , 25, $ystartpos);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "Speaker error", 50, $ystartpos - 7);
	
	$out .= sprintf("jg.setStroke(1);\n");
	$out .= sprintf("jg.setColor(\"brown\");\n");
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 205, $ystartpos      , 225, $ystartpos);
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 205, $ystartpos -4   , 205, $ystartpos + 4);
	$out .= sprintf("jg.drawLine(%.0f, %.0f, %.0f, %.0f);\n"        , 225, $ystartpos -4   , 225, $ystartpos + 4);
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "Empty Segment", 250, $ystartpos - 7);
	
	$ystartpos += 25;
	
	$out .= sprintf("jg.setColor(\"#eeffee\");\n");
	$out .= sprintf("jg.fillRect(%.0f, %.0f, %.0f, %.0f);\n", 5, $ystartpos - 3     , 25, 8);
	$out .= sprintf("jg.setColor(\"green\");\n");
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "References", 50, $ystartpos - 7);
	
	$ystartpos += 25;
	
	$out .= sprintf("jg.setColor(\"#ffeeee\");\n");
	$out .= sprintf("jg.fillRect(%.0f, %.0f, %.0f, %.0f);\n", 5, $ystartpos - 3    , 25, 8);
	$out .= sprintf("jg.setColor(\"red\");\n");
	$out .= sprintf("jg.drawStringRect(\"%s\", %.0f, %.0f, 100, \"left\");\n", "Systems", 50, $ystartpos - 7);
	
	return $out;
}

GetOptions
(
    'alignfile=s' => \$AlignFile,
    'mapfile=s'   => \$MapFile,
    'outputdir=s' => \$Outputdir,
    'version'     => sub { print "align2html version: $VERSION\n"; exit },
    'help'        => sub { usage (); exit },
);

die "ERROR: An Align file must be set." if($AlignFile eq "");
die "ERROR: An Output directory must be set." if($Outputdir eq "");

system("mkdir -p $Outputdir");

loadAlignFile($AlignFile);

foreach my $SegGrpID ( sort keys %SegGroups )
{
	my $out = "";
	my $outputfilename = "segmentgroup-$SegGrpID.html";
	my $mySG = $SegGroups{$SegGrpID};
	
	$mySG->Compute();
	
	$out .= GetHTMLHeader($SegGrpID);
	CalculateMaxScale($mySG->{MINX}, $mySG->{MAXX}, 1, $mySG->{HEIGHT});
	$out .= $mySG->GetFillREFHYP($maxscalewidth);
	$out .= GetDrawScale($mySG->{MINX}, $mySG->{MAXX}, 1, $mySG->{HEIGHT});
	$out .= $mySG->GetSeparationLines($maxscalewidth);
	$out .= $mySG->GetDraw($pixelpersec);
	
	foreach (@{ $Alignments{$SegGrpID} })
	{
		my $typ = $_->[0];
		my $tknrefid = $_->[1];
		my $tknhypid = $_->[2];
			
		$out .= DrawLinkToken($mySG, $tknrefid, $tknhypid, $mySG->{MINX} - $pixelpersec, $typ);
	}
	
	$out .= DrawKey($mySG->{HEIGHT} + 25);
	
	$out .= GetHTMLFooter($mySG->{MAXX} - $mySG->{MINX} + $pixelpersec, $mySG->{HEIGHT} + 185, $mySG->{HASFAKETIME}, $mySG);
	
	open(FILESG, ">$Outputdir/$outputfilename") or die "Can't open file $Outputdir/$outputfilename for write";
	print FILESG "$out\n";
	close FILESG;
	
	my @refspkrs;
	foreach my $segref (keys %{ $mySG->{REF} }){ push(@refspkrs, $mySG->{REF}{$segref}->{SPKRID}) if($mySG->{REF}{$segref}->{SPKRID} ne "ref:INTER_SEGMENT_GAP"); }
	my $countSpeakers = scalar unique @refspkrs;
	
	my @sysspkrs;
	foreach my $segsys (keys %{ $mySG->{SYS} }){ push(@refspkrs, $mySG->{SYS}{$segsys}->{SPKRID}) if(!($mySG->{SYS}{$segsys}->HasOnlyOneFakeToken())); }
	my $counthypSpeakers = scalar unique @sysspkrs;
	
	$mySG->{ALIGNED} = 1 if( ($countSpeakers == 0) && ($counthypSpeakers == 0) );
	$mySG->{ALIGNED} = 1 if( ($mySG->{ALIGNED} == 0) && ($countSpeakers == 1) && ($counthypSpeakers == 0) && ($mySG->{HASFAKETIME} == 1) );
	
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NCORR} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NCORR}));
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NINS} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NINS}));
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NSUB} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NSUB}));
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NDEL} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NDEL}));
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NREF} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NREF}));
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NSPKRERR} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NSPKRERR}));
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{TOTTIME} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{TOTTIME}));
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{TOTTIMEALIGNED} = 0 if(!exists($FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{TOTTIMEALIGNED}));
	
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NCORR} += $mySG->{NCORR};
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NINS} += $mySG->{NINS};
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NSUB} += $mySG->{NSUB};
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NDEL} += $mySG->{NDEL};
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NREF} += $mySG->{NREF};
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{NSPKRERR} += $mySG->{NSPKRERR};
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{TOTTIME} += $mySG->{ET} - $mySG->{BT};
	push( @{ $FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{LISTSG} }, $SegGrpID);
	push( @{ $FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{LISTSGALIGNED} }, $SegGrpID) if($mySG->{ALIGNED} == 1);
	$FileChannelSG{$mySG->{FILE}}{$mySG->{CHANNEL}}{TOTTIMEALIGNED} += $mySG->{ET} - $mySG->{BT} if($mySG->{ALIGNED} == 1);
	
	$Overlap{$countSpeakers}{NREF} = 0 if(!exists($Overlap{$countSpeakers}{NREF}));
	$Overlap{$countSpeakers}{NREF} += $mySG->{NREF};
	
	$Overlap{$countSpeakers}{NCORR} = 0 if(!exists($Overlap{$countSpeakers}{NCORR}));
	$Overlap{$countSpeakers}{NCORR} += $mySG->{NCORR};
	$Overlap{$countSpeakers}{PCORR} = "NA";
	$Overlap{$countSpeakers}{PCORR} = sprintf("%.1f", 100*$Overlap{$countSpeakers}{NCORR}/$Overlap{$countSpeakers}{NREF}) if($Overlap{$countSpeakers}{NREF} != 0);
	
	$Overlap{$countSpeakers}{NSUB} = 0 if(!exists($Overlap{$countSpeakers}{NSUB}));
	$Overlap{$countSpeakers}{NSUB} += $mySG->{NSUB};
	$Overlap{$countSpeakers}{PSUB} = "NA";
	$Overlap{$countSpeakers}{PSUB} = sprintf("%.1f", 100*$Overlap{$countSpeakers}{NSUB}/$Overlap{$countSpeakers}{NREF}) if($Overlap{$countSpeakers}{NREF} != 0);
	
	$Overlap{$countSpeakers}{NINS} = 0 if(!exists($Overlap{$countSpeakers}{NINS}));
	$Overlap{$countSpeakers}{NINS} += $mySG->{NINS};	
	$Overlap{$countSpeakers}{PINS} = "NA";
	$Overlap{$countSpeakers}{PINS} = sprintf("%.1f", 100*$Overlap{$countSpeakers}{NINS}/$Overlap{$countSpeakers}{NREF}) if($Overlap{$countSpeakers}{NREF} != 0);
	
	$Overlap{$countSpeakers}{NDEL} = 0 if(!exists($Overlap{$countSpeakers}{NDEL}));
	$Overlap{$countSpeakers}{NDEL} += $mySG->{NDEL};
	$Overlap{$countSpeakers}{PDEL} = "NA";
	$Overlap{$countSpeakers}{PDEL} = sprintf("%.1f", 100*$Overlap{$countSpeakers}{NDEL}/$Overlap{$countSpeakers}{NREF}) if($Overlap{$countSpeakers}{NREF} != 0);
	
	$Overlap{$countSpeakers}{NSPKRERR} = 0 if(!exists($Overlap{$countSpeakers}{NSPKRERR}));
	$Overlap{$countSpeakers}{NSPKRERR} += $mySG->{NSPKRERR};
	$Overlap{$countSpeakers}{PSPKRERR} = "NA";
	$Overlap{$countSpeakers}{PSPKRERR} = sprintf("%.1f", 100*$Overlap{$countSpeakers}{NSPKRERR}/$Overlap{$countSpeakers}{NREF}) if($Overlap{$countSpeakers}{NREF} != 0);
	
	$Overlap{$countSpeakers}{NERR} = $Overlap{$countSpeakers}{NSUB} + $Overlap{$countSpeakers}{NINS} + $Overlap{$countSpeakers}{NDEL} + $Overlap{$countSpeakers}{NSPKRERR};
	$Overlap{$countSpeakers}{PERR} = "NA";
	$Overlap{$countSpeakers}{PERR} = sprintf("%.1f", 100*$Overlap{$countSpeakers}{NERR}/$Overlap{$countSpeakers}{NREF}) if($Overlap{$countSpeakers}{NREF} != 0);
	
	$Overlap{$countSpeakers}{TOTTIME} = 0 if(!exists($Overlap{$countSpeakers}{TOTTIME}));
	$Overlap{$countSpeakers}{TOTTIME} += $mySG->{ET} - $mySG->{BT};
	
	push( @{ $Overlap{$countSpeakers}{LISTSG} }, $SegGrpID);
	$Overlap{$countSpeakers}{NLISTSG} = scalar @{ $Overlap{$countSpeakers}{LISTSG} };
	
	$Overlap{$countSpeakers}{TOTTIMEALIGNED} = 0 if(!exists($Overlap{$countSpeakers}{TOTTIMEALIGNED}));
	$Overlap{$countSpeakers}{TOTTIMEALIGNED} += $mySG->{ET} - $mySG->{BT} if($mySG->{ALIGNED} == 1);

	push( @{ $Overlap{$countSpeakers}{LISTSGALIGNED} }, $SegGrpID) if($mySG->{ALIGNED} == 1);
	$Overlap{$countSpeakers}{NLISTSGALIGNED} = 0;
	$Overlap{$countSpeakers}{NLISTSGALIGNED} = scalar(@{ $Overlap{$countSpeakers}{LISTSGALIGNED} }) if(exists($Overlap{$countSpeakers}{LISTSGALIGNED}));
	
	$Overlap{$countSpeakers}{NUMHYPTOKENS} = 0 if(!exists($Overlap{$countSpeakers}{NUMHYPTOKENS}));
	$Overlap{$countSpeakers}{NUMHYPTOKENS} += $SegGroups{$SegGrpID}->GetNumHypWords();
}

open(FILEINDEX, ">$Outputdir/index.html") or die "$?";

print FILEINDEX "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">\n";
print FILEINDEX "<html>\n";
print FILEINDEX "<head>\n";
print FILEINDEX "<meta http-equiv=\"Content-Type\" content=\"text/html;charset=ISO-8859-1\">\n";
print FILEINDEX "<title>Overall Performance Summary by Meeting</title>\n";
print FILEINDEX "</head>\n";
print FILEINDEX "<body>\n";
print FILEINDEX "<center>\n";
print FILEINDEX "<h2>Overall Performance Summary by Meeting</h2>\n";
print FILEINDEX "<table style=\"text-align: left;\" border=\"1\" cellpadding=\"1\" cellspacing=\"0\">\n";
print FILEINDEX "<tbody>\n";
print FILEINDEX "<tr>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: left;\"><b>File\/Channel</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total<br>Time</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions<br>Aligned</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total Time<br>Aligned</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Aligned<br>Words</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Corr</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Sub</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ins</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Del</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>SpErr</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Err</b></td>\n";
print FILEINDEX "</tr>\n";

my $TnbrSG = 0;
my $Ttottime = 0;
my $TnbrSGAligned = 0;
my $Ttottimealigned = 0;
my $Tnumref = 0;
my $TpercCorr = 0;
my $TpercSub = 0;
my $TpercIns = 0;
my $TpercDel = 0;
my $TpercSpErr = 0;
my $TpercErr = 0;
my $TnumCorr = 0;
my $TnumSub = 0;
my $TnumIns = 0;
my $TnumDel = 0;
my $TnumSpErr = 0;
my $TnumErr = 0;

my $ccount = 0;

foreach my $file(sort keys %FileChannelSG)
{
	foreach my $channel(sort keys %{ $FileChannelSG{$file} })
	{
		my $cleanfile = $file;
		$cleanfile =~ s/\//\_/g;		
		my $filename = "$cleanfile" . "_" . "$channel" . ".html";
		
		my $percCorr = "NA";
		my $percSub = "NA";
		my $percIns = "NA";
		my $percDel = "NA";
		my $percErr = "NA";
		my $percSpErr = "NA";
		
		my $numref = $FileChannelSG{$file}{$channel}{NREF};
		$Tnumref += $numref;
		
		my $nbrSG = scalar(@{ $FileChannelSG{$file}{$channel}{LISTSG} });
		$TnbrSG += $nbrSG;
		
		my $tottime = sprintf("%.3f", $FileChannelSG{$file}{$channel}{TOTTIME});
		$Ttottime += $tottime;
		
		my $nbrSGAligned = scalar(@{ $FileChannelSG{$file}{$channel}{LISTSGALIGNED} });
        $TnbrSGAligned += $nbrSGAligned;
		
		my $tottimealigned = sprintf("%.3f", $FileChannelSG{$file}{$channel}{TOTTIMEALIGNED});
		$Ttottimealigned += $tottimealigned;
		
		if($FileChannelSG{$file}{$channel}{NREF} != 0)
		{
			$percCorr = sprintf("%.1f%% (%d)", 100*$FileChannelSG{$file}{$channel}{NCORR}/$FileChannelSG{$file}{$channel}{NREF}, $FileChannelSG{$file}{$channel}{NCORR});
			$percSub = sprintf("%.1f%% (%d)", 100*$FileChannelSG{$file}{$channel}{NSUB}/$FileChannelSG{$file}{$channel}{NREF}, $FileChannelSG{$file}{$channel}{NSUB});
			$percIns = sprintf("%.1f%% (%d)", 100*$FileChannelSG{$file}{$channel}{NINS}/$FileChannelSG{$file}{$channel}{NREF}, $FileChannelSG{$file}{$channel}{NINS});
			$percDel = sprintf("%.1f%% (%d)", 100*$FileChannelSG{$file}{$channel}{NDEL}/$FileChannelSG{$file}{$channel}{NREF}, $FileChannelSG{$file}{$channel}{NDEL});
			$percErr = sprintf("%.1f%% (%d)", 100*($FileChannelSG{$file}{$channel}{NDEL}+$FileChannelSG{$file}{$channel}{NINS}+$FileChannelSG{$file}{$channel}{NSUB}+$FileChannelSG{$file}{$channel}{NSPKRERR})/$FileChannelSG{$file}{$channel}{NREF}, $FileChannelSG{$file}{$channel}{NDEL}+$FileChannelSG{$file}{$channel}{NINS}+$FileChannelSG{$file}{$channel}{NSUB}+$FileChannelSG{$file}{$channel}{NSPKRERR});
			$percSpErr = sprintf("%.1f%% (%d)", 100*$FileChannelSG{$file}{$channel}{NSPKRERR}/$FileChannelSG{$file}{$channel}{NREF}, $FileChannelSG{$file}{$channel}{NSPKRERR});
			
			$TpercCorr += sprintf("%.3f", 100*$FileChannelSG{$file}{$channel}{NCORR}/$FileChannelSG{$file}{$channel}{NREF});
			$TpercSub += sprintf("%.3f", 100*$FileChannelSG{$file}{$channel}{NSUB}/$FileChannelSG{$file}{$channel}{NREF});
			$TpercIns += sprintf("%.3f", 100*$FileChannelSG{$file}{$channel}{NINS}/$FileChannelSG{$file}{$channel}{NREF});
			$TpercDel += sprintf("%.3f", 100*$FileChannelSG{$file}{$channel}{NDEL}/$FileChannelSG{$file}{$channel}{NREF});
			$TpercSpErr += sprintf("%.3f", 100*$FileChannelSG{$file}{$channel}{NSPKRERR}/$FileChannelSG{$file}{$channel}{NREF});
			$TpercErr += sprintf("%.3f", 100*($FileChannelSG{$file}{$channel}{NDEL}+$FileChannelSG{$file}{$channel}{NINS}+$FileChannelSG{$file}{$channel}{NSUB}+$FileChannelSG{$file}{$channel}{NSPKRERR})/$FileChannelSG{$file}{$channel}{NREF});
			$TnumCorr += $FileChannelSG{$file}{$channel}{NCORR};
			$TnumSub += $FileChannelSG{$file}{$channel}{NSUB};
			$TnumIns += $FileChannelSG{$file}{$channel}{NINS};
			$TnumDel += $FileChannelSG{$file}{$channel}{NDEL};
			$TnumSpErr += $FileChannelSG{$file}{$channel}{NSPKRERR};
			$TnumErr += $FileChannelSG{$file}{$channel}{NDEL}+$FileChannelSG{$file}{$channel}{NINS}+$FileChannelSG{$file}{$channel}{NSUB}+$FileChannelSG{$file}{$channel}{NSPKRERR};
			
			$ccount++;
		}
		
		print FILEINDEX "<tr>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: left;\"><a href=\"$filename\" target=\"_blank\">$file\/$channel</a></td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$nbrSG</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$tottime</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$nbrSGAligned</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$tottimealigned</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$numref</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$percCorr</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$percSub</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$percIns</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$percDel</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$percSpErr</td>\n";
		print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$percErr</td>\n";
		print FILEINDEX "</tr>\n";		
	}
}

my $dTpercCorr = sprintf("%.1f%% (%d)",  100*$TnumCorr/$Tnumref, $TnumCorr);
my $dTpercSub = sprintf("%.1f%% (%d)",   100*$TnumSub/$Tnumref, $TnumSub);
my $dTpercIns = sprintf("%.1f%% (%d)",   100*$TnumIns/$Tnumref, $TnumIns);
my $dTpercDel = sprintf("%.1f%% (%d)",   100*$TnumDel/$Tnumref, $TnumDel);
my $dTpercSpErr = sprintf("%.1f%% (%d)", 100*$TnumSpErr/$Tnumref, $TnumSpErr);
my $dTpercErr = sprintf("%.1f%% (%d)",   100*$TnumErr/$Tnumref, $TnumErr);

print FILEINDEX "<tr>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: left;\"><b>Ensemble</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$TnbrSG</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$Ttottime</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$TnbrSGAligned</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$Ttottimealigned</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$Tnumref</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$dTpercCorr</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$dTpercSub</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$dTpercIns</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$dTpercDel</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$dTpercSpErr</td>\n";
print FILEINDEX "<td style=\"vertical-align: top; text-align: center;\">$dTpercErr</td>\n";
print FILEINDEX "</tr>\n";
print FILEINDEX "</tbody>\n";
print FILEINDEX "</table>\n";

print FILEINDEX "<br>\n";

my $Total_regs = 0;
my $Total_tottime = 0;
my $Total_regsalign = 0;
my $Total_tottimealign = 0;
my $Total_nref = 0;
my $Total_ncorr = 0;
my $Total_nsub = 0;
my $Total_nins = 0;
my $Total_ndel = 0;
my $Total_nspkerr = 0;
my $Total_nerr = 0;

print FILEINDEX "<table style=\"text-align: left;\" border=\"1\" cellpadding=\"1\" cellspacing=\"0\">\n";
print FILEINDEX "<tbody>\n";
print FILEINDEX "<tr>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Speaker<br>Overlap</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total<br>Time</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions<br>Aligned</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total Time<br>Aligned</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Aligned<br>Words</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Corr</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Sub</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ins</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Del</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>SpErr</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Err</b></td>\n";
print FILEINDEX "</tr>\n";

foreach my $spkover (sort {$a <=> $b} keys %Overlap)
{
	my $display_regs = $Overlap{$spkover}{NLISTSG};
	my $display_tottime = sprintf("%.3f", $Overlap{$spkover}{TOTTIME});
	my $display_regsalign = $Overlap{$spkover}{NLISTSGALIGNED};
	my $display_tottimealign = sprintf("%.3f", $Overlap{$spkover}{TOTTIMEALIGNED});
	my $display_nref = "-";
	my $display_ncorr = "-";
	my $display_nsub = "-";
	my $display_nins = "-";
	my $display_ndel = "-";
	my $display_nspkerr = "-";
	my $display_nerr = "-";
	
	$Total_regs += $Overlap{$spkover}{NLISTSG};
	$Total_tottime += $Overlap{$spkover}{TOTTIME};
	$Total_regsalign += $Overlap{$spkover}{NLISTSGALIGNED};
	$Total_tottimealign += $Overlap{$spkover}{TOTTIMEALIGNED};
	
	if($spkover == 0)
	{
		$display_nins = sprintf("(%d)", $Overlap{$spkover}{NUMHYPTOKENS});
		
		$Total_nins += $Overlap{$spkover}{NUMHYPTOKENS};
		$Total_nerr += $Total_nins;
	}
	
	if($Overlap{$spkover}{NREF} != 0)
	{
		$display_nref = $Overlap{$spkover}{NREF};
		$display_ncorr = sprintf("%.1f%% (%d)", $Overlap{$spkover}{PCORR}, $Overlap{$spkover}{NCORR});
		$display_nsub = sprintf("%.1f%% (%d)", $Overlap{$spkover}{PSUB}, $Overlap{$spkover}{NSUB});
		$display_nins = sprintf("%.1f%% (%d)", $Overlap{$spkover}{PINS}, $Overlap{$spkover}{NINS});
		$display_ndel = sprintf("%.1f%% (%d)", $Overlap{$spkover}{PDEL}, $Overlap{$spkover}{NDEL});
		$display_nspkerr = sprintf("%.1f%% (%d)", $Overlap{$spkover}{PSPKRERR}, $Overlap{$spkover}{NSPKRERR});
		$display_nerr = sprintf("%.1f%% (%d)", $Overlap{$spkover}{PERR}, $Overlap{$spkover}{NERR});
		
		$Total_nref += $Overlap{$spkover}{NREF};
		$Total_ncorr += $Overlap{$spkover}{NCORR};
		$Total_nsub += $Overlap{$spkover}{NSUB};
		$Total_nins += $Overlap{$spkover}{NINS};
		$Total_ndel += $Overlap{$spkover}{NDEL};
		$Total_nspkerr += $Overlap{$spkover}{NSPKRERR};
		$Total_nerr += $Overlap{$spkover}{NERR};
	}
	
	print FILEINDEX "<tr>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>$spkover<b></td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regs</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottime</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regsalign</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottimealign</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nref</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ncorr</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nsub</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nins</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ndel</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nspkerr</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nerr</td>\n";
	print FILEINDEX "</tr>\n";
	
	# data for Cumuloverlap table
	$CumulOverlap{$spkover}{NLISTSG} = $Total_regs;
	$CumulOverlap{$spkover}{TOTTIME} = $Total_tottime;
	$CumulOverlap{$spkover}{NLISTSGALIGNED} = $Total_regsalign;
	$CumulOverlap{$spkover}{TOTTIMEALIGNED} = $Total_tottimealign;
	$CumulOverlap{$spkover}{NREF} = $Total_nref;
	$CumulOverlap{$spkover}{NCORR} = $Total_ncorr;
	$CumulOverlap{$spkover}{NSUB} = $Total_nsub;
	$CumulOverlap{$spkover}{NINS} = $Total_nins;
	$CumulOverlap{$spkover}{NDEL} = $Total_ndel;
	$CumulOverlap{$spkover}{NSPKRERR} = $Total_nspkerr;
	$CumulOverlap{$spkover}{NERR} = $Total_nerr;
}

my $display_Total_regs = $Total_regs;
my $display_Total_tottime = sprintf("%.3f", $Total_tottime);
my $display_Total_regsalign = $Total_regsalign;
my $display_Total_tottimealign = sprintf("%.3f", $Total_tottimealign);
my $display_Total_nref = $Total_nref;
my $display_Total_ncorr = sprintf("%.1f%% (%d)", 100*$Total_ncorr/$Total_nref, $Total_ncorr);
my $display_Total_nsub = sprintf("%.1f%% (%d)", 100*$Total_nsub/$Total_nref, $Total_nsub);
my $display_Total_nins = sprintf("%.1f%% (%d)", 100*$Total_nins/$Total_nref, $Total_nins);
my $display_Total_ndel = sprintf("%.1f%% (%d)", 100*$Total_ndel/$Total_nref, $Total_ndel);
my $display_Total_nspkerr = sprintf("%.1f%% (%d)", 100*$Total_nspkerr/$Total_nref, $Total_nspkerr);
my $display_Total_nerr = sprintf("%.1f%% (%d)", 100*$Total_nerr/$Total_nref, $Total_nerr);

print FILEINDEX "<tr>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ensemble</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_regs</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_tottime</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_regsalign</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_tottimealign</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nref</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_ncorr</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nsub</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nins</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_ndel</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nspkerr</td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nerr</td>\n";
print FILEINDEX "</tr>\n";

print FILEINDEX "</tbody>\n";
print FILEINDEX "</table>\n";

print FILEINDEX "<br>\n";

print FILEINDEX "<table style=\"text-align: left;\" border=\"1\" cellpadding=\"1\" cellspacing=\"0\">\n";
print FILEINDEX "<tbody>\n";
print FILEINDEX "<tr>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Speaker<br>Cumulative<br>Overlap</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total<br>Time</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions<br>Aligned</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total Time<br>Aligned</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Aligned<br>Words</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Corr</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Sub</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ins</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Del</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>SpErr</b></td>\n";
print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Err</b></td>\n";
print FILEINDEX "</tr>\n";

foreach my $cumulspkover (sort {$a <=> $b} keys %CumulOverlap)
{
	my $display_regs = $CumulOverlap{$cumulspkover}{NLISTSG};
	my $display_tottime = sprintf("%.3f", $CumulOverlap{$cumulspkover}{TOTTIME});
	my $display_regsalign = $CumulOverlap{$cumulspkover}{NLISTSGALIGNED};
	my $display_tottimealign = sprintf("%.3f", $CumulOverlap{$cumulspkover}{TOTTIMEALIGNED});
	my $display_nref = "-";
	my $display_ncorr = "-";
	my $display_nsub = "-";
	my $display_nins = "-";
	my $display_ndel = "-";
	my $display_nspkerr = "-";
	my $display_nerr = "-";
	
	if($cumulspkover == 0)
	{
		$display_nins = sprintf("(%d)", $CumulOverlap{$cumulspkover}{NINS});
	}
	
	if($CumulOverlap{$cumulspkover}{NREF} != 0)
	{
		$display_nref = $CumulOverlap{$cumulspkover}{NREF};
		$display_ncorr = sprintf("%.1f%% (%d)", $CumulOverlap{$cumulspkover}{NCORR}     *100/$CumulOverlap{$cumulspkover}{NREF}, $CumulOverlap{$cumulspkover}{NCORR});
		$display_nsub = sprintf("%.1f%% (%d)", $CumulOverlap{$cumulspkover}{NSUB}       *100/$CumulOverlap{$cumulspkover}{NREF}, $CumulOverlap{$cumulspkover}{NSUB});
		$display_nins = sprintf("%.1f%% (%d)", $CumulOverlap{$cumulspkover}{NINS}       *100/$CumulOverlap{$cumulspkover}{NREF}, $CumulOverlap{$cumulspkover}{NINS});
		$display_ndel = sprintf("%.1f%% (%d)", $CumulOverlap{$cumulspkover}{NDEL}       *100/$CumulOverlap{$cumulspkover}{NREF}, $CumulOverlap{$cumulspkover}{NDEL});
		$display_nspkerr = sprintf("%.1f%% (%d)", $CumulOverlap{$cumulspkover}{NSPKRERR}*100/$CumulOverlap{$cumulspkover}{NREF}, $CumulOverlap{$cumulspkover}{NSPKRERR});
		$display_nerr = sprintf("%.1f%% (%d)", $CumulOverlap{$cumulspkover}{NERR}       *100/$CumulOverlap{$cumulspkover}{NREF}, $CumulOverlap{$cumulspkover}{NERR});
	}

	print FILEINDEX "<tr>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>$cumulspkover<b></td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regs</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottime</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regsalign</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottimealign</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nref</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ncorr</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nsub</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nins</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ndel</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nspkerr</td>\n";
	print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nerr</td>\n";
	print FILEINDEX "</tr>\n";
}


print FILEINDEX "</tbody>\n";
print FILEINDEX "</table>\n";


print FILEINDEX "</center>\n";
print FILEINDEX "</body>\n";
print FILEINDEX "</html>\n";

close FILEINDEX;

foreach my $file(sort keys %FileChannelSG)
{
	foreach my $channel(sort keys %{ $FileChannelSG{$file} })
	{
		my %overallmeeting;
		my %cumuloverallmeeting;
		my $cleanfile = $file;
		$cleanfile =~ s/\//\_/g;		
		my $filename = "$cleanfile" . "_" . "$channel" . ".html";
		my $bottom = "";
	
		open(FILEINDEX, ">$Outputdir/$filename") or die "$?";

		print FILEINDEX "<!DOCTYPE html PUBLIC \"-//W3C//DTD HTML 4.01 Transitional//EN\">\n";
		print FILEINDEX "<html>\n";
		print FILEINDEX "<head>\n";
		print FILEINDEX "<meta http-equiv=\"Content-Type\" content=\"text/html;charset=ISO-8859-1\">\n";
		print FILEINDEX "<title>Overall Performance Summary for Meeting $file\/$channel</title>\n";
		print FILEINDEX "</head>\n";
		print FILEINDEX "<body>\n";
		
		print FILEINDEX "<center>\n";
		print FILEINDEX "<h2>Overall Performance Summary for Meeting $file\/$channel</h2>\n";
		print FILEINDEX "<table style=\"text-align: left;\" border=\"1\" cellpadding=\"1\" cellspacing=\"0\">\n";
		print FILEINDEX "<tbody>\n";
		
		print FILEINDEX "<tr>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: left;\"><b>Region ID</b></b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Begin<br>Time</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>End<br>Time</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Duration</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref<br>Speaker</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Hyp<br>Speaker</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Aligned<br>Words</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Corr</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Sub</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ins</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Del</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>SpErr</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Err</b></td>\n";
		print FILEINDEX "</tr>\n";
			
		foreach my $SG_ID (sort {$a <=> $b} @{ $FileChannelSG{$file}{$channel}{LISTSG} })
		{
			my $BT = sprintf("%.3f", $SegGroups{$SG_ID}->{BT});
			my $ET = sprintf("%.3f", $SegGroups{$SG_ID}->{ET});
			my $Duration = sprintf("%.3f", $SegGroups{$SG_ID}->{ET} - $SegGroups{$SG_ID}->{BT});
			
			my @refspkrs;
			my @sysspkrs;
			
			foreach my $segref (keys %{ $SegGroups{$SG_ID}->{REF} }) { push(@refspkrs, $SegGroups{$SG_ID}->{REF}{$segref}->{SPKRID}) if($SegGroups{$SG_ID}->{REF}{$segref}->{SPKRID} ne "ref:INTER_SEGMENT_GAP"); }
			foreach my $segsys (keys %{ $SegGroups{$SG_ID}->{SYS} }) { push(@sysspkrs, $SegGroups{$SG_ID}->{SYS}{$segsys}->{SPKRID}) if(!($SegGroups{$SG_ID}->{SYS}{$segsys}->HasOnlyOneFakeToken())); }
			
			my $nrefspkr = scalar unique @refspkrs;
			my $nhypspkr = scalar unique @sysspkrs;
			
			$SegGroups{$SG_ID}->{ALIGNED} = 1 if( ($nrefspkr == 0) && ($nhypspkr == 0) );
			$SegGroups{$SG_ID}->{ALIGNED} = 1 if( ($SegGroups{$SG_ID}->{ALIGNED} == 0) && ($nrefspkr == 1) && ($nhypspkr == 0) && ($SegGroups{$SG_ID}->{HASFAKETIME} == 1) );
			
			my $titleSG = "<a href=\"segmentgroup-$SG_ID.html\" target=\"_blank\">$SG_ID</a>";
			my $nref = "-";
			my $percCorr = "-";
			my $percSub = "-";
			my $percIns = "-";
			my $percDel = "-";
			my $percErr = "-";
			my $percSpErr = "-";
			
			if($SegGroups{$SG_ID}->{NREF} != 0)
			{
				$percCorr = sprintf("%.1f%% (%d)", 100*$SegGroups{$SG_ID}->{NCORR}/$SegGroups{$SG_ID}->{NREF}, $SegGroups{$SG_ID}->{NCORR});
				$percSub = sprintf("%.1f%% (%d)", 100*$SegGroups{$SG_ID}->{NSUB}/$SegGroups{$SG_ID}->{NREF}, $SegGroups{$SG_ID}->{NSUB});
				$percIns = sprintf("%.1f%% (%d)", 100*$SegGroups{$SG_ID}->{NINS}/$SegGroups{$SG_ID}->{NREF}, $SegGroups{$SG_ID}->{NINS});
				$percDel = sprintf("%.1f%% (%d)", 100*$SegGroups{$SG_ID}->{NDEL}/$SegGroups{$SG_ID}->{NREF}, $SegGroups{$SG_ID}->{NDEL});
				$percErr = sprintf("%.1f%% (%d)", 100*($SegGroups{$SG_ID}->{NDEL}+$SegGroups{$SG_ID}->{NINS}+$SegGroups{$SG_ID}->{NSUB}+$SegGroups{$SG_ID}->{NSPKRERR})/$SegGroups{$SG_ID}->{NREF}, $SegGroups{$SG_ID}->{NDEL}+$SegGroups{$SG_ID}->{NINS}+$SegGroups{$SG_ID}->{NSUB}+$SegGroups{$SG_ID}->{NSPKRERR});
				$percSpErr = sprintf("%.1f%% (%d)", 100*$SegGroups{$SG_ID}->{NSPKRERR}/$SegGroups{$SG_ID}->{NREF}, $SegGroups{$SG_ID}->{NSPKRERR});
			}
			else
			{
                $percIns = sprintf("(%d)", $SegGroups{$SG_ID}->{NINS}) if($SegGroups{$SG_ID}->{ALIGNED} == 1);
			}
						
            if($SegGroups{$SG_ID}->{ALIGNED} == 0)
			{
				$titleSG = "*<i>$titleSG</i>";
				$nrefspkr = "<i>$nrefspkr</i>";
				$nhypspkr = "<i>$nhypspkr</i>";
				$BT = "<i>$BT</i>";
				$ET = "<i>$ET</i>";
				$Duration = "<i>$Duration</i>";
				$bottom = "<br>*: Segment Group ignored for scoring.<br>";
			}
			else
			{
                $nref = $SegGroups{$SG_ID}->{NREF};
			}
					
			print FILEINDEX "<tr>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: left;\">$titleSG</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$BT</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$ET</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$Duration</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$nrefspkr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$nhypspkr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$nref</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$percCorr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$percSub</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$percIns</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$percDel</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$percSpErr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$percErr</td>\n";
			print FILEINDEX "</tr>\n";
			
			$BT =~ s/<i>//;
			$ET =~ s/<i>//;
			$nrefspkr =~ s/<i>//;
			$nhypspkr =~ s/<i>//;
			$BT =~ s/<\/i>//;
			$ET =~ s/<\/i>//;
			$nrefspkr =~ s/<\/i>//;
			$nhypspkr =~ s/<\/i>//;
			
			my $countSpeakers = $nrefspkr;
			
			$overallmeeting{$countSpeakers}{NREF} = 0 if(!exists($overallmeeting{$countSpeakers}{NREF}));
			$overallmeeting{$countSpeakers}{NREF} += $SegGroups{$SG_ID}->{NREF};
			
			$overallmeeting{$countSpeakers}{NCORR} = 0 if(!exists($overallmeeting{$countSpeakers}{NCORR}));
			$overallmeeting{$countSpeakers}{NCORR} += $SegGroups{$SG_ID}->{NCORR};
			$overallmeeting{$countSpeakers}{PCORR} = "NA";
			$overallmeeting{$countSpeakers}{PCORR} = sprintf("%.1f", 100*$overallmeeting{$countSpeakers}{NCORR}/$overallmeeting{$countSpeakers}{NREF}) if($overallmeeting{$countSpeakers}{NREF} != 0);
			
			$overallmeeting{$countSpeakers}{NSUB} = 0 if(!exists($overallmeeting{$countSpeakers}{NSUB}));
			$overallmeeting{$countSpeakers}{NSUB} += $SegGroups{$SG_ID}->{NSUB};
			$overallmeeting{$countSpeakers}{PSUB} = "NA";
			$overallmeeting{$countSpeakers}{PSUB} = sprintf("%.1f", 100*$overallmeeting{$countSpeakers}{NSUB}/$overallmeeting{$countSpeakers}{NREF}) if($overallmeeting{$countSpeakers}{NREF} != 0);
			
			$overallmeeting{$countSpeakers}{NINS} = 0 if(!exists($overallmeeting{$countSpeakers}{NINS}));
			$overallmeeting{$countSpeakers}{NINS} += $SegGroups{$SG_ID}->{NINS};
			$overallmeeting{$countSpeakers}{PINS} = "NA";
			$overallmeeting{$countSpeakers}{PINS} = sprintf("%.1f", 100*$overallmeeting{$countSpeakers}{NINS}/$overallmeeting{$countSpeakers}{NREF}) if($overallmeeting{$countSpeakers}{NREF} != 0);
			
			$overallmeeting{$countSpeakers}{NDEL} = 0 if(!exists($overallmeeting{$countSpeakers}{NDEL}));
			$overallmeeting{$countSpeakers}{NDEL} += $SegGroups{$SG_ID}->{NDEL};
			$overallmeeting{$countSpeakers}{PDEL} = "NA";
			$overallmeeting{$countSpeakers}{PDEL} = sprintf("%.1f", 100*$overallmeeting{$countSpeakers}{NDEL}/$overallmeeting{$countSpeakers}{NREF}) if($overallmeeting{$countSpeakers}{NREF} != 0);
			
			$overallmeeting{$countSpeakers}{NSPKRERR} = 0 if(!exists($overallmeeting{$countSpeakers}{NSPKRERR}));
			$overallmeeting{$countSpeakers}{NSPKRERR} += $SegGroups{$SG_ID}->{NSPKRERR};
			$overallmeeting{$countSpeakers}{PSPKRERR} = "NA";
			$overallmeeting{$countSpeakers}{PSPKRERR} = sprintf("%.1f", 100*$overallmeeting{$countSpeakers}{NSPKRERR}/$overallmeeting{$countSpeakers}{NREF}) if($overallmeeting{$countSpeakers}{NREF} != 0);
			
			$overallmeeting{$countSpeakers}{NERR} = $overallmeeting{$countSpeakers}{NSUB} + $overallmeeting{$countSpeakers}{NINS} + $overallmeeting{$countSpeakers}{NDEL} + $overallmeeting{$countSpeakers}{NSPKRERR};
			$overallmeeting{$countSpeakers}{PERR} = "NA";
			$overallmeeting{$countSpeakers}{PERR} = sprintf("%.1f", 100*$overallmeeting{$countSpeakers}{NERR}/$overallmeeting{$countSpeakers}{NREF}) if($overallmeeting{$countSpeakers}{NREF} != 0);
			
			$overallmeeting{$countSpeakers}{TOTTIME} = 0 if(!exists($overallmeeting{$countSpeakers}{TOTTIME}));
			$overallmeeting{$countSpeakers}{TOTTIME} += $SegGroups{$SG_ID}->{ET} - $SegGroups{$SG_ID}->{BT};
			
			push( @{ $overallmeeting{$countSpeakers}{LISTSG} }, $SG_ID);
			$overallmeeting{$countSpeakers}{NLISTSG} = scalar @{ $overallmeeting{$countSpeakers}{LISTSG} };
			
			$overallmeeting{$countSpeakers}{TOTTIMEALIGNED} = 0 if(!exists($overallmeeting{$countSpeakers}{TOTTIMEALIGNED}));
			$overallmeeting{$countSpeakers}{TOTTIMEALIGNED} += $SegGroups{$SG_ID}->{ET} - $SegGroups{$SG_ID}->{BT} if($SegGroups{$SG_ID}->{ALIGNED} == 1);
		
			push( @{ $overallmeeting{$countSpeakers}{LISTSGALIGNED} }, $SG_ID) if($SegGroups{$SG_ID}->{ALIGNED} == 1);
			$overallmeeting{$countSpeakers}{NLISTSGALIGNED} = 0;
			$overallmeeting{$countSpeakers}{NLISTSGALIGNED} = scalar(@{ $overallmeeting{$countSpeakers}{LISTSGALIGNED} }) if(exists($overallmeeting{$countSpeakers}{LISTSGALIGNED}));
			
			$overallmeeting{$countSpeakers}{NUMHYPTOKENS} = 0 if(!exists($overallmeeting{$countSpeakers}{NUMHYPTOKENS}));
			$overallmeeting{$countSpeakers}{NUMHYPTOKENS} += $SegGroups{$SG_ID}->GetNumHypWords();
		}
		
		print FILEINDEX "</tbody>\n";
		print FILEINDEX "</table>\n";
		print FILEINDEX "$bottom";
		print FILEINDEX "<br>\n";
		
		my $Total_regs = 0;
		my $Total_tottime = 0;
		my $Total_regsalign = 0;
		my $Total_tottimealign = 0;
		my $Total_nref = 0;
		my $Total_ncorr = 0;
		my $Total_nsub = 0;
		my $Total_nins = 0;
		my $Total_ndel = 0;
		my $Total_nspkerr = 0;
		my $Total_nerr = 0;
		
		print FILEINDEX "<table style=\"text-align: left;\" border=\"1\" cellpadding=\"1\" cellspacing=\"0\">\n";
		print FILEINDEX "<tbody>\n";
		print FILEINDEX "<tr>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Speaker<br>Overlap</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total<br>Time</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions<br>Aligned</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total Time<br>Aligned</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Aligned<br>Words</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Corr</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Sub</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ins</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Del</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>SpErr</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Err</b></td>\n";
		print FILEINDEX "</tr>\n";
		
		foreach my $spkover (sort {$a <=> $b} keys %overallmeeting)
		{
			my $display_regs = $overallmeeting{$spkover}{NLISTSG};
			my $display_tottime = sprintf("%.3f", $overallmeeting{$spkover}{TOTTIME});
			my $display_regsalign = $overallmeeting{$spkover}{NLISTSGALIGNED};
			my $display_tottimealign = sprintf("%.3f", $overallmeeting{$spkover}{TOTTIMEALIGNED});
			my $display_nref = "-";
			my $display_ncorr = "-";
			my $display_nsub = "-";
			my $display_nins = "-";
			my $display_ndel = "-";
			my $display_nspkerr = "-";
			my $display_nerr = "-";
			
			$Total_regs += $overallmeeting{$spkover}{NLISTSG};
			$Total_tottime += $overallmeeting{$spkover}{TOTTIME};
			$Total_regsalign += $overallmeeting{$spkover}{NLISTSGALIGNED};
			$Total_tottimealign += $overallmeeting{$spkover}{TOTTIMEALIGNED};
			
			if($spkover == 0)
			{
				$display_nins = sprintf("(%d)", $overallmeeting{$spkover}{NUMHYPTOKENS});
				$Total_nins += $overallmeeting{$spkover}{NUMHYPTOKENS};
				$Total_nerr += $Total_nins;
			}
			
			if($overallmeeting{$spkover}{NREF} != 0)
			{
				$display_nref = $overallmeeting{$spkover}{NREF};
				$display_ncorr = sprintf("%.1f%% (%d)", $overallmeeting{$spkover}{PCORR}, $overallmeeting{$spkover}{NCORR});
				$display_nsub = sprintf("%.1f%% (%d)", $overallmeeting{$spkover}{PSUB}, $overallmeeting{$spkover}{NSUB});
				$display_nins = sprintf("%.1f%% (%d)", $overallmeeting{$spkover}{PINS}, $overallmeeting{$spkover}{NINS});
				$display_ndel = sprintf("%.1f%% (%d)", $overallmeeting{$spkover}{PDEL}, $overallmeeting{$spkover}{NDEL});
				$display_nspkerr = sprintf("%.1f%% (%d)", $overallmeeting{$spkover}{PSPKRERR}, $overallmeeting{$spkover}{NSPKRERR});
				$display_nerr = sprintf("%.1f%% (%d)", $overallmeeting{$spkover}{PERR}, $overallmeeting{$spkover}{NERR});
				$Total_nref += $overallmeeting{$spkover}{NREF};
				$Total_ncorr += $overallmeeting{$spkover}{NCORR};
				$Total_nsub += $overallmeeting{$spkover}{NSUB};
				$Total_nins += $overallmeeting{$spkover}{NINS};
				$Total_ndel += $overallmeeting{$spkover}{NDEL};
				$Total_nspkerr += $overallmeeting{$spkover}{NSPKRERR};
				$Total_nerr += $overallmeeting{$spkover}{NERR};
			}
			
			print FILEINDEX "<tr>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>$spkover<b></td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regs</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottime</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regsalign</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottimealign</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nref</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ncorr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nsub</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nins</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ndel</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nspkerr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nerr</td>\n";
			print FILEINDEX "</tr>\n";
			
			# data for Cumuloverlap table
			$cumuloverallmeeting{$spkover}{NLISTSG} = $Total_regs;
			$cumuloverallmeeting{$spkover}{TOTTIME} = $Total_tottime;
			$cumuloverallmeeting{$spkover}{NLISTSGALIGNED} = $Total_regsalign;
			$cumuloverallmeeting{$spkover}{TOTTIMEALIGNED} = $Total_tottimealign;
			$cumuloverallmeeting{$spkover}{NREF} = $Total_nref;
			$cumuloverallmeeting{$spkover}{NCORR} = $Total_ncorr;
			$cumuloverallmeeting{$spkover}{NSUB} = $Total_nsub;
			$cumuloverallmeeting{$spkover}{NINS} = $Total_nins;
			$cumuloverallmeeting{$spkover}{NDEL} = $Total_ndel;
			$cumuloverallmeeting{$spkover}{NSPKRERR} = $Total_nspkerr;
			$cumuloverallmeeting{$spkover}{NERR} = $Total_nerr;
		}
		
		my $display_Total_regs = $Total_regs;
		my $display_Total_tottime = sprintf("%.3f", $Total_tottime);
		my $display_Total_regsalign = $Total_regsalign;
		my $display_Total_tottimealign = sprintf("%.3f", $Total_tottimealign);
		my $display_Total_nref = $Total_nref;
		my $display_Total_ncorr = sprintf("%.1f%% (%d)", 100*$Total_ncorr/$Total_nref, $Total_ncorr);
		my $display_Total_nsub = sprintf("%.1f%% (%d)", 100*$Total_nsub/$Total_nref, $Total_nsub);
		my $display_Total_nins = sprintf("%.1f%% (%d)", 100*$Total_nins/$Total_nref, $Total_nins);
		my $display_Total_ndel = sprintf("%.1f%% (%d)", 100*$Total_ndel/$Total_nref, $Total_ndel);
		my $display_Total_nspkerr = sprintf("%.1f%% (%d)", 100*$Total_nspkerr/$Total_nref, $Total_nspkerr);
		my $display_Total_nerr = sprintf("%.1f%% (%d)", 100*$Total_nerr/$Total_nref, $Total_nerr);
		
		print FILEINDEX "<tr>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ensemble</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_regs</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_tottime</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_regsalign</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_tottimealign</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nref</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_ncorr</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nsub</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nins</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_ndel</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nspkerr</td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_Total_nerr</td>\n";
		print FILEINDEX "</tr>\n";
		
		print FILEINDEX "</tbody>\n";
		print FILEINDEX "</table>\n";
		
		print FILEINDEX "<br>\n";

		print FILEINDEX "<table style=\"text-align: left;\" border=\"1\" cellpadding=\"1\" cellspacing=\"0\">\n";
		print FILEINDEX "<tbody>\n";
		print FILEINDEX "<tr>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Speaker<br>Cumulative<br>Overlap</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total<br>Time</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Regions<br>Aligned</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Total Time<br>Aligned</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>\#Ref Aligned<br>Words</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Corr</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Sub</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Ins</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Del</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>SpErr</b></td>\n";
		print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>Err</b></td>\n";
		print FILEINDEX "</tr>\n";
		
		foreach my $cumulspkover (sort {$a <=> $b} keys %cumuloverallmeeting)
		{
			my $display_regs = $cumuloverallmeeting{$cumulspkover}{NLISTSG};
			my $display_tottime = sprintf("%.3f", $cumuloverallmeeting{$cumulspkover}{TOTTIME});
			my $display_regsalign = $cumuloverallmeeting{$cumulspkover}{NLISTSGALIGNED};
			my $display_tottimealign = sprintf("%.3f", $cumuloverallmeeting{$cumulspkover}{TOTTIMEALIGNED});
			my $display_nref = "-";
			my $display_ncorr = "-";
			my $display_nsub = "-";
			my $display_nins = "-";
			my $display_ndel = "-";
			my $display_nspkerr = "-";
			my $display_nerr = "-";
			
			if($cumulspkover == 0)
			{
				$display_nins = sprintf("(%d)", $CumulOverlap{$cumulspkover}{NINS});
			}
			
			if($cumuloverallmeeting{$cumulspkover}{NREF} != 0)
			{
				$display_nref = $cumuloverallmeeting{$cumulspkover}{NREF};
				$display_ncorr = sprintf("%.1f%% (%d)", $cumuloverallmeeting{$cumulspkover}{NCORR}     *100/$cumuloverallmeeting{$cumulspkover}{NREF}, $cumuloverallmeeting{$cumulspkover}{NCORR});
				$display_nsub = sprintf("%.1f%% (%d)", $cumuloverallmeeting{$cumulspkover}{NSUB}       *100/$cumuloverallmeeting{$cumulspkover}{NREF}, $cumuloverallmeeting{$cumulspkover}{NSUB});
				$display_nins = sprintf("%.1f%% (%d)", $cumuloverallmeeting{$cumulspkover}{NINS}       *100/$cumuloverallmeeting{$cumulspkover}{NREF}, $cumuloverallmeeting{$cumulspkover}{NINS});
				$display_ndel = sprintf("%.1f%% (%d)", $cumuloverallmeeting{$cumulspkover}{NDEL}       *100/$cumuloverallmeeting{$cumulspkover}{NREF}, $cumuloverallmeeting{$cumulspkover}{NDEL});
				$display_nspkerr = sprintf("%.1f%% (%d)", $cumuloverallmeeting{$cumulspkover}{NSPKRERR}*100/$cumuloverallmeeting{$cumulspkover}{NREF}, $cumuloverallmeeting{$cumulspkover}{NSPKRERR});
				$display_nerr = sprintf("%.1f%% (%d)", $cumuloverallmeeting{$cumulspkover}{NERR}       *100/$cumuloverallmeeting{$cumulspkover}{NREF}, $cumuloverallmeeting{$cumulspkover}{NERR});
			}
		
			print FILEINDEX "<tr>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\"><b>$cumulspkover<b></td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regs</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottime</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_regsalign</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_tottimealign</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nref</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ncorr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nsub</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nins</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_ndel</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nspkerr</td>\n";
			print FILEINDEX "<td style=\"vertical-align: center; text-align: center;\">$display_nerr</td>\n";
			print FILEINDEX "</tr>\n";
		}
		
		
		print FILEINDEX "</tbody>\n";
		print FILEINDEX "</table>\n";


		print FILEINDEX "<center>\n";
		print FILEINDEX "</body>\n";
		print FILEINDEX "</html>\n";
		
		close FILEINDEX;
	}
}

if (! $Installed)
{
    system ('cp', '-r', 'js', $Outputdir);
    system ('cp', '-r', 'images', $Outputdir);
}
else
{
    extractJavaImages($Outputdir);
}

1;

