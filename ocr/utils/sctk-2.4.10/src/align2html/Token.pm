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

package Token;
use strict;
use Data::Dumper;

sub new
{
    my $class = shift;
    my $self = {};

	$self->{ID} = shift;
	$self->{BT} = shift;
	$self->{DUR} = shift;
	$self->{MID} = 0;
	$self->{TEXT} = shift;
	$self->{LENGTH} = 0;
	$self->{XSTARTPOS} = 0;
	$self->{XENDPOS} = 0;
	$self->{YPOS} = 0;
	$self->{XMIDPOS} = 0;
	$self->{TEXTPOS} = "top";
	$self->{WIDTHLINE} = 1;
	$self->{PIXPERSEC} = 100;
	$self->{REFORSYS} = shift;
	$self->{PREVTKNID} = [()];
	$self->{NEXTTKNID} = [()];
	$self->{DISPLAY} = 0;
	$self->{NUMPREV} = 0;
	$self->{NUMNEXT} = 0;
	$self->{SEGGRPID} = shift;
	$self->{SEGID} = shift;
	$self->{SPKRID} = shift;
	$self->{CONF} = shift;
	$self->{REALTIME} = 1;	
	
	my $tmpprev = shift;
	my $tmpnext = shift;
	
	$self->{SEGBT} = shift;
	$self->{SEGET} = shift;
	
	bless $self;
	
	$self->{REALTIME} = 0 if( ($self->{BT} eq "") || ($self->{DUR} eq "") );
	
	if($self->{REALTIME} == 0)
	{
		$self->{BT} = $self->{SEGBT};
		$self->{DUR} = sprintf("%.3f", $self->{SEGET}-$self->{SEGBT});
	}
	
    $self->{MID} = sprintf("%.0f", $self->{BT} + $self->{DUR}/2);
    $self->{XSTARTPOS} = sprintf("%.0f", $self->{PIXPERSEC}*$self->{BT});
	$self->{XMIDPOS} = sprintf("%.0f", $self->{PIXPERSEC}*($self->{BT}+$self->{DUR}/2));
	$self->{XENDPOS} = sprintf("%.0f", $self->{PIXPERSEC}*($self->{BT}+$self->{DUR}));
	$self->{LENGTH} = sprintf("%.0f", $self->{PIXPERSEC}*$self->{DUR});
	$self->{TEXTPOS} = "bottom" if($self->{REFORSYS} eq "SYS");
	
	push(@{ $self->{PREVTKNID} }, sort {$a <=> $b} split(/\|/, $tmpprev) );
	push(@{ $self->{NEXTTKNID} }, sort {$a <=> $b} split(/\|/, $tmpnext) );
	
	$self->{NUMPREV} = scalar( @{ $self->{PREVTKNID} } );
	$self->{NUMNEXT} = scalar( @{ $self->{NEXTTKNID} } );
	
    return $self;
}

sub MakeFakeTime
{
	my ($self, $beginindex, $endindex, $maxind) = @_;
	
	$self->{DUR} = sprintf("%.3f", ($self->{SEGET} - $self->{SEGBT})*( ($endindex - $beginindex)/$maxind ) );
	$self->{BT} = sprintf("%.3f", $self->{SEGBT} + ($self->{SEGET} - $self->{SEGBT})*$beginindex/$maxind);
	$self->{MID} = sprintf("%.0f", $self->{BT} + $self->{DUR}/2);
    $self->{XSTARTPOS} = sprintf("%.0f", $self->{PIXPERSEC}*$self->{BT});
	$self->{XMIDPOS} = sprintf("%.0f", $self->{PIXPERSEC}*($self->{BT}+$self->{DUR}/2));
	$self->{XENDPOS} = sprintf("%.0f", $self->{PIXPERSEC}*($self->{BT}+$self->{DUR}));
	$self->{LENGTH} = sprintf("%.0f", $self->{PIXPERSEC}*$self->{DUR});
}

sub DoDisplay
{
	my ($self) = @_;
	$self->{DISPLAY} = 1;
}

sub AddPrevToken
{
	my ($self, $prevtknid) = @_;
	push(@{ $self->{PREVTKNID} }, $prevtknid);
}

sub AddNextToken
{
	my ($self, $nexttknid) = @_;
	push(@{ $self->{NEXTTKNID} }, $nexttknid);
}

sub SetYPos
{
	my ($self, $ypos) = @_;
	$self->{YPOS} += $ypos;
}


sub SetWidthLine
{
	my ($self, $lwidth) = @_;
	$self->{WIDTHLINE} = $lwidth;
}

sub IsFakeToken
{
    my ($self) = @_;
    return( ($self->{SEGBT} == 0) && ($self->{SEGET} == 0) && ($self->{TEXT} eq "") );
}

sub GetDraw
{
	my ($self, $minx) = @_;
	my $output = "";

	return $output if(! $self->{DISPLAY});
	
	if($self->{TEXT} ne "")
	{
		my $modiftexty = ($self->{TEXTPOS} eq "top" ? -17 : 6);
		my $bullet = "images/white_dk.gif";
		$bullet = "images/white_lt.gif" if($self->{REALTIME} == 0);
		
		$output .= sprintf("jg.setColor(\"DimGray\");\n");
		$output .= sprintf("jg.setStroke(%.0f);\n", $self->{WIDTHLINE});
		$output .= sprintf("jg.drawLine(%.0f*scale, %.0f, %.0f*scale, %.0f);\n", $self->{XSTARTPOS} - $minx, $self->{YPOS}, $self->{XENDPOS} - $minx, $self->{YPOS});
		$output .= sprintf("jg.setFont(\"arial\",\"11px\",Font.PLAIN);\n");
		$output .= sprintf("jg.drawStringRect(\"%s\", %.0f*scale + 1, %.0f, %.0f*scale - 1, \"center\");\n", $self->{TEXT}, $self->{XSTARTPOS} - $minx, $self->{YPOS}+$modiftexty, $self->{LENGTH});
		$output .= sprintf("jg.drawImage(\"%s\", %.0f*scale - 7, %.0f - 7, 14, 14);\n", $bullet, $self->{XSTARTPOS} - $minx, $self->{YPOS});
		$output .= sprintf("jg.drawImage(\"%s\", %.0f*scale - 7, %.0f - 7, 14, 14);\n", $bullet, $self->{XENDPOS} - $minx, $self->{YPOS});
	}
	elsif(!( ($self->{SEGBT} == 0) && ($self->{SEGET} == 0) && ($self->{TEXT} eq "") ) )
	{
		$output .= sprintf("jg.setColor(\"brown\");\n");
		$output .= sprintf("jg.setStroke(%.0f);\n", $self->{WIDTHLINE});
		$output .= sprintf("jg.drawLine(%.0f*scale, %.0f, %.0f*scale, %.0f);\n", $self->{XSTARTPOS} - $minx, $self->{YPOS}, $self->{XENDPOS} - $minx, $self->{YPOS});
		$output .= sprintf("jg.drawLine(%.0f*scale, %.0f, %.0f*scale, %.0f);\n", $self->{XSTARTPOS} - $minx, $self->{YPOS} - 4, $self->{XSTARTPOS} - $minx, $self->{YPOS} + 4);
		$output .= sprintf("jg.drawLine(%.0f*scale, %.0f, %.0f*scale, %.0f);\n", $self->{XENDPOS} - $minx, $self->{YPOS} - 4, $self->{XENDPOS} - $minx, $self->{YPOS} + 4);
	}
	
	return $output;
}

1;

