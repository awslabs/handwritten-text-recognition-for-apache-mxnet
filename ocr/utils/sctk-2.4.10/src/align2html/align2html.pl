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

package SegmentGroup;
use strict;
use Data::Dumper;

sub new
{
    my $class = shift;
    my $self = {};

	$self->{ID} = shift;
    $self->{REF} = {};
    $self->{SYS} = {};
    $self->{STARTYPOS} = 25;
    $self->{YPOSSTEP} = 20;
    $self->{MINX} = 10000000;
	$self->{MAXX} = 0;
	$self->{HEIGHT} = 0;
	$self->{SPKRHEIGHT} = {};
	$self->{HASFAKETIME} = 0;
	$self->{SPKRYS} = {};
	$self->{FILE} = shift;
	$self->{CHANNEL} = shift;
	$self->{NCORR} = 0;
	$self->{NINS} = 0;
	$self->{NSUB} = 0;
	$self->{NDEL} = 0;
	$self->{NREF} = 0;
	$self->{NSPKRERR} = 0;
	$self->{ET} = 0;
	$self->{BT} = 0;
	$self->{ALIGNED} = 0;
	
    bless $self;    
    return $self;
}

sub unique
{
	my %saw;
	return grep(!$saw{$_}++, sort @_);
}

sub addRefSegment
{
	my ($self, $seg) = @_;
	$self->{REF}{$seg->{ID}} = $seg;
}

sub addSysSegment
{
	my ($self, $seg) = @_;
	$self->{SYS}{$seg->{ID}} = $seg;
}

sub ContainsFakeTimes
{
	my ($self) = @_;
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		$self->{HASFAKETIME} = 1 if($self->{REF}{$segrefid}->{HASFAKETIME} == 1);
		return;
	}
		
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		$self->{HASFAKETIME} = 1 if($self->{SYS}{$segsysid}->{HASFAKETIME} == 1);
		return;
	}
}

sub GetNumHypWords
{
	my ($self) = @_;
	
	my $number = 0;
	foreach my $segsysid(keys %{ $self->{SYS} }) { $number += scalar keys %{ $self->{SYS}{$segsysid}->{TOKENS} } if(!($self->{SYS}{$segsysid}->HasOnlyOneFakeToken())); }
	return $number;
}

sub GetToken
{
	my ($self, $tokenID) = @_;
	
	foreach my $segrefid(keys %{ $self->{REF} })
	{
		return $self->{REF}{$segrefid}->{TOKENS}{$tokenID} if(exists($self->{REF}{$segrefid}->{TOKENS}{$tokenID}));
	}
		
	foreach my $segsysid(keys %{ $self->{SYS} })
	{
		return $self->{SYS}{$segsysid}->{TOKENS}{$tokenID} if(exists($self->{SYS}{$segsysid}->{TOKENS}{$tokenID}));
	}
	
	return undef;
}

sub CalculateBeginEndTime
{
	my ($self) = @_;
	
	$self->{ET} = 0;
	$self->{BT} = 1000000;
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		foreach my $tknid(sort keys %{ $self->{REF}{$segrefid}->{TOKENS} })
		{
			next if( ($self->{REF}{$segrefid}->{TOKENS}{$tknid}->{SEGBT} == 0) && ($self->{REF}{$segrefid}->{TOKENS}{$tknid}->{SEGET} == 0) && ($self->{REF}{$segrefid}->{TOKENS}{$tknid}->{TEXT} eq "") );
			$self->{BT} = $self->{REF}{$segrefid}->{TOKENS}{$tknid}->{SEGBT} if($self->{REF}{$segrefid}->{TOKENS}{$tknid}->{SEGBT} < $self->{BT});
			$self->{ET} = $self->{REF}{$segrefid}->{TOKENS}{$tknid}->{SEGET} if($self->{REF}{$segrefid}->{TOKENS}{$tknid}->{SEGET} > $self->{ET});
		}
	}
	
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		foreach my $tknid(sort keys %{ $self->{SYS}{$segsysid}->{TOKENS} })
		{
			next if( ($self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{BT} == 0) && ($self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{BT} + $self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{DUR} == 0) && ($self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{TEXT} eq "") );
			$self->{BT} = $self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{BT} if($self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{BT} < $self->{BT});
			$self->{ET} = sprintf("%.3f", $self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{BT} + $self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{DUR}) if(sprintf("%.3f", $self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{BT} + $self->{SYS}{$segsysid}->{TOKENS}{$tknid}->{DUR}) > $self->{ET});
		}
	}
}

sub CalculateMinMaxX
{
	my ($self) = @_;
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		$self->{MINX} = $self->{REF}{$segrefid}->{MINX} if($self->{MINX} > $self->{REF}{$segrefid}->{MINX});
		$self->{MAXX} = $self->{REF}{$segrefid}->{MAXX} if($self->{MAXX} < $self->{REF}{$segrefid}->{MAXX});
	}
		
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		$self->{MINX} = $self->{SYS}{$segsysid}->{MINX} if($self->{MINX} > $self->{SYS}{$segsysid}->{MINX});
		$self->{MAXX} = $self->{SYS}{$segsysid}->{MAXX} if($self->{MAXX} < $self->{SYS}{$segsysid}->{MAXX});
	}
}

sub CalculateHeigh
{
	my ($self) = @_;
	
	my %spkrYpos;
	
	$self->{HEIGHT} = 2 * $self->{STARTYPOS} + $self->{YPOSSTEP};
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		my $spkrID = $self->{REF}{$segrefid}->{SPKRID};

		if(!exists($spkrYpos{REF}{$spkrID}))
		{
			$spkrYpos{REF}{$spkrID} = 1;
			$self->{SPKRHEIGHT}{$spkrID} = $self->{REF}{$segrefid}->{HEIGHT};
		}
		else
		{
			$self->{SPKRHEIGHT}{$spkrID} = $self->{REF}{$segrefid}->{HEIGHT} if($self->{REF}{$segrefid}->{HEIGHT} > $self->{SPKRHEIGHT}{$spkrID});
		}
	}
		
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		my $spkrID = $self->{SYS}{$segsysid}->{SPKRID};
		
		if(!exists($spkrYpos{SYS}{$spkrID}))
		{
			$spkrYpos{SYS}{$spkrID} = 1;
			$self->{SPKRHEIGHT}{$spkrID} = $self->{SYS}{$segsysid}->{HEIGHT};
		}
		else
		{
			$self->{SPKRHEIGHT}{$spkrID} = $self->{SYS}{$segsysid}->{HEIGHT} if($self->{SYS}{$segsysid}->{HEIGHT} > $self->{SPKRHEIGHT}{$spkrID});
		}
	}
	
	foreach my $idd (keys %{ $self->{SPKRHEIGHT} } )
	{
		$self->{HEIGHT} += $self->{SPKRHEIGHT}{$idd} + $self->{YPOSSTEP};
	}
	
	return $self->{HEIGHT};
}

sub SetYs
{
	my ($self) = @_;
	
	my $output = "";
	my $currYpos = $self->{STARTYPOS};
	my $YposStep = $self->{YPOSSTEP};
	
	my %spkrYpos;
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		my $spkrID = $self->{REF}{$segrefid}->{SPKRID};		
		my $Ypos;
		
		if(!exists($spkrYpos{REF}{$spkrID}))
		{
			$Ypos = $currYpos;
			$spkrYpos{REF}{$spkrID} = $Ypos;
			$currYpos += $self->{SPKRHEIGHT}{$spkrID} + $YposStep;
		}
		else
		{
			$Ypos = $spkrYpos{REF}{$spkrID};
		}
		
		$self->{REF}{$segrefid}->SetY($Ypos);
		$self->{SPKRYS}{$spkrID} = $Ypos;
	}
	
	$currYpos += 2*$self->{YPOSSTEP};
	
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		my $spkrID = $self->{SYS}{$segsysid}->{SPKRID};		
		my $Ypos;
		
		if(!exists($spkrYpos{SYS}{$spkrID}))
		{
			$Ypos = $currYpos;
			$spkrYpos{SYS}{$spkrID} = $Ypos;
			$currYpos += $self->{SPKRHEIGHT}{$spkrID} + $YposStep;
		}
		else
		{
			$Ypos = $spkrYpos{SYS}{$spkrID};
		}
		
		$self->{SYS}{$segsysid}->SetY($Ypos);
		$self->{SPKRYS}{$spkrID} = $Ypos;
	}
}

sub GetFillREFHYP
{
	my ($self, $maxscalewidth) = @_;	
	my $output = "";
	my @valuessep1;
	
	foreach my $spkrID(keys %{ $self->{SPKRYS} })
	{
		push(@valuessep1, $self->{SPKRYS}{$spkrID} - 25);
		push(@valuessep1, $self->{SPKRYS}{$spkrID} + $self->{SPKRHEIGHT}{$spkrID});
	}
		
	my @valuessep = sort {$a <=> $b} @valuessep1;
	
	my @refspkrs;
	foreach my $segref (keys %{ $self->{REF} }) { push(@refspkrs, $self->{REF}{$segref}->{SPKRID}); }
	my $nrefspkr = scalar unique @refspkrs;
	
	my @sysspkrs;
	foreach my $segsys (keys %{ $self->{SYS} }) { push(@sysspkrs, $self->{SYS}{$segsys}->{SPKRID}); }
	my $nsysspkr = scalar unique @sysspkrs;
	
	if($nrefspkr == 0)
	{
		my $y1 = $valuessep[0];
		my $y2 = $valuessep[scalar(@valuessep) - 1];
		$output .= sprintf("jg.setColor(\"#ffeeee\");\n");
		$output .= sprintf("jg.fillRect(0, %.0f, scale*%.0f - 1, %.0f);\n", $y1 + 1, $maxscalewidth, $y2 - $y1 - 1);
	}
	elsif($nsysspkr == 0)
	{
		my $y1 = $valuessep[0];
		my $y2 = $valuessep[scalar(@valuessep) - 1];
		$output .= sprintf("jg.setColor(\"#eeffee\");\n");
		$output .= sprintf("jg.fillRect(0, %.0f, scale*%.0f - 1, %.0f);\n", $y1 + 1, $maxscalewidth, $y2 - $y1 - 1);
	}
	else
	{
		my $y1 = $valuessep[0];
		my $ym = ($valuessep[2*$nrefspkr - 1] + $valuessep[2*$nrefspkr])/2;
		my $y2 = $valuessep[scalar(@valuessep) - 1];
		$output .= sprintf("jg.setColor(\"#eeffee\");\n");
		$output .= sprintf("jg.fillRect(0, %.0f, scale*%.0f - 1, %.0f);\n", $y1 + 1, $maxscalewidth, $ym - $y1 - 1);
		$output .= sprintf("jg.setColor(\"#ffeeee\");\n");
		$output .= sprintf("jg.fillRect(0, %.0f, scale*%.0f - 1, %.0f);\n", $ym - $y1 + 1, $maxscalewidth, $y2 - $ym -1);
	}

	return $output;
}

sub GetSeparationLines
{
	my ($self, $maxscalewidth) = @_;
	my $output = "";
	
	$output .= sprintf("jg.setColor(\"black\");\n");
	$output .= sprintf("jg.setFont(\"verdana\",\"12px\",Font.PLAIN);\n");
	
	foreach my $spkrID(keys %{ $self->{SPKRYS} })
	{
		my $spkName = $spkrID;
		$spkName =~ s/^ref://;
		$spkName =~ s/^hyp://;
		$output .= sprintf("jg.drawStringRect(\"%s\",%.0f, %.0f, scale*%.0f, \"left\");\n", $spkName, 0, $self->{SPKRYS}{$spkrID} - 8 + ($self->{SPKRHEIGHT}{$spkrID}-30)/2, $maxscalewidth);
	}
	
	$output .= sprintf("jg.setColor(\"yellow\");\n");
	$output .= sprintf("jg.setStroke(1);\n");
	
	my @valuessep1;
	
	foreach my $spkrID(keys %{ $self->{SPKRYS} })
	{
		push(@valuessep1, $self->{SPKRYS}{$spkrID} - 25);
		push(@valuessep1, $self->{SPKRYS}{$spkrID} + $self->{SPKRHEIGHT}{$spkrID});
	}
	
	my @valuessep = sort {$a <=> $b} @valuessep1;
		
	$output .= sprintf("jg.drawLine(0, %.0f, scale*%.0f - 1, %.0f);\n", $valuessep[0], $maxscalewidth, $valuessep[0]);
	
	for(my $i=1; $i<scalar(@valuessep)-1; $i += 2)
	{
		my $ypostmp = ($valuessep[$i] + $valuessep[$i + 1])/2;
		
		$output .= sprintf("jg.drawLine(0, %.0f, scale*%.0f - 1, %.0f);\n", $ypostmp, $maxscalewidth, $ypostmp);
	}
	
	$output .= sprintf("jg.drawLine(0, %.0f, scale*%.0f - 1, %.0f);\n", $valuessep[scalar(@valuessep) - 1], $maxscalewidth, $valuessep[scalar(@valuessep) - 1]);
	
	return $output;
}

sub Compute
{
	my ($self) = @_;
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		$self->{REF}{$segrefid}->Compute();
	}
	
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		$self->{SYS}{$segsysid}->Compute();
	}
	
	$self->CalculateMinMaxX();
	$self->CalculateHeigh();
	$self->ContainsFakeTimes();
	$self->SetYs();
	$self->CalculateBeginEndTime();
}

sub isInSpeakers
{
	my ($self, $spkr) = @_;
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		return 1 if($self->{REF}{$segrefid}->{SPKRID} =~ /^$spkr$/i);
	}
		
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		return 1 if($self->{SYS}{$segsysid}->{SPKRID} =~ /^$spkr$/i);
	}
	
	return 0;
}

sub GetDraw
{
	my ($self, $decalx) = @_;
	my $output = "";
	
	foreach my $segrefid(sort keys %{ $self->{REF} })
	{
		$output .= $self->{REF}{$segrefid}->GetDraw($self->{MINX} - $decalx);		
	}
		
	foreach my $segsysid(sort keys %{ $self->{SYS} })
	{
		$output .= $self->{SYS}{$segsysid}->GetDraw($self->{MINX} - $decalx);	
	}
	
	return $output;
}

1;

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

package Segment;
use strict;
use Data::Dumper;


sub new
{
    my $class = shift;
    my $self = {};

	$self->{ID} = shift;
	$self->{SPKRID} = shift;
	$self->{TOKENS} = {};
	$self->{MINX} = 10000000;
	$self->{MAXX} = 0;
	$self->{HEIGHT} = 0;
	$self->{HASFAKETIME} = 0;
	
    bless $self;
    return $self;
}

sub AddToken
{
	my ($self, $tkn) = @_;
	$self->{TOKENS}{$tkn->{ID}} = $tkn;
}

sub SetY
{
	my ($self, $ypos) = @_;
		
	foreach my $tknid(keys %{ $self->{TOKENS} })
	{
		$self->{TOKENS}{$tknid}->SetYPos($ypos);
	}
}

sub isTokenInSegment
{
	my ($self, $tokenid) = @_;
	
	foreach my $tknid(keys %{ $self->{TOKENS} })
	{
		return 1 if($tknid == $tokenid);
	}
	
	return 0;
}

sub CalculateMinMaxX
{
	my ($self) = @_;
	
	foreach my $tknid(keys %{ $self->{TOKENS} })
	{
		next if( ($self->{TOKENS}{$tknid}->{XSTARTPOS} == 0) && ($self->{TOKENS}{$tknid}->{XENDPOS} == 0) && ($self->{TOKENS}{$tknid}->{TEXT} eq "") );
		$self->{MINX} = $self->{TOKENS}{$tknid}->{XSTARTPOS} if($self->{MINX} > $self->{TOKENS}{$tknid}->{XSTARTPOS});
		$self->{MAXX} = $self->{TOKENS}{$tknid}->{XENDPOS}   if($self->{MAXX} < $self->{TOKENS}{$tknid}->{XENDPOS});
	}
}

sub HasOnlyOneFakeToken
{
    my ($self) = @_;
    
    foreach my $tknid(keys %{ $self->{TOKENS} })
	{
		return 0 if( !( ($self->{TOKENS}{$tknid}->{XSTARTPOS} == 0) && ($self->{TOKENS}{$tknid}->{XENDPOS} == 0) && ($self->{TOKENS}{$tknid}->{TEXT} eq "") ) );
	}
	
	return 1;
}

sub CleanSegmentPrevNext
{
	my ($self) = @_;
	
	foreach my $tknid(keys %{ $self->{TOKENS} })
	{
		my $myToken = $self->{TOKENS}{$tknid};
		
		for(my $i=0; $i< @{ $myToken->{PREVTKNID} }; $i++)
		{
			delete $myToken->{PREVTKNID}[$i] if(!$self->isTokenInSegment($myToken->{PREVTKNID}[$i]));
		}
		
		for(my $i=0; $i< @{ $myToken->{NEXTTKNID} }; $i++)
		{
			delete $myToken->{NEXTTKNID}[$i] if(!$self->isTokenInSegment($myToken->{NEXTTKNID}[$i]));
		}
		
		$myToken->{NUMPREV} = scalar( @{ $myToken->{PREVTKNID} } );
		$myToken->{NUMNEXT} = scalar( @{ $myToken->{NEXTTKNID} } );
	}
}

sub MultiGraphXY
{
	my ($self) = @_;
	
	my $maxheigh = 0;
	
	if(scalar(keys %{ $self->{TOKENS} }) != 0)
	{
		my %tokenPositions;
		my @listTokenID;
		my $currentlevel = 0;
		my $maxlengh = 0;
	
		foreach my $tknid(sort {$a <=> $b} keys %{ $self->{TOKENS} })
		{
			$tokenPositions{$tknid}{XB} = 0;
			$tokenPositions{$tknid}{XE} = 1;
			$tokenPositions{$tknid}{Y} = 0;
			$tokenPositions{$tknid}{TEXT} = $self->{TOKENS}{$tknid}->{TEXT};
			
			push(@listTokenID, $tknid);
			
			if( scalar(@{ $self->{TOKENS}{$tknid}->{PREVTKNID} }) == 0 )
			{
				$tokenPositions{$tknid}{Y} = $currentlevel;
				$currentlevel++;
			}
		}
			
		for(my $i=0; $i<@listTokenID; $i++)
		{
			my $tokenid = $listTokenID[$i];
			my $token = $self->{TOKENS}{$tokenid};
			my @prevtkn = @{ $token->{PREVTKNID} };
			my @nexttkn = @{ $token->{NEXTTKNID} };
			
			my $tokenBegin = $tokenPositions{$tokenid}{XB};
			my $tokenEnd = $tokenPositions{$tokenid}{XE};
			my $tokenY = $tokenPositions{$tokenid}{Y};
			
			for(my $j=0; $j<@nexttkn; $j++)
			{
				my $nexttokenid = $nexttkn[$j];
				
				my $nexttokenBegin = $tokenPositions{$nexttokenid}{XB};
				my $nexttokenEnd = $tokenPositions{$nexttokenid}{XE};
				my $nextTokenY = $tokenPositions{$nexttokenid}{Y};
				
				if( $nexttokenBegin < $tokenEnd)
				{
					$tokenPositions{$nexttokenid}{XB} = $tokenEnd;
					$tokenPositions{$nexttokenid}{XE} = $tokenPositions{$nexttokenid}{XB} + 1;
					$i = -1;
				}
				
				my $nbrprev = scalar( @{ $self->{TOKENS}{$nexttokenid}->{PREVTKNID} } ) - 1;
				
				if($nextTokenY < ($tokenY + $j - $nbrprev))
				{
					$tokenPositions{$nexttokenid}{Y} = $tokenY + $j - $nbrprev;
					$i = -1;
				}
			}
			
			for(my $j=0; $j<@prevtkn; $j++)
			{
				my $prevtokenid = $prevtkn[$j];
				
				my $prevtokenBegin = $tokenPositions{$prevtokenid}{XB};
				my $prevtokenEnd = $tokenPositions{$prevtokenid}{XE};
				
				if( $prevtokenEnd != $tokenBegin)
				{
					$tokenPositions{$prevtokenid}{XE} = $tokenBegin;
					$i = -1;
				}
			}
		}
		
		foreach my $tken (sort {$a <=> $b} keys %tokenPositions)
		{
			$maxheigh = $tokenPositions{$tken}{Y} if($tokenPositions{$tken}{Y} > $maxheigh);
			$maxlengh = $tokenPositions{$tken}{XE} if($tokenPositions{$tken}{XE} > $maxlengh);
		}
		
		foreach my $tknid(keys %{ $self->{TOKENS} })
		{
			my $updown = 1;
			my $base = 0;
			
			if($self->{TOKENS}{$tknid}->{REFORSYS} eq "REF")
			{
				$updown = -1;
				$base = $maxheigh;
			}
		
			if($self->{TOKENS}{$tknid}->{REALTIME} == 0)
			{
				$self->{TOKENS}{$tknid}->MakeFakeTime($tokenPositions{$tknid}{XB}, $tokenPositions{$tknid}{XE}, $maxlengh);
				$self->{HASFAKETIME} = 1;
			}
			
			$self->{TOKENS}{$tknid}->{YPOS} += $base*30 + $updown*30*$tokenPositions{$tknid}{Y};
		}
	}
	
	$self->{HEIGHT} = ($maxheigh+1)*30;
}

sub Compute
{
	my ($self) = @_;
	
	# Calculate Min and Max X
	$self->CalculateMinMaxX();
	
	# Clean up the previous and Next of Tokens
	$self->CleanSegmentPrevNext();

	# Assign X and Y for graph
	$self->MultiGraphXY();	
}

sub GetDraw
{
	my ($self, $minx) = @_;
	my $output2 = "";
	
	my @listfirsts;
	my @listlasts;
	
	foreach my $tknid(keys %{ $self->{TOKENS} })
	{
		my @prevtkn = @{ $self->{TOKENS}{$tknid}->{PREVTKNID} };
		my @nexttkn = @{ $self->{TOKENS}{$tknid}->{NEXTTKNID} };
		
		if(scalar(@prevtkn) > 1)
		{
			for(my $i=1; $i<@prevtkn; $i++)
			{
				my $end1x = $self->{TOKENS}{$prevtkn[$i-1]}->{XENDPOS} - $minx;
				my $end1y = $self->{TOKENS}{$prevtkn[$i-1]}->{YPOS};
				my $end2x = $self->{TOKENS}{$prevtkn[$i]}->{XENDPOS} - $minx;
				my $end2y = $self->{TOKENS}{$prevtkn[$i]}->{YPOS};
				
				$output2 .= sprintf("jg.drawLine(scale*%.0f, %.0f, scale*%.0f, %.0f);\n", $end1x, $end1y, $end2x, $end2y);
			}
		}
		
		push(@listfirsts, $tknid) if(scalar(@prevtkn) == 0);
		
		if(scalar(@nexttkn) > 1)
		{
			for(my $i=1; $i<@nexttkn; $i++)
			{
				my $begin1x = $self->{TOKENS}{$nexttkn[$i-1]}->{XSTARTPOS} - $minx;
				my $begin1y = $self->{TOKENS}{$nexttkn[$i-1]}->{YPOS};
				my $begin2x = $self->{TOKENS}{$nexttkn[$i]}->{XSTARTPOS} - $minx;
				my $begin2y = $self->{TOKENS}{$nexttkn[$i]}->{YPOS};
				
				$output2 .= sprintf("jg.drawLine(scale*%.0f, %.0f, scale*%.0f, %.0f);\n", $begin1x, $begin1y, $begin2x, $begin2y);
			}			
		}
		
		push(@listlasts, $tknid) if(scalar(@nexttkn) == 0);
	}
	
	if(scalar(@listfirsts) > 1)
	{
		for(my $i=1; $i<@listfirsts; $i++)
		{
			my $begin1x = $self->{TOKENS}{$listfirsts[$i-1]}->{XSTARTPOS} - $minx;
			my $begin1y = $self->{TOKENS}{$listfirsts[$i-1]}->{YPOS};
			my $begin2x = $self->{TOKENS}{$listfirsts[$i]}->{XSTARTPOS} - $minx;
			my $begin2y = $self->{TOKENS}{$listfirsts[$i]}->{YPOS};
			
			$output2 .= sprintf("jg.drawLine(scale*%.0f, %.0f, scale*%.0f, %.0f);\n", $begin1x, $begin1y, $begin2x, $begin2y);
		}			
	}
	
	if(scalar(@listlasts) > 1)
	{
		for(my $i=1; $i<@listlasts; $i++)
		{
			my $end1x = $self->{TOKENS}{$listlasts[$i-1]}->{XENDPOS} - $minx;
			my $end1y = $self->{TOKENS}{$listlasts[$i-1]}->{YPOS};
			my $end2x = $self->{TOKENS}{$listlasts[$i]}->{XENDPOS} - $minx;
			my $end2y = $self->{TOKENS}{$listlasts[$i]}->{YPOS};
			
			$output2 .= sprintf("jg.drawLine(scale*%.0f, %.0f, scale*%.0f, %.0f);\n", $end1x, $end1y, $end2x, $end2y);
		}			
	}
	
	my $output = "";
	
	if($output2 ne "")
	{
		$output .= sprintf("jg.setColor(\"grey\");\n");
		$output .= sprintf("jg.setStroke(Stroke.DOTTED);\n");
		$output .= $output2;
	}
	
	foreach my $tknid(keys %{ $self->{TOKENS} })
	{		
		$output .= $self->{TOKENS}{$tknid}->GetDraw($minx);
	}
	
	return $output;
}

1;

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

package main;

Getopt::Long::Configure(qw( auto_abbrev no_ignore_case ));

my $VERSION = "0.6";
my $AlignFile = "";
my $MapFile = "";
my $Outputdir = "";
my $NonLinearSegments = 0;
my $pixelpersec = 100;
my $maxscalewidth = 0;

## This variable says if the program is installed as a standalone executable
my $Installed = 1;

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
my $here = "";
$here = << 'EOTAR';
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
M8B\VHI)GZ!\```````````````!_Y@%8KS5]`'@`````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
M````````````````````````````````````````````````````````````
9````````````````````````````````````

EOTAR
open UNTAR, "| (cd $outDir ; tar zxf -)" || die "Failed to UUDECODE/TAR";
binmode(UNTAR);
print UNTAR unpack("u",$here);
close UNTAR
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

