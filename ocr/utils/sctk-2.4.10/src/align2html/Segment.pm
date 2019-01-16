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

use Token;

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

