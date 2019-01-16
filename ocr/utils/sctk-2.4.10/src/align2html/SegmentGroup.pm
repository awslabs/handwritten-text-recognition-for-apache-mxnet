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

