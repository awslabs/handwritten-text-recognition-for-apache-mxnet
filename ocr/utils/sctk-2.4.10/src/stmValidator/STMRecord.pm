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

package STMRecord;
use strict;
 
sub new {
    my $class = shift;
    my $self = {};
    $self->{FILE} = shift;
    $self->{CHAN} = shift;
    $self->{SPKR} = shift;
    $self->{BT} = shift;
    $self->{ET} = shift;
    $self->{LABEL} = shift;
    $self->{TEXT} = shift;
    bless $self;
    return $self;
}
 
sub toString {
    my $self = shift;
    "STM: ".
	" FILE=".$self->{FILE}.
	" CHAN=".$self->{CHAN}.
	" SPKR=".$self->{SPKR}.
	" BT=".$self->{BT}.
	" ET=".$self->{ET}.
	" LABEL=".$self->{LABEL}.
	" TEXT=".$self->{TEXT};
}

sub getText{
    my $self = shift;
    $self->{TEXT};

}

sub getSpkr{
    my $self = shift;
    $self->{SPKR};
}

sub getBt{
    my $self = shift;
    $self->{BT};
}

sub getEt{
    my $self = shift;
    $self->{ET};
}

sub overlapsWith{
    my ($self, $bt, $et, $tolerance) = @_;
    return 0 if ($self->{BT} + $tolerance > $et);
    return 0 if ($self->{ET} - $tolerance < $bt);
    return 1;
}
1;
