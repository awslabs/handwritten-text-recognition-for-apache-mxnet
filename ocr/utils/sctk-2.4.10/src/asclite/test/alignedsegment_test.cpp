/*
 * ASCLITE
 * Author: Jerome Ajot, Jon Fiscus, Nicolas Radde, Chris Laprun
 *
 * This software was developed at the National Institute of Standards and Technology by 
 * employees of the Federal Government in the course of their official duties. Pursuant
 * to title 17 Section 105 of the United States Code this software is not subject to
 * copyright protection and is in the public domain. ASCLITE is an experimental system.
 * NIST assumes no responsibility whatsoever for its use by other parties, and makes no
 * guarantees, expressed or implied, about its quality, reliability, or any other
 * characteristic. We would appreciate acknowledgement if the software is used.
 *
 * THIS SOFTWARE IS PROVIDED "AS IS."  With regard to this software, NIST MAKES NO EXPRESS
 * OR IMPLIED WARRANTY AS TO ANY MATTER WHATSOEVER, INCLUDING MERCHANTABILITY,
 * OR FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "alignedsegment_test.h"
#include "alignedsegment.h"
#include "segment.h"
#include "token.h"
#include "speech.h"
#include "speechset.h"

void AlignedSegmentTest::TestAll()
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* seg = Segment::CreateWithDuration(0, 10000, speech);
	Segment* seg2 = Segment::CreateWithDuration(0, 10000, speech);
	AlignedSegment* as = new AlignedSegment(seg);
	assert(as->GetReferenceSegment() == seg);
	Token* ref1 = Token::CreateWithDuration(0, 5000, seg, "a");
	Token* ref2 = Token::CreateWithDuration(0, 5000, NULL);
	Token* hyp1 = Token::CreateWithDuration(200, 4300, seg2, "a");
	Token* hyp2 = Token::CreateWithDuration(100, 4800, seg2, "b");

	// add improper ref (the parent segment of ref2 must be seg to be correct)
	assert(as->AddTokenAlignment(ref2, "hyp1", hyp1) < 0);

	// add correct ref
	assert(as->AddTokenAlignment(ref1, "hyp1", hyp1) > 0);
	assert(as->AddTokenAlignment(ref1, "hyp2", hyp2) > 0);

	delete speechSet;
	delete speech;
	delete seg;
	delete seg2;
	delete ref1;
	delete ref2;
	delete hyp1;
	delete hyp2;
}
