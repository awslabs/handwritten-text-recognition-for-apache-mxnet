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

#ifndef ALIGNEDSPEECH_TEST_H
#define ALIGNEDSPEECH_TEST_H

#include "stdinc.h"
#include "alignedspeech.h"
#include "speech.h"
#include "segment.h"
#include "alignedsegment.h"
#include "speechset.h"

class AlignedSpeechTest {
public:
	AlignedSpeechTest();
	~AlignedSpeechTest();
	
	void TestAll();
	void TestIncorrectAddition();
	void TestAdditions();
	void TestIteration();
	
	Segment* CreateSegment(int i);
		
private:
    SpeechSet* speechSet;
		Speech* speech;
		AlignedSpeech* asp;
		Segment* seg1;	
		Segment* seg2;
		Segment* seg3;
		Segment* seg4;
		AlignedSegment* asg1;
		AlignedSegment* asg2;
		AlignedSegment* asg3;
		AlignedSegment* asg4;
};

#endif
