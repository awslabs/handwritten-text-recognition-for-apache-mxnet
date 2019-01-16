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

#include "alignedspeech_test.h"
#include "alignedsegmentiterator.h"

AlignedSpeechTest::AlignedSpeechTest() {
	speechSet = new SpeechSet();
  speech = new Speech(speechSet);
	asp = new AlignedSpeech(speech);
	seg1 = CreateSegment(1);	
	seg2 = CreateSegment(2);
	seg3 = CreateSegment(3);
	seg4 = CreateSegment(4);
	
	assert(asp->GetReferenceSpeech() == speech);	
}

Segment* AlignedSpeechTest::CreateSegment(int i) {
	Segment* result = Segment::CreateWithDuration(i * 10000, 10000, speech);
	std::ostringstream oss;
	oss << i;
	string str = oss.str();
	result->SetChannel(str);
	result->SetSource("source" + str);
	result->SetSpeakerId("speaker" + str);
	return result;
}

AlignedSpeechTest::~AlignedSpeechTest()
{
	/*
	if(asp) delete asp;
	if(asg1) delete asg1;
	if(asg2 && asg2 != asg1) delete asg2;
	if(asg3) delete asg3;
	if(asg4) delete asg4;
	if(seg1) delete seg1;
	if(seg2) delete seg2;
	if(seg3) delete seg3;
	if(seg4) delete seg4;
	*/
}

void AlignedSpeechTest::TestAll() {	
	TestIncorrectAddition();
	
	TestAdditions();
	
	TestIteration();
}

void AlignedSpeechTest::TestIncorrectAddition() {
	cout << "Prevent incorrect segment addition... ";
	cout.flush();
	Segment* incorrect = Segment::CreateWithDuration(0, 10000, NULL);	
	assert(asp->GetOrCreateAlignedSegmentFor(incorrect, true) == NULL);
	delete incorrect;
	cout << "OK" << endl;
}

void AlignedSpeechTest::TestAdditions() {	
	cout << "Don't create new AlignedSegment...";
	cout.flush();
	assert(asp->GetOrCreateAlignedSegmentFor(seg1, false) == NULL);
	cout << "OK" << endl;
	
	cout << "Create and check it was correctly created...";
	cout.flush();
	asg1 = asp->GetOrCreateAlignedSegmentFor(seg1, true);
	assert(asp->GetOrCreateAlignedSegmentFor(seg1, true) == asg1);
	assert(asg1->GetReferenceSegment() == seg1);
	cout << "OK" << endl;
	
	cout << "Populate and check that all went well...";
	cout.flush();
	asg2 = asp->GetOrCreateAlignedSegmentFor(seg2, true);
	assert(asp->GetOrCreateAlignedSegmentFor(seg2, false) == asg2);
	asg3 = asp->GetOrCreateAlignedSegmentFor(seg3, true);
	assert(asp->GetOrCreateAlignedSegmentFor(seg3, false) == asg3);
	asg4 = asp->GetOrCreateAlignedSegmentFor(seg4, true);
	assert(asp->GetOrCreateAlignedSegmentFor(seg4, false) == asg4);
	cout << "OK" << endl;
}

void AlignedSpeechTest::TestIteration() {
	cout << "Checking that iteration works as expected...";
	cout.flush();
	AlignedSegmentIterator* alSegments = asp->AlignedSegments();
	AlignedSegment* asg = NULL;
	
	vector< AlignedSegment* > asv;
	while(alSegments->Current(&asg)) {
		asv.push_back(asg);
	}
	
	assert(asv.size() == 4); // need better tests...
	
	/*vector< AlignedSegment* >::iterator result = find(asv.begin(), asv.end(), asg1);
	assert(result == asv.end() || *result == asg1);
	result = find(asv.begin(), asv.end(), asg2);
	assert(result == asv.end() || *result == asg2);
	result = find(asv.begin(), asv.end(), asg3);
	assert(result == asv.end() || *result == asg3);
	result = find(asv.begin(), asv.end(), asg4);
	assert(result == asv.end() || *result == asg4);*/
	
	cout << "OK" << endl;
}
