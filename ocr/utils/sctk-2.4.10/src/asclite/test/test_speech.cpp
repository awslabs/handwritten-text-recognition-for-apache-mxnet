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

#include "test_speech.h" // class's header file

// class constructor
TestSpeech::TestSpeech()
{
	// insert your code here
}

// class destructor
TestSpeech::~TestSpeech()
{
	// insert your code here
}

/*
 * Divers accessors test for the segments.
 */
void TestSpeech::testSegmentAccessor()
{
	SpeechSet* set = new SpeechSet();
	Speech* speech = new Speech(set);
	Segment* seg1 = Segment::CreateWithEndTime(0, 1000, speech);
	Segment* seg2 = Segment::CreateWithEndTime(1000, 2000, speech);
	speech->AddSegment(seg1);
	speech->AddSegment(seg2);

	//testing
	assert(2 == speech->NbOfSegments());
	assert(seg1 == speech->GetSegment(0));
	assert(seg2 == speech->GetSegment(1));

	delete set;
}

/**
 * Test the NextSegment function
 */
void TestSpeech::testNextSegment()
{
	SpeechSet* set = new SpeechSet();
	Speech* speech = new Speech(set);
	Segment* seg1 = Segment::CreateWithEndTime(0, 1000, speech);
	Segment* seg2 = Segment::CreateWithEndTime(2000, 3000, speech);
	seg1->SetSource("test");
	seg2->SetSource("test");
	seg1->SetChannel("1");
	seg2->SetChannel("1");
	speech->AddSegment(seg1);
	speech->AddSegment(seg2);

	//testing
	assert(speech->NextSegment(500, "test", "1") == seg1);
	assert(speech->NextSegment(1000, "test", "1") == seg2);
	assert(speech->NextSegment(1500, "test", "1") == seg2);
	assert(!speech->NextSegment(3000, "test", "1"));
}

/**
 * Test the GetSegmentsByTime function.
 */
void TestSpeech::testGetSegmentByTime()
{
	SpeechSet* set = new SpeechSet();
	Speech* speech = new Speech(set);
	Segment* seg1 = Segment::CreateWithEndTime(0, 1000, speech);
	Segment* seg2 = Segment::CreateWithEndTime(2000, 3000, speech);
	Segment* seg3 = Segment::CreateWithEndTime(5000, 6000, speech);
	seg1->SetSource("test");
	seg2->SetSource("test");
	seg3->SetSource("test");
	seg1->SetChannel("1");
	seg2->SetChannel("1");
	seg3->SetChannel("1");

	speech->AddSegment(seg1);
	speech->AddSegment(seg2);

	//testing
	//overall case
	vector<Segment*> segs = speech->GetSegmentsByTime(0, 3000, "test", "1");
	assert(segs.size() == 2);
	assert(segs[0] == seg1);
	assert(segs[1] == seg2);

	//limit case
	vector<Segment*> segs2 = speech->GetSegmentsByTime(1000, 5000, "test", "1");
	assert(segs2.size() == 1);
	assert(segs2[0] == seg2);

	//middle case
	vector<Segment*> segs3 = speech->GetSegmentsByTime(500, 2500, "test", "1");
	assert(segs3.size() == 1);
	assert(segs3[0] == seg1);
}


/*
 * Launch all the test of the class.
 */
void TestSpeech::testAll()
{
	cout << "- test SegmentAccessor" << endl;
    testSegmentAccessor();
    cout << "- test NextSegment" << endl;
    testNextSegment();
    cout << "- test GetSegmentByTime" << endl;    
    testGetSegmentByTime();
}
