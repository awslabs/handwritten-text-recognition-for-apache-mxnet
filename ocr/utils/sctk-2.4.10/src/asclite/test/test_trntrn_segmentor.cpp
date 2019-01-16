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
 
#include "test_trntrn_segmentor.h"

// class constructor
TRNTRNSegmentorTest::TRNTRNSegmentorTest()
{
	// insert your code here
}

// class destructor
TRNTRNSegmentorTest::~TRNTRNSegmentorTest()
{
	// insert your code here
}

/*
 * Launch all the tests on the TRN to TRN segmentor
 */
void TRNTRNSegmentorTest::testAll()
{
	cout << "- test 2segmentsCase" << endl;
	test2segmentsCase();
	cout << "- test 4segments2SpkrCase" << endl;
	test4segments2spkrCase();
}

/*
 * Test the case where the segmentor need to align 2 segments to
 * 2 segments.
 */
void TRNTRNSegmentorTest::test2segmentsCase()
{
	SpeechSet* speechSet_ref = new SpeechSet();
	SpeechSet* speechSet_hyp = new SpeechSet();
	Speech* speech_ref = new Speech(speechSet_ref);
	Speech* speech_hyp = new Speech(speechSet_hyp);
	Segment* seg_ref1 = Segment::CreateWithEndTime(-1, -1, speech_ref);
	seg_ref1->SetId("id-1");
	Segment* seg_ref2 = Segment::CreateWithEndTime(-1, -1, speech_ref);
	seg_ref2->SetId("id-2");
	Segment* seg_hyp1 = Segment::CreateWithEndTime(-1, -1, speech_hyp);
	seg_hyp1->SetId("id-1");
	Segment* seg_hyp2 = Segment::CreateWithEndTime(-1, -1, speech_hyp);
	seg_hyp2->SetId("id-2");
	speech_ref->AddSegment(seg_ref1);
	speech_ref->AddSegment(seg_ref2);
	speech_hyp->AddSegment(seg_hyp1);
	speech_hyp->AddSegment(seg_hyp2);
	SpeechSet* temp_ref = new SpeechSet();
	SpeechSet* temp_hyp = new SpeechSet();
	temp_ref->AddSpeech(speech_ref);
	temp_hyp->AddSpeech(speech_hyp);
	TRNTRNSegmentor* segmentor = new TRNTRNSegmentor();

	segmentor->Reset(temp_ref, temp_hyp);
	assert(segmentor->HasNext());
	SegmentsGroup* seg_group = segmentor->Next();
	assert(seg_group->GetNumberOfReferences() == 1);
	assert(seg_group->GetNumberOfHypothesis() == 1);
	assert(seg_group->GetReference(0).size() == 1);
	assert(seg_group->GetHypothesis(0).size() == 1);
	//cout << seg_group->GetReference(0)[0]->GetId() << endl;
	//cout << seg_group->GetHypothesis(0)[0]->GetId() << endl;
	assert(seg_group->GetReference(0)[0]->GetId().compare(seg_group->GetHypothesis(0)[0]->GetId()) == 0);
	assert(segmentor->HasNext());
	seg_group = segmentor->Next();
	assert(seg_group->GetNumberOfReferences() == 1);
	assert(seg_group->GetNumberOfHypothesis() == 1);
	assert(seg_group->GetReference(0).size() == 1);
	assert(seg_group->GetHypothesis(0).size() == 1);
	assert(seg_group->GetReference(0)[0]->GetId().compare(seg_group->GetHypothesis(0)[0]->GetId()) == 0);
	assert(!segmentor->HasNext());
}

/*
 * Test the case where the segmentor need to align 4 segments to
 * 4 segments. there are affected to 2 different speakers
 */
void TRNTRNSegmentorTest::test4segments2spkrCase()
{
	SpeechSet* speechSet_ref = new SpeechSet();
	SpeechSet* speechSet_hyp = new SpeechSet();
	Speech* speech_ref = new Speech(speechSet_ref);
	Speech* speech_hyp = new Speech(speechSet_hyp);
	Segment* seg_ref1 = Segment::CreateWithEndTime(-1, -1, speech_ref);
	seg_ref1->SetId("jo-1");
	Segment* seg_ref2 = Segment::CreateWithEndTime(-1, -1, speech_ref);
	seg_ref2->SetId("jo-2");
	Segment* seg_ref3 = Segment::CreateWithEndTime(-1, -1, speech_ref);
	seg_ref3->SetId("al-1");
	Segment* seg_ref4 = Segment::CreateWithEndTime(-1, -1, speech_ref);
	seg_ref4->SetId("al-2");
	Segment* seg_hyp1 = Segment::CreateWithEndTime(-1, -1, speech_hyp);
	seg_hyp1->SetId("jo-1");
	Segment* seg_hyp2 = Segment::CreateWithEndTime(-1, -1, speech_hyp);
	seg_hyp2->SetId("jo-2");
	Segment* seg_hyp3 = Segment::CreateWithEndTime(-1, -1, speech_hyp);
	seg_hyp3->SetId("al-1");
	Segment* seg_hyp4 = Segment::CreateWithEndTime(-1, -1, speech_hyp);
	seg_hyp4->SetId("al-2");
	speech_ref->AddSegment(seg_ref1);
	speech_ref->AddSegment(seg_ref2);
	speech_ref->AddSegment(seg_ref3);
	speech_ref->AddSegment(seg_ref4);
	speech_hyp->AddSegment(seg_hyp1);
	speech_hyp->AddSegment(seg_hyp2);
	speech_hyp->AddSegment(seg_hyp3);
	speech_hyp->AddSegment(seg_hyp4);
	SpeechSet* temp_ref = new SpeechSet();
	SpeechSet* temp_hyp = new SpeechSet();
	temp_ref->AddSpeech(speech_ref);
	temp_hyp->AddSpeech(speech_hyp);
	TRNTRNSegmentor* segmentor = new TRNTRNSegmentor();

	segmentor->Reset(temp_ref, temp_hyp);
	assert(segmentor->HasNext());
	SegmentsGroup* seg_group = segmentor->Next();
	assert(seg_group->GetNumberOfReferences() == 1);
	assert(seg_group->GetNumberOfHypothesis() == 1);
	assert(seg_group->GetReference(0).size() == 1);
	assert(seg_group->GetHypothesis(0).size() == 1);
	assert(seg_group->GetReference(0)[0]->GetId().compare(seg_group->GetHypothesis(0)[0]->GetId()) == 0);
	assert(segmentor->HasNext());
	seg_group = segmentor->Next();
	assert(seg_group->GetNumberOfReferences() == 1);
	assert(seg_group->GetNumberOfHypothesis() == 1);
	assert(seg_group->GetReference(0).size() == 1);
	assert(seg_group->GetHypothesis(0).size() == 1);
	assert(seg_group->GetReference(0)[0]->GetId().compare(seg_group->GetHypothesis(0)[0]->GetId()) == 0);
	assert(segmentor->HasNext());
	seg_group = segmentor->Next();
	assert(seg_group->GetNumberOfReferences() == 1);
	assert(seg_group->GetNumberOfHypothesis() == 1);
	assert(seg_group->GetReference(0).size() == 1);
	assert(seg_group->GetHypothesis(0).size() == 1);
	assert(seg_group->GetReference(0)[0]->GetId().compare(seg_group->GetHypothesis(0)[0]->GetId()) == 0);
	assert(segmentor->HasNext());
	seg_group = segmentor->Next();
	assert(seg_group->GetNumberOfReferences() == 1);
	assert(seg_group->GetNumberOfHypothesis() == 1);
	assert(seg_group->GetReference(0).size() == 1);
	assert(seg_group->GetHypothesis(0).size() == 1);
	assert(seg_group->GetReference(0)[0]->GetId().compare(seg_group->GetHypothesis(0)[0]->GetId()) == 0);
	assert(!segmentor->HasNext());
}
