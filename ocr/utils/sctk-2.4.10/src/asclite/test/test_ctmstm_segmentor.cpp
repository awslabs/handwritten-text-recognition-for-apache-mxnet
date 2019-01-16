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

#include "test_ctmstm_segmentor.h" // class's header file

// class constructor
CTMSTMRTTMSegmentorTest::CTMSTMRTTMSegmentorTest()
{
	// insert your code here
}

// class destructor
CTMSTMRTTMSegmentorTest::~CTMSTMRTTMSegmentorTest()
{
	// insert your code here
}

void CTMSTMRTTMSegmentorTest::testBasicIteration()
{
  SpeechSet* speechSet_ref = new SpeechSet();
  SpeechSet* speechSet_hyp = new SpeechSet();
  Speech* speech_ref = new Speech(speechSet_ref);
  Speech* speech_hyp = new Speech(speechSet_hyp);
  Segment* seg_ref = Segment::CreateWithEndTime(0, 5000, speech_ref);
  seg_ref->SetSource("source1");
  seg_ref->SetChannel("1");
  Segment* seg_hyp = Segment::CreateWithEndTime(0, 5000, speech_hyp);
  seg_hyp->SetSource("source1");
  seg_hyp->SetChannel("1");
  speech_ref->AddSegment(seg_ref);
  speech_hyp->AddSegment(seg_hyp);
  speechSet_ref->AddSpeech(speech_ref);
  speechSet_hyp->AddSpeech(speech_hyp);
  CTMSTMRTTMSegmentor* segmentor = new CTMSTMRTTMSegmentor();

  segmentor->Reset(speechSet_ref, speechSet_hyp);
  assert(segmentor->HasNext());
  SegmentsGroup* seg_group = segmentor->Next();
  assert(!segmentor->HasNext());
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 1);
}

void CTMSTMRTTMSegmentorTest::testSimpleRefOverlapIteration()
{
  SpeechSet* speechSet_ref = new SpeechSet();
  SpeechSet* speechSet_hyp = new SpeechSet();
  Speech* speech_ref = new Speech(speechSet_ref);
  Speech* speech_hyp = new Speech(speechSet_hyp);
  Segment* seg1_ref = Segment::CreateWithEndTime(0, 3000, speech_ref);
  seg1_ref->SetSource("source1");
  seg1_ref->SetChannel("1");
  Segment* seg2_ref = Segment::CreateWithEndTime(2000, 5000, speech_ref);
  seg2_ref->SetSource("source1");
  seg2_ref->SetChannel("1");
  Segment* seg_hyp = Segment::CreateWithEndTime(0, 4000, speech_hyp);
  seg_hyp->SetSource("source1");
  seg_hyp->SetChannel("1");
  speech_ref->AddSegment(seg1_ref);
  speech_ref->AddSegment(seg2_ref);
  speech_hyp->AddSegment(seg_hyp);
  speechSet_ref->AddSpeech(speech_ref);
  speechSet_hyp->AddSpeech(speech_hyp);
  CTMSTMRTTMSegmentor* segmentor = new CTMSTMRTTMSegmentor();

  segmentor->Reset(speechSet_ref, speechSet_hyp);
  assert(segmentor->HasNext());
  SegmentsGroup* seg_group = segmentor->Next();
  assert(!segmentor->HasNext());
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 1);
  assert(seg_group->GetReference(0).size() == 2);
  assert(seg_group->GetHypothesis(0).size() == 1);
}

void CTMSTMRTTMSegmentorTest::testSequentialSegments()
{
  /**   ,------------.--------------,----------.
   *      seg1_ref      seg2_ref          seg3_ref
   **/       
  typedef pair<int, int> p2f;

  SpeechSet* speechSet_ref = new SpeechSet();
  SpeechSet* speechSet_hyp = new SpeechSet();
  Speech* speech_ref = new Speech(speechSet_ref);
  Speech* speech_hyp = new Speech(speechSet_hyp);
  Segment* seg;

  vector < pair < int, int> > time;
  time.push_back(p2f(0,3000));
  time.push_back(p2f(3000,5000));
  time.push_back(p2f(5000,10000));
  time.push_back(p2f(10000,13000));
  
  for (unsigned int i=0; i<time.size(); i++)
  {
    seg = Segment::CreateWithEndTime(time[i].first, time[i].second, speech_ref);
    seg->SetSource("source1");
    seg->SetChannel("1");
    speech_ref->AddSegment(seg);
  }

  time.clear();
  time.push_back(p2f(0,1000));
  time.push_back(p2f(1000,2000));
  time.push_back(p2f(2000,3000));
  time.push_back(p2f(9000,10100));
  time.push_back(p2f(12900, 13500));
  for (unsigned int i=0; i<time.size(); i++){
    seg = Segment::CreateWithEndTime(time[i].first, time[i].second, speech_hyp);
    seg->SetSource("source1");
    seg->SetChannel("1");
    speech_hyp->AddSegment(seg);
  }

  speechSet_ref->AddSpeech(speech_ref);
  speechSet_hyp->AddSpeech(speech_hyp);
  CTMSTMRTTMSegmentor* segmentor = new CTMSTMRTTMSegmentor();

  segmentor->Reset(speechSet_ref, speechSet_hyp);
  assert(segmentor->HasNext());
  SegmentsGroup* seg_group = segmentor->Next();
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 1);
  assert(seg_group->GetReference(0).size() == 1);
  assert(seg_group->GetHypothesis(0).size() == 3);

  assert(segmentor->HasNext());
  seg_group = segmentor->Next();
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 0);
  assert(seg_group->GetReference(0).size() == 1);
//  This one shouldnt exist
//  assert(seg_group->GetHypothesis(0).size() == 0);

  assert(segmentor->HasNext());
  seg_group = segmentor->Next();
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 1);
  assert(seg_group->GetReference(0).size() == 1);
  assert(seg_group->GetHypothesis(0).size() == 1);
  assert(segmentor->HasNext());

  seg_group = segmentor->Next();
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 0); // THIS ONE NEED TO BE CHECK WITH JON
//  assert(seg_group->GetNumberOfHypothesis() == 1);
  assert(seg_group->GetReference(0).size() == 1);
//  assert(seg_group->GetHypothesis(0).size() == 1);

  assert(!segmentor->HasNext());

}

void CTMSTMRTTMSegmentorTest::testEmptySegmentOverlapCaseIteration()
{
  SpeechSet* speechSet_ref = new SpeechSet();
  SpeechSet* speechSet_hyp = new SpeechSet();
  Speech* speech1_ref = new Speech(speechSet_ref);
  Speech* speech2_ref = new Speech(speechSet_ref);
  Speech* speech_hyp = new Speech(speechSet_hyp);
  Segment* seg1_ref = Segment::CreateWithEndTime(0, 3000, speech1_ref);
  seg1_ref->SetSource("source1");
  seg1_ref->SetChannel("1");
  Segment* seg2_ref = Segment::CreateWithEndTime(2000, 5000, speech2_ref);
  seg2_ref->SetSource("source1");
  seg2_ref->SetChannel("1");
  Segment* seg1_hyp = Segment::CreateWithEndTime(0, 1000, speech_hyp);
  seg1_hyp->SetSource("source1");
  seg1_hyp->SetChannel("1");
  Segment* seg2_hyp = Segment::CreateWithEndTime(1000, 2000, speech_hyp);
  seg2_hyp->SetSource("source1");
  seg2_hyp->SetChannel("1");
  Segment* seg3_hyp = Segment::CreateWithEndTime(2000, 3000, speech_hyp);
  seg3_hyp->SetSource("source1");
  seg3_hyp->SetChannel("1");
  Segment* seg4_hyp = Segment::CreateWithEndTime(3000, 4000, speech_hyp);
  seg4_hyp->SetSource("source1");
  seg4_hyp->SetChannel("1");
  Segment* seg5_hyp = Segment::CreateWithEndTime(4000, 5000, speech_hyp);
  seg5_hyp->SetSource("source1");
  seg5_hyp->SetChannel("1");

  speech1_ref->AddSegment(seg1_ref);
  speech2_ref->AddSegment(seg2_ref);
  speech_hyp->AddSegment(seg1_hyp);
  speech_hyp->AddSegment(seg2_hyp);
  speech_hyp->AddSegment(seg3_hyp);
  speech_hyp->AddSegment(seg4_hyp);
  speech_hyp->AddSegment(seg5_hyp);        
  speechSet_ref->AddSpeech(speech1_ref);
  speechSet_ref->AddSpeech(speech2_ref);
  speechSet_hyp->AddSpeech(speech_hyp);
  CTMSTMRTTMSegmentor* segmentor = new CTMSTMRTTMSegmentor();

  segmentor->Reset(speechSet_ref, speechSet_hyp);
  assert(segmentor->HasNext());
  SegmentsGroup* seg_group = segmentor->Next();
  assert(!segmentor->HasNext());
  assert(seg_group->GetNumberOfReferences() == 2);
  assert(seg_group->GetNumberOfHypothesis() == 1);
  assert(seg_group->GetReference(0).size() == 1);
  assert(seg_group->GetReference(1).size() == 1);
  assert(seg_group->GetHypothesis(0).size() == 5);
}

void CTMSTMRTTMSegmentorTest::testRefShorterThanHypCase()
{
  SpeechSet* speechSet_ref = new SpeechSet();
  SpeechSet* speechSet_hyp = new SpeechSet();
  Speech* speech1_ref = new Speech(speechSet_ref);
  Speech* speech2_ref = new Speech(speechSet_ref);
  Speech* speech_hyp = new Speech(speechSet_hyp);
  Segment* seg1_ref = Segment::CreateWithEndTime(0, 3000, speech1_ref);
  seg1_ref->SetSource("source1");
  seg1_ref->SetChannel("1");
  Segment* seg2_ref = Segment::CreateWithEndTime(3000, 4000, speech2_ref);
  seg2_ref->SetSource("source1");
  seg2_ref->SetChannel("1");
  Segment* seg1_hyp = Segment::CreateWithEndTime(0, 1500, speech_hyp);
  seg1_hyp->SetSource("source1");
  seg1_hyp->SetChannel("1");
  Segment* seg2_hyp = Segment::CreateWithEndTime(1500, 4500, speech_hyp);
  seg2_hyp->SetSource("source1");
  seg2_hyp->SetChannel("1");
  speech1_ref->AddSegment(seg1_ref);
  speech2_ref->AddSegment(seg2_ref);
  speech_hyp->AddSegment(seg1_hyp);
  speech_hyp->AddSegment(seg2_hyp);
  speechSet_ref->AddSpeech(speech1_ref);
  speechSet_ref->AddSpeech(speech2_ref);
  speechSet_hyp->AddSpeech(speech_hyp);
  CTMSTMRTTMSegmentor* segmentor = new CTMSTMRTTMSegmentor();
  
  segmentor->Reset(speechSet_ref, speechSet_hyp);
  assert(segmentor->HasNext());
  SegmentsGroup* seg_group = segmentor->Next();
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 1);
  assert(seg_group->GetReference(0).size() == 1);
  assert(seg_group->GetHypothesis(0).size() == 1);
  assert(segmentor->HasNext());
  seg_group = segmentor->Next();
  assert(seg_group->GetNumberOfReferences() == 1);
  assert(seg_group->GetNumberOfHypothesis() == 1);
  assert(seg_group->GetReference(0).size() == 1);
  assert(seg_group->GetHypothesis(0).size() == 1);
  assert(!segmentor->HasNext());
  
}

void CTMSTMRTTMSegmentorTest::testAll()
{
  cout << "- test BasicIteration" << endl;
  testBasicIteration();
  cout << "- test SimpleRefOverlapIteration" << endl;
  testSimpleRefOverlapIteration();
  cout << "- test Sequential segments that do not overlap" << endl;
  testSequentialSegments();
  cout << "- test Overlap with an empty segment" << endl;
  testEmptySegmentOverlapCaseIteration();
  cout << "- test Reference Shorter than the Hyp segment" << endl;
  testRefShorterThanHypCase();
}
