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

#include "test_segment.h" // class's header file
#include "segment.h"

// class constructor
TestSegment::TestSegment()
{
	// insert your code here
}

// class destructor
TestSegment::~TestSegment()
{
	// insert your code here
}

/**
 * Test the basics accessor to the 
 * objects info
 */
void TestSegment::testBasicAccessor()
{
	Segment* seg = Segment::CreateWithDuration(0, 0, NULL);
	seg->SetSpeakerId("bob");
	seg->SetChannel("A");
	seg->SetSource("test");
	assert(seg->GetSpeakerId() == "bob");
	assert(seg->GetChannel() == "A");
	assert(seg->GetSource() == "test");
	delete(seg);
}

/**
 * Test the is First Token methods behavior.
 * Test the is Last Token methods behavior.
 */
void TestSegment::testIsFirstOrLastTokenMethod()
{
  Segment* seg = Segment::CreateWithEndTime(0, 20000, NULL);
  Token* tok1 = Token::CreateWithEndTime(0, 4000, seg);
  Token* tok2 = Token::CreateWithEndTime(1000, 4500, seg);
  Token* tok3 = Token::CreateWithEndTime(3000, 5000, seg);
  
  Token* tok4 = Token::CreateWithEndTime(5000, 10000, seg);
  
  Token* tok5 = Token::CreateWithEndTime(10000, 19000, seg);
  Token* tok6 = Token::CreateWithEndTime(11000, 20000, seg);
  Token* tok7 = Token::CreateWithEndTime(10000, 18000, seg);
  
  seg->AddFirstToken(tok1);
  seg->AddFirstToken(tok2);
  seg->AddFirstToken(tok3);
  seg->AddLastToken(tok5);
  seg->AddLastToken(tok6);
  seg->AddLastToken(tok7);
  
  tok1->AddNextToken(tok4);
  tok2->AddNextToken(tok4);
  tok3->AddNextToken(tok4);
  tok4->AddPrecToken(tok1);
  tok4->AddPrecToken(tok2);
  tok4->AddPrecToken(tok3);
  
  tok5->AddPrecToken(tok4);
  tok6->AddPrecToken(tok4);
  tok7->AddPrecToken(tok4);
  tok4->AddNextToken(tok5);
  tok4->AddNextToken(tok6);
  tok4->AddNextToken(tok7);
  
  assert(seg->isFirstToken(tok1));
  assert(seg->isFirstToken(tok2));
  assert(seg->isFirstToken(tok3));
  
  assert(seg->isLastToken(tok5));
  assert(seg->isLastToken(tok6));
  assert(seg->isLastToken(tok7));
}

/**
 * Test merge two segments.
 */
void TestSegment::testMerge()
{
  Segment* seg1 = Segment::CreateWithEndTime(0, 5000, NULL);
  Segment* seg2 = Segment::CreateWithEndTime(5000, 10000, NULL);
  Segment* seg3 = Segment::CreateWithEndTime(10000, 15000, NULL);
  
  Token* tok1 = Token::CreateWithEndTime(0, 1000, seg1);
  Token* tok2 = Token::CreateWithEndTime(2000, 3000, seg1);
  Token* tok3 = Token::CreateWithEndTime(5000, 6000, seg2);
  Token* tok4 = Token::CreateWithEndTime(7000, 8000, seg2);
  Token* tok5 = Token::CreateWithEndTime(11000, 12000, seg3);
  Token* tok6 = Token::CreateWithEndTime(13000, 14000, seg3);
  
  seg1->AddFirstToken(tok1);
  seg1->AddLastToken(tok2);
  seg2->AddFirstToken(tok3);
  seg2->AddLastToken(tok4);
  seg3->AddFirstToken(tok5);
  seg3->AddLastToken(tok6);

  vector<Segment*> segs1;
  segs1.push_back(seg1);
  segs1.push_back(seg2);
  Segment* merged_seg1 = Segment::Merge(segs1);
  assert(merged_seg1->isFirstToken(tok1));
  assert(merged_seg1->isLastToken(tok4));

  vector<Segment*> segs2;
  segs2.push_back(seg1);
  segs2.push_back(seg2);
  segs2.push_back(seg3);
  Segment* merged_seg2 = Segment::Merge(segs2);
  assert(merged_seg2->isFirstToken(tok1));
  assert(merged_seg2->isLastToken(tok6));
}

/**
 * Test the output a plan version of the segment method.
 * TODO: Need to be more tough.
 */
void TestSegment::testToTopologicalOrderedStruct()
{
  Segment* seg = Segment::CreateWithEndTime(0, 20000, NULL);
  
  /* The following graph struct is tested.
   * ->1-3-4-5->
   *      / \
   *   ->2  6->
   */
  Token* tok1 = Token::CreateWithEndTime(0, 0, seg);
  Token* tok2 = Token::CreateWithEndTime(0, 0, seg);
  Token* tok3 = Token::CreateWithEndTime(0, 0, seg);
  Token* tok4 = Token::CreateWithEndTime(0, 0, seg);
  Token* tok5 = Token::CreateWithEndTime(0, 0, seg);
  Token* tok6 = Token::CreateWithEndTime(0, 0, seg);
  seg->AddFirstToken(tok1);
  seg->AddFirstToken(tok2);
  tok1->AddNextToken(tok3);
  tok2->AddNextToken(tok4);
  tok3->AddPrecToken(tok1);
  tok3->AddNextToken(tok4);
  tok4->AddPrecToken(tok2);
  tok4->AddPrecToken(tok3);
  tok4->AddNextToken(tok5);
  tok4->AddNextToken(tok6);
  tok5->AddPrecToken(tok4);
  tok6->AddPrecToken(tok4);
  seg->AddLastToken(tok5);
  seg->AddLastToken(tok6);
  
  vector<Token*> topo_seg = seg->ToTopologicalOrderedStruct();
  
  // the size is suppose to be +1 (empty token at first)
  assert(topo_seg.size() == 6);
  //TODO put some more here  
}

void TestSegment::testAll()
{
    cout << "- test BasicsAccessor" << endl;
    testBasicAccessor();
    cout << "- test 'Is First/Last Token ?'" << endl;
    testIsFirstOrLastTokenMethod();
    cout << "- test Merge a bunch of Segments" << endl;
    testMerge();
    cout << "- test To Topological structure" << endl;
		testToTopologicalOrderedStruct();
}
