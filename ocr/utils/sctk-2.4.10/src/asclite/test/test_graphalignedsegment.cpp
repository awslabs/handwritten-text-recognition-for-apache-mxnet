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

#include "test_graphalignedsegment.h" // class's header file
#include "speech.h"
#include "segment.h"

// class constructor
GraphAlignedSegmentTest::GraphAlignedSegmentTest()
{
	// insert your code here
}

// class destructor
GraphAlignedSegmentTest::~GraphAlignedSegmentTest()
{
	// insert your code here
}

/*
 * Launch all the tests on GraphAlignedSegmentTest
 */
void GraphAlignedSegmentTest::testAll()
{
	cout << "- test Operator(==) " << endl;
  testEqualOperator();
}

/*
 * Test the Equal operator behavior.
 */
void GraphAlignedSegmentTest::testEqualOperator()
{
  SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* segment = Segment::CreateWithEndTime(0, 10000, speech);
	Token* t1 = Token::CreateWithEndTime(4000, 10000, segment, "first");
  Token* t2 = Token::CreateWithEndTime(4000, 10000, segment, "first");
  Token* t3 = Token::CreateWithEndTime(4000, 10000, segment, "second");
  Token* t4 = Token::CreateWithEndTime(4000, 10000, segment, "bis");
	GraphAlignedToken* gat1 = new GraphAlignedToken(2);
	GraphAlignedToken* gat2 = new GraphAlignedToken(2);
	GraphAlignedToken* gat3 = new GraphAlignedToken(2);
  GraphAlignedToken* gat4 = new GraphAlignedToken(2);
  gat1->SetToken(0, t1);
  gat1->SetToken(1, t3);
  gat2->SetToken(0, t1);
  gat2->SetToken(1, t3);
  gat3->SetToken(0, t2);
  gat3->SetToken(1, t3);
  gat4->SetToken(0, t1);
  gat4->SetToken(1, t4);
  GraphAlignedSegment* gas1 = new GraphAlignedSegment(1);
  GraphAlignedSegment* gas2 = new GraphAlignedSegment(1);
  GraphAlignedSegment* gas3 = new GraphAlignedSegment(1);
  GraphAlignedSegment* gas4 = new GraphAlignedSegment(0);
  gas1->AddFrontGraphAlignedToken(gat1);
  gas1->AddFrontGraphAlignedToken(gat4);
  gas2->AddFrontGraphAlignedToken(gat1);
  gas2->AddFrontGraphAlignedToken(gat4);
  gas3->AddFrontGraphAlignedToken(gat2);
  gas3->AddFrontGraphAlignedToken(gat4);
  gas4->AddFrontGraphAlignedToken(gat1);
  gas4->AddFrontGraphAlignedToken(gat3);
  assert(*gas1 == *gas2);
  assert(*gas1 == *gas3);
  assert(!(*gas1 == *gas4));
	/*
	delete gas1;
	delete gas2;
	delete gas3;
	delete gas4;
	*/
	delete speechSet;
}
