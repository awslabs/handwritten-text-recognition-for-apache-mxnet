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

#include "test_graphalignedtoken.h" // class's header file
#include "speech.h"
#include "segment.h"

// class constructor
GraphAlignedTokenTest::GraphAlignedTokenTest()
{
	// insert your code here
}

// class destructor
GraphAlignedTokenTest::~GraphAlignedTokenTest()
{
	// insert your code here
}

/*
 * Launch all the test methods on GraphAlignedToken
 */
void GraphAlignedTokenTest::testAll()
{
	cout << "- test Operator(==) " << endl;
  testEqualOperator();
}

/*
 * Test the behavior of the Equal operator on GraphAlignedToken
 */
void GraphAlignedTokenTest::testEqualOperator()
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* segment = Segment::CreateWithDuration(0, 10000, speech);
	Token* t1 = Token::CreateWithEndTime(4000, 10000, segment, "first");
	Token* t2 = Token::CreateWithEndTime(4000, 10000, segment, "first");
	Token* t3 = Token::CreateWithEndTime(4000, 10000, segment, "second");
	Token* t4 = Token::CreateWithEndTime(4000, 10000, segment, "bis");
	GraphAlignedToken* gat1 = new GraphAlignedToken(2);
	GraphAlignedToken* gat2 = new GraphAlignedToken(2);
	GraphAlignedToken* gat3 = new GraphAlignedToken(2);
	GraphAlignedToken* gat4 = new GraphAlignedToken(2);

	GraphAlignedToken* gat5 = new GraphAlignedToken(2);
	GraphAlignedToken* gat6 = new GraphAlignedToken(2);
	GraphAlignedToken* gat7 = new GraphAlignedToken(2);
		
	gat1->SetToken(0, t1);
	gat1->SetToken(1, t3);
	gat2->SetToken(0, t1);
	gat2->SetToken(1, t3);
	gat3->SetToken(0, t2);
	gat3->SetToken(1, t3);
	gat4->SetToken(0, t1);
	gat4->SetToken(1, t4);

	gat5->SetToken(0, t1);
	gat5->SetToken(1, NULL);
	gat6->SetToken(0, t1);
	gat6->SetToken(1, NULL);
	gat7->SetToken(0, NULL);
	gat7->SetToken(1, NULL);

	assert(*gat1 == *gat2);
	assert(*gat1 == *gat3);
	assert(*gat1 != *gat4); 

	assert(*gat1 != *gat5);
	assert(*gat5 != *gat1);
		
	assert(*gat5 == *gat6);
	assert(*gat6 != *gat7);

	delete gat1;
	delete gat2;
	delete gat3;
	delete gat4;
	delete gat5;
	delete gat6;
	delete speechSet;
}
