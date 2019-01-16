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

#include "test_token.h" // class's header file
#include "speech.h"
#include "segment.h"

// class constructor
TestToken::TestToken()
{
	// insert your code here
}

// class destructor
TestToken::~TestToken()
{
	// insert your code here
}

void TestToken::TestText()
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* segment = Segment::CreateWithDuration(0, 10000, speech);
	Token* token = Token::CreateWithDuration(0, 0, segment);
	token->SetSourceText("hop");
	assert(token->GetText() == "hop");

	//JGF Note: I only added the full test cases for optional frags for the SetProperty("align.optionally", "both") test
	Properties::SetProperty("align.optionally", "both");
		// (hop)
	token->SetSourceText("(hop)");
	assert(token->GetSourceText() == "(hop)");
	assert(token->GetText() == "hop");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
	assert(token->IsOptional());
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "hop");
		// hop-
	Properties::SetProperty("align.fragment_are_correct", "false");
	token->SetSourceText("hop-");
	assert(token->GetSourceText() == "hop-");
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	assert(token->GetFragmentStatus() == Token::BEGIN_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "hop-");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
		// (hop-)
	Properties::SetProperty("align.fragment_are_correct", "false");
	token->SetSourceText("(hop-)");
	assert(token->GetSourceText() == "(hop-)");
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	assert(token->GetFragmentStatus() == Token::BEGIN_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "hop-");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
		// -hop
	Properties::SetProperty("align.fragment_are_correct", "false");
	token->SetSourceText("-hop");
	assert(token->GetSourceText() == "-hop");
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	assert(token->GetFragmentStatus() == Token::END_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "-hop");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
		// (-hop)
	Properties::SetProperty("align.fragment_are_correct", "false");
	token->SetSourceText("(-hop)");
	assert(token->GetSourceText() == "(-hop)");
	Properties::SetProperty("align.fragment_are_correct", "true");
		cout << "Get " << token->GetText() << endl;
	assert(token->GetText() == "hop");
	assert(token->GetFragmentStatus() == Token::END_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "-hop");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);

	Properties::SetProperty("align.optionally", "ref");
	speechSet->SetOrigin("ref");
	token->SetSourceText("(hop)");
	assert(token->GetSourceText() == "(hop)");
	assert(token->GetText() == "hop");
	assert(token->IsOptional());
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "hop");
	token->SetSourceText("(hop-)");
	assert(token->GetSourceText() == "(hop-)");
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	assert(token->GetFragmentStatus() == Token::BEGIN_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "hop-");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);

	Properties::SetProperty("align.optionally", "ref");
	speechSet->SetOrigin("hyp");
	token->SetSourceText("(hop)");
	assert(token->GetSourceText() == "(hop)");
	assert(token->GetText() == "(hop)");
	assert(!token->IsOptional());
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "(hop)");
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "(hop)");
	token->SetSourceText("(hop-)");
	assert(token->GetSourceText() == "(hop-)");
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "(hop-)");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "(hop-)");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);

	Properties::SetProperty("align.optionally", "hyp");
	speechSet->SetOrigin("hyp");
	token->SetSourceText("(hop)");
	assert(token->GetSourceText() == "(hop)");
	assert(token->GetText() == "hop");
	assert(token->IsOptional());
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "hop");
	token->SetSourceText("(hop-)");
	assert(token->GetSourceText() == "(hop-)");
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "hop");
	assert(token->GetFragmentStatus() == Token::BEGIN_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "hop-");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);

	Properties::SetProperty("align.optionally", "hyp");
	speechSet->SetOrigin("ref");
	token->SetSourceText("(hop)");
	assert(token->GetSourceText() == "(hop)");
	assert(token->GetText() == "(hop)");
	assert(!token->IsOptional());
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "(hop)");
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "(hop)");
	token->SetSourceText("(hop-)");
	assert(token->GetSourceText() == "(hop-)");
	Properties::SetProperty("align.fragment_are_correct", "true");
	assert(token->GetText() == "(hop-)");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);
	Properties::SetProperty("align.fragment_are_correct", "false");
	assert(token->GetText() == "(hop-)");
	assert(token->GetFragmentStatus() == Token::NOT_FRAGMENT);

	delete token;
	delete segment;
	delete speech;
	delete speechSet;
}

void TestToken::TestTime()
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* segment = Segment::CreateWithDuration(0, 10000, speech);
	Token* token = Token::CreateWithEndTime(25460, 26460, segment); //build with (start, end)
	assert((token->GetDuration() - 1000) < 10);
	delete token;

	token = Token::CreateWithDuration(25460, 2000, segment); //build with (start, duration)
	assert((token->GetEndTime() - 27460) < 10);
	 
	delete token;
	delete segment;
	delete speech;
	delete speechSet;
}

void TestToken::TestPrecNextGraph()
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* segment = Segment::CreateWithDuration(0, 10000, speech);
	/* initialize tokens with a graph like
	*     B
	*    / \
	* --A   D--
	*    \ /
	*     C
	*/
	Token* tokenA = Token::CreateWithDuration(0, 0, segment);
	tokenA->SetSourceText("A");
	Token* tokenB = Token::CreateWithDuration(0, 0, segment);
	tokenB->SetSourceText("B");
	Token* tokenC = Token::CreateWithDuration(0, 0, segment);
	tokenC->SetSourceText("C");
	Token* tokenD = Token::CreateWithDuration(0, 0, segment);
	tokenD->SetSourceText("D");
	tokenA->AddNextToken(tokenB);
	tokenA->AddNextToken(tokenC);
	tokenB->AddPrecToken(tokenA);
	tokenB->AddNextToken(tokenD);
	tokenC->AddPrecToken(tokenA);
	tokenC->AddNextToken(tokenD);
	tokenD->AddPrecToken(tokenB);
	tokenD->AddPrecToken(tokenC);     

	// a few basics assert
	assert(tokenA->GetNextToken(0)->GetNextToken(0) == tokenA->GetNextToken(1)->GetNextToken(0));
	assert(tokenD->GetPrecToken(0)->GetPrecToken(0) == tokenA->GetPrecToken(1)->GetPrecToken(0));

	//TODO need to be more complex
	delete tokenA;
	delete tokenB;
	delete tokenC;
	delete tokenD;
	delete segment;
	delete speech;
}

void TestToken::TestOverlap()
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* segment = Segment::CreateWithDuration(0, 10000, speech);
	Token* t1 = Token::CreateWithEndTime(4000, 10000, segment, "first");
	Token* t2 = Token::CreateWithEndTime(4000, 11000, segment, "second");
	Token* t3 = Token::CreateWithEndTime(5000, 10000, segment, "third");
	Token* t4 = Token::CreateWithEndTime(6000, 9000, segment, "fourth");
	Token* t5 = Token::CreateWithEndTime(12000, 15000, segment, "fifth");
	Token* t6 = Token::CreateWithEndTime(10000, 15000, segment, "sixth");

	assert(t1->OverlapWith(t1));
	assert(t1->OverlapWith(t2));
	assert(t2->OverlapWith(t1));
	assert(t1->OverlapWith(t3));
	assert(t3->OverlapWith(t1));
	assert(t1->OverlapWith(t4));
	assert(t4->OverlapWith(t1));
	assert(!t1->OverlapWith(t5));
	assert(!t5->OverlapWith(t1));
	assert(t1->OverlapWith(t6));
	assert(t6->OverlapWith(t1));

	delete t1;
	delete t2;
	delete t3;
	delete t4;
	delete t5;
	delete t6;
	delete segment;
	delete speech;
	delete speechSet;
}

void TestToken::TestEquals()
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* segment = Segment::CreateWithDuration(0, 10000, speech);
	Token* t1 = Token::CreateWithEndTime(4000, 10000, segment, "first");
	Token* t2 = Token::CreateWithEndTime(4000, 10000, segment, "first");
	Token* t3 = Token::CreateWithEndTime(5000, 10000, segment, "first");
	Token* t4 = Token::CreateWithEndTime(4000, 10000, segment, "FIRST");
	Token* t5 = Token::CreateWithEndTime(4000, 10000, segment, "-----");
	assert(t1->Equals(t2));
	assert(!t1->Equals(t3));
	assert(!t1->Equals(t4));
	assert(!t1->Equals(t5));
	delete t1;
	delete t2;
	delete t3;
	delete t4;
	delete t5;
	delete segment;
	delete speech;
	delete speechSet;
}

void TestToken::TestIsEquivalentTo()
{
  SpeechSet* speechSet = new SpeechSet();
  Speech* speech = new Speech(speechSet);
  Segment* segment = Segment::CreateWithDuration(0, 10000, speech);
  Properties::SetProperty("align.optionally", "both");
  Properties::SetProperty("align.fragment_are_correct", "true");
  Token* t1 = Token::CreateWithEndTime(4000, 10000, segment, "first");
  Token* t2 = Token::CreateWithEndTime(5000, 10000, segment, "(first)");
  Token* t3 = Token::CreateWithEndTime(4000, 10000, segment, "fi-");
  Token* t4 = Token::CreateWithEndTime(4000, 10000, segment, "(fi-)");
  Token* t5 = Token::CreateWithEndTime(4000, 10000, segment, "-st");
  Token* t6 = Token::CreateWithEndTime(4000, 10000, segment, "(-st)");
  Token* t7 = Token::CreateWithEndTime(5000, 10000, segment, "(a-)");
  Token* t8 = Token::CreateWithEndTime(5000, 10000, segment, "that");
  assert( t1->IsEquivalentTo(t1));
  assert( t1->IsEquivalentTo(t2));
  assert( t1->IsEquivalentTo(t3));
  assert( t1->IsEquivalentTo(t4));
  assert( t1->IsEquivalentTo(t5));
  assert( t1->IsEquivalentTo(t6));
  assert(!t1->IsEquivalentTo(t7));
  assert(!t1->IsEquivalentTo(t8));

  assert( t2->IsEquivalentTo(t1));
  assert( t2->IsEquivalentTo(t2));
  assert( t2->IsEquivalentTo(t3));
  assert( t2->IsEquivalentTo(t4));
  assert( t2->IsEquivalentTo(t5));
  assert( t2->IsEquivalentTo(t6));
  assert(!t2->IsEquivalentTo(t7));
  assert(!t2->IsEquivalentTo(t8));

  assert( t3->IsEquivalentTo(t1));
  assert( t3->IsEquivalentTo(t2));
  assert( t3->IsEquivalentTo(t3));
  assert( t3->IsEquivalentTo(t4));
  assert(!t3->IsEquivalentTo(t5));
  assert(!t3->IsEquivalentTo(t6));
  assert(!t3->IsEquivalentTo(t7));
  assert(!t3->IsEquivalentTo(t8));

  assert( t4->IsEquivalentTo(t1));
  assert( t4->IsEquivalentTo(t2));
  assert( t4->IsEquivalentTo(t3));
  assert( t4->IsEquivalentTo(t4));
  assert(!t4->IsEquivalentTo(t5));
  assert(!t4->IsEquivalentTo(t6));
  assert(!t4->IsEquivalentTo(t7));
  assert(!t4->IsEquivalentTo(t8));

  assert( t5->IsEquivalentTo(t1));
  assert( t5->IsEquivalentTo(t2));
  assert(!t5->IsEquivalentTo(t3));
  assert(!t5->IsEquivalentTo(t4));
  assert( t5->IsEquivalentTo(t5));
  assert( t5->IsEquivalentTo(t6));
  assert(!t5->IsEquivalentTo(t7));
  assert(!t5->IsEquivalentTo(t8));

  assert( t6->IsEquivalentTo(t1));
  assert( t6->IsEquivalentTo(t2));
  assert(!t6->IsEquivalentTo(t3));
  assert(!t6->IsEquivalentTo(t4));
  assert( t6->IsEquivalentTo(t5));
  assert( t6->IsEquivalentTo(t6));
  assert(!t6->IsEquivalentTo(t7));
  assert(!t6->IsEquivalentTo(t8));

  assert(!t7->IsEquivalentTo(t1));
  assert(!t7->IsEquivalentTo(t2));
  assert(!t7->IsEquivalentTo(t3));
  assert(!t7->IsEquivalentTo(t4));
  assert(!t7->IsEquivalentTo(t5));
  assert(!t7->IsEquivalentTo(t6));
  assert( t7->IsEquivalentTo(t7));
  assert(!t7->IsEquivalentTo(t8));

  assert(!t8->IsEquivalentTo(t1));
  assert(!t8->IsEquivalentTo(t2));
  assert(!t8->IsEquivalentTo(t3));
  assert(!t8->IsEquivalentTo(t4));
  assert(!t8->IsEquivalentTo(t5));
  assert(!t8->IsEquivalentTo(t6));
  assert(!t8->IsEquivalentTo(t7));
  assert( t8->IsEquivalentTo(t8));


  delete t1;
  delete t2;
  delete t3;
  delete t4;
  delete t5;
  delete t6;
  delete t7;
  delete t8;
  delete segment;
  delete speech;
  delete speechSet;
}

void TestToken::TestAll()
{
  cout << "- test Text" << endl;
  TestText();
  cout << "- test Time" << endl;
  TestTime();
  cout << "- test Overlap" << endl;
  TestOverlap();
  cout << "- test Equals" << endl;
  TestEquals();
  cout << "- test IsEquivalentTo" << endl;
  TestIsEquivalentTo();
  //cout << "- test PrecNextGraph" << endl;
  //TestPrecNextGraph();     
}
