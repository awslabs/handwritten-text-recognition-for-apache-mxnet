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

#include "tokenalignment_test.h"

TokenAlignmentTest::TokenAlignmentTest() {
  speechSet = new SpeechSet();
	speech = new Speech(speechSet);
	segment = Segment::CreateWithDuration(0, 0, speech);
	m_ref = Token::CreateWithDuration(0, 0, segment, "dog");
	m_hyp1 = Token::CreateWithDuration(0, 0, segment, "dog1");	
	m_hyp2 = Token::CreateWithDuration(0, 0, segment);
	m_hyp3 = Token::CreateWithDuration(0, 0, segment, "dog3");
	m_tokenAlignment = new TokenAlignment(m_ref);
}

TokenAlignmentTest::~TokenAlignmentTest() {
	delete m_ref;
	delete m_hyp1;
	delete m_hyp2;
	delete m_hyp3;
	delete m_tokenAlignment;
	delete segment;
	delete speech;
	delete speechSet;
}

void TokenAlignmentTest::TestAll()
{
	cout << "- test AddAlignment" << endl;
	TestAddAlignment();
	cout << "- test NullRef" << endl;
	TestNullRef();
	cout << "- test NullHyp" << endl;
	TestNullHyp();
}

void TokenAlignmentTest::TestNullRef() {
	TokenAlignment* ta = new TokenAlignment(NULL);
	assert(ta->AddAlignmentFor("hyp1", m_hyp1));
	assert(ta->GetReferenceToken() == NULL);
	assert(ta->GetAlignmentFor("hyp1")->GetToken() == m_hyp1);
	delete ta;
}

void TokenAlignmentTest::TestNullHyp() {
	TokenAlignment* ta = new TokenAlignment(m_ref);
	assert(ta->AddAlignmentFor("hyp1", NULL));
	assert(ta->GetReferenceToken() == m_ref);
	assert(ta->GetAlignmentFor("hyp1")->GetToken() == NULL);
	delete ta;
}

void TokenAlignmentTest::TestAddAlignment() {
	string hyp1 = "hyp1";
	assert(m_tokenAlignment->AddAlignmentFor(hyp1, m_hyp1) > 0); // insertion worked
	assert(m_tokenAlignment->AddAlignmentFor(hyp1, m_hyp1) == 0); // already inserted should return 0
	assert(m_tokenAlignment->GetAlignmentFor(hyp1)->Equals(new TokenAlignment::AlignmentEvaluation(m_hyp1, TokenAlignment::UNAVAILABLE))); // alignement for hyp1 
	m_tokenAlignment->GetAlignmentFor(hyp1)->SetResult(TokenAlignment::SUBSTITUTION);
	// assert(m_tokenAlignment->GetAlignmentFor(NULL) == NULL); // What about NULL???
	assert(m_tokenAlignment->GetAlignmentFor("bogus") == NULL);
	assert(m_tokenAlignment->GetResultFor(hyp1) == TokenAlignment::SUBSTITUTION);
	assert(m_tokenAlignment->GetResultFor("bogus") == TokenAlignment::INVALID_SYSTEM);
}
