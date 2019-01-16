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

#include "test_trn_inputparser.h" // class's header file

// class constructor
TRNInputParserTest::TRNInputParserTest()
{
	Properties::SetProperty("inputparser.trn.uid", "spu_id");
	Properties::SetProperty("align.optionally", "both");
	Properties::SetProperty("align.fragment_are_correct", "true");
}

// class destructor
TRNInputParserTest::~TRNInputParserTest()
{
	// insert your code here
}

/*
 * Launch all the tests on the TRN input parser
 */
void TRNInputParserTest::testAll()
{
	cout << "- test BasicImport" << endl;
  testBasicImport();
}

/*
 * Test the import of the Basic trn file.
 */
void TRNInputParserTest::testBasicImport()
{
	TRNInputParser* parser = new TRNInputParser();
	Token* tok;
	Token* tok2;
	Token* tok3;
	Token* tok4;
	SpeechSet* speechs = parser->loadFile(Properties::GetProperty("dataDirectory") + "/basic.trn");
	assert(speechs->GetNumberOfSpeech() == 3);
	assert(speechs->GetSpeech(0)->NbOfSegments() == 1);
	assert(speechs->GetSpeech(1)->NbOfSegments() == 1);
	assert(speechs->GetSpeech(2)->NbOfSegments() == 1);

	//seg1
	assert(string(speechs->GetSpeech(0)->GetSegment(0)->GetId()).compare("(hop_806)") == 0);
	assert(string(speechs->GetSpeech(0)->GetSegment(0)->GetSpeakerId()).compare("hop") == 0);
	assert(speechs->GetSpeech(0)->GetSegment(0)->GetNumberOfFirstToken() == 4);
	tok = speechs->GetSpeech(0)->GetSegment(0)->GetFirstToken(0);
	tok2 = speechs->GetSpeech(0)->GetSegment(0)->GetFirstToken(1);
	tok3 = speechs->GetSpeech(0)->GetSegment(0)->GetFirstToken(2);
	tok4 = speechs->GetSpeech(0)->GetSegment(0)->GetFirstToken(3);
	assert(string(tok->GetText()).compare("I'm") == 0);
	assert(string(tok2->GetText()).compare("am") == 0);
	assert(string(tok3->GetText()).compare("huh") == 0);
	assert(tok3->GetFragmentStatus());
	assert(tok3->IsOptional());
	assert(string(tok4->GetText()).compare("happy") == 0);
	assert(tok->GetNbOfNextTokens() == 1);
	assert(tok2->GetNbOfNextTokens() == 1);
	assert(tok3->GetNbOfNextTokens() == 1);
	assert(tok4->GetNbOfNextTokens() == 1);
	assert(tok->GetNextToken(0) == tok2->GetNextToken(0));
	assert(tok->GetNextToken(0) == tok3->GetNextToken(0));
	assert(tok->GetNextToken(0) == tok4);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("happy") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("today") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("?") == 0);

	//seg2
	assert(string(speechs->GetSpeech(1)->GetSegment(0)->GetId()).compare("(paf_405)") == 0);
	assert(string(speechs->GetSpeech(1)->GetSegment(0)->GetSpeakerId()).compare("paf") == 0);
	assert(speechs->GetSpeech(1)->GetSegment(0)->GetNumberOfFirstToken() == 1);
	tok = speechs->GetSpeech(1)->GetSegment(0)->GetFirstToken(0);
	assert(string(tok->GetText()).compare("I") == 0);
	assert(tok->GetNbOfNextTokens() == 2);
	tok2 = tok->GetNextToken(0);
	assert(string(tok2->GetText()).compare("wasn't") == 0);
	tok = tok->GetNextToken(1);
	assert(string(tok->GetText()).compare("was") == 0);
	assert(tok->GetNbOfNextTokens() == 1);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("not") == 0);
	assert(tok2->GetNbOfNextTokens() == 1);
	assert(tok->GetNbOfNextTokens() == 1);
	assert(tok->GetNextToken(0) == tok2->GetNextToken(0));
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("aware") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("of") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("this") == 0);

	//seg3
	assert(string(speechs->GetSpeech(2)->GetSegment(0)->GetId()).compare("(sep_005)") == 0);
	assert(string(speechs->GetSpeech(2)->GetSegment(0)->GetSpeakerId()).compare("sep") == 0);
	assert(speechs->GetSpeech(2)->GetSegment(0)->GetNumberOfFirstToken() == 3);
	assert(speechs->GetSpeech(2)->GetSegment(0)->GetNumberOfLastToken() == 3);
	tok = speechs->GetSpeech(2)->GetSegment(0)->GetFirstToken(0);
	tok2 = speechs->GetSpeech(2)->GetSegment(0)->GetFirstToken(1);
	tok3 = speechs->GetSpeech(2)->GetSegment(0)->GetFirstToken(2);
	assert(string(tok->GetText()).compare("a") == 0);
	assert(string(tok2->GetText()).compare("ah") == 0);
	assert(string(tok3->GetText()).compare("separe") == 0);
	assert(tok->GetNextToken(0) == tok2->GetNextToken(0));
	assert(tok->GetNextToken(0) == tok3);
	assert(tok3->GetNbOfNextTokens() == 3);
	tok = tok3->GetNextToken(0);
	tok2 = tok3->GetNextToken(1);
	tok3 = tok3->GetNextToken(2);
	assert(string(tok->GetText()).compare("o") == 0);
	assert(string(tok2->GetText()).compare("oh") == 0);
	assert(string(tok3->GetText()).compare("separe") == 0);
	assert(tok->GetNextToken(0) == tok2->GetNextToken(0));
	assert(tok->GetNextToken(0) == tok3);
	assert(speechs->GetSpeech(2)->GetSegment(0)->isLastToken(tok3));
	assert(tok3->GetNbOfNextTokens() == 2);
	tok = tok3->GetNextToken(0);
	tok2 = tok3->GetNextToken(1);
	assert(string(tok->GetText()).compare("end") == 0);
	assert(string(tok2->GetText()).compare("end") == 0);
	assert(speechs->GetSpeech(2)->GetSegment(0)->isLastToken(tok));
	assert(tok2->GetNbOfNextTokens() == 1);
	tok = tok2->GetNextToken(0);
	assert(string(tok->GetText()).compare("too") == 0);
	assert(speechs->GetSpeech(2)->GetSegment(0)->isLastToken(tok));
}
