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

#include "test_ctm_inputparser.h" // class's header file

// class constructor
CTMInputParserTest::CTMInputParserTest()
{
	Properties::SetProperty("align.optionally", "none");
	Properties::SetProperty("align.fragment_are_correct", "false");
}

// class destructor
CTMInputParserTest::~CTMInputParserTest()
{
	// insert your code here
}

/*
 * Launch all test
 */
void CTMInputParserTest::testAll()
{
	cout << "- test BasicImport" << endl;
  testBasicImport();
}

/*
 * Test the import of the Basic file
 */
void CTMInputParserTest::testBasicImport()
{
	CTMInputParser* parser = new CTMInputParser();
	SpeechSet* speechs = parser->loadFile(Properties::GetProperty("dataDirectory") + "/basic.ctm");
	assert(speechs->GetNumberOfSpeech() == 1);
	assert(speechs->GetSpeech(0)->NbOfSegments() == 19);	
	
	//test each words
	assert(speechs->GetSpeech(0)->GetSegment(0)->GetFirstToken(0)->GetText() == "JAVA");
	assert(speechs->GetSpeech(0)->GetSegment(1)->GetFirstToken(0)->GetText() == "WILL");
	assert(speechs->GetSpeech(0)->GetSegment(2)->GetFirstToken(0)->GetText() == "RULES");
	assert(speechs->GetSpeech(0)->GetSegment(3)->GetFirstToken(0)->GetText() == "THE");
	assert(speechs->GetSpeech(0)->GetSegment(4)->GetFirstToken(0)->GetText() == "WORLD");
	assert(speechs->GetSpeech(0)->GetSegment(5)->GetFirstToken(0)->GetText() == "OR");
	assert(speechs->GetSpeech(0)->GetSegment(6)->GetFirstToken(0)->GetText() == "NOT");
	assert(speechs->GetSpeech(0)->GetSegment(7)->GetFirstToken(0)->GetText() == "NICO");
	assert(speechs->GetSpeech(0)->GetSegment(8)->GetFirstToken(0)->GetText() == "IS");
	assert(speechs->GetSpeech(0)->GetSegment(9)->GetFirstToken(0)->GetText() == "HERE");
	Properties::SetProperty("align.optionally", "none");
	assert(speechs->GetSpeech(0)->GetSegment(10)->GetFirstToken(0)->GetText() == "(HUH)");
	assert(!speechs->GetSpeech(0)->GetSegment(10)->GetFirstToken(0)->IsOptional());
	Properties::SetProperty("align.optionally", "both");
	assert(speechs->GetSpeech(0)->GetSegment(10)->GetFirstToken(0)->GetText() == "HUH");
	assert(speechs->GetSpeech(0)->GetSegment(10)->GetFirstToken(0)->IsOptional());
	assert(speechs->GetSpeech(0)->GetSegment(11)->GetFirstToken(0)->GetText() == "start");
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetNumberOfFirstToken() == 3);
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetFirstToken(0)->GetText() == "begin");
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetFirstToken(1)->GetText() == "alt1");
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetFirstToken(2)->GetText() == "alt2");
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetNumberOfLastToken() == 3);
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetLastToken(0)->GetText() == "end");
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetLastToken(1)->GetText() == "alt1");
	assert(speechs->GetSpeech(0)->GetSegment(12)->GetLastToken(2)->GetText() == "alt2");
	assert(speechs->GetSpeech(0)->GetSegment(13)->GetFirstToken(0)->GetText() == "last");
	assert(speechs->GetSpeech(0)->GetSegment(14)->GetFirstToken(0)->GetText() == "verylast");
  
	assert(speechs->GetSpeech(0)->GetSegment(15)->GetFirstToken(0)->GetText() == "first");
	assert(speechs->GetSpeech(0)->GetSegment(15)->GetFirstToken(0)->GetNbOfNextTokens() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetNumberOfFirstToken() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetFirstToken(0)->GetText() == "A1_1");
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetFirstToken(1)->GetText() == "A2_1");
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetNumberOfLastToken() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetLastToken(0)->GetText() == "A1_2");
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetLastToken(0)->GetNbOfNextTokens() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetLastToken(1)->GetText() == "A2_2");
	assert(speechs->GetSpeech(0)->GetSegment(16)->GetLastToken(1)->GetNbOfNextTokens() == 2);

	assert(speechs->GetSpeech(0)->GetSegment(17)->GetNumberOfFirstToken() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(17)->GetFirstToken(0)->GetNbOfPrecTokens() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(17)->GetFirstToken(0)->GetText() == "A3_1");
	assert(speechs->GetSpeech(0)->GetSegment(17)->GetFirstToken(1)->GetNbOfPrecTokens() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(17)->GetFirstToken(1)->GetText() == "A4_1");
	assert(speechs->GetSpeech(0)->GetSegment(17)->GetNumberOfLastToken() == 2);
	assert(speechs->GetSpeech(0)->GetSegment(17)->GetLastToken(0)->GetText() == "A3_2");
	assert(speechs->GetSpeech(0)->GetSegment(17)->GetLastToken(1)->GetText() == "A4_2");

	assert(speechs->GetSpeech(0)->GetSegment(18)->GetFirstToken(0)->GetText() == "last");
	delete speechs;
	delete parser;
}
