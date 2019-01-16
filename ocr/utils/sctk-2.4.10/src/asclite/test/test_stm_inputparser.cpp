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

#include "test_stm_inputparser.h" // class's header file

// class constructor
STMInputParserTest::STMInputParserTest()
{
	// insert your code here
}

// class destructor
STMInputParserTest::~STMInputParserTest()
{
	// insert your code here
}

/*
 * Launch all the test of STM input parser
 */
void STMInputParserTest::testAll()
{
	cout << "- test BasicImport" << endl;
  testBasicImport();
}

/*
 * Test the import of the Basic file
 */
void STMInputParserTest::testBasicImport()
{
	STMInputParser* parser = new STMInputParser();
	Token* tok;
	SpeechSet* speechs = parser->loadFile(Properties::GetProperty("dataDirectory") + "/basic.stm");
	assert(speechs->GetNumberOfSpeech() == 4);
	
	//seg1
	assert(speechs->GetSpeech(0)->NbOfSegments() == 1);
	assert(string(speechs->GetSpeech(0)->GetSegment(0)->GetSpeakerId()).compare("CHRIS") == 0);
	assert(speechs->GetSpeech(0)->GetSegment(0)->GetNumberOfFirstToken() == 1);
	tok = speechs->GetSpeech(0)->GetSegment(0)->GetFirstToken(0);
  
	assert(string(tok->GetText()).compare("JAVA") == 0);
	//cout << tok->GetNbOfNextTokens() << endl;
	assert(tok->GetNbOfNextTokens() == 1);
	tok = tok->GetNextToken(0);
	//cout << tok->GetText() << endl;
	assert(string(tok->GetText()).compare("WILL") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("RULES") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("THE") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("WORLD") == 0);
	
	//seg2
	assert(speechs->GetSpeech(1)->NbOfSegments() == 1);
	assert(string(speechs->GetSpeech(1)->GetSegment(0)->GetSpeakerId()).compare("JEROME") == 0);
	assert(speechs->GetSpeech(1)->GetSegment(0)->GetNumberOfFirstToken() == 1);
	tok = speechs->GetSpeech(1)->GetSegment(0)->GetFirstToken(0);
	assert(string(tok->GetText()).compare("OR") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("NOT") == 0);
	
	//seg3 -- JON is before NICO in alphabetical order (even if NICO's segment is before JON in STM)
	assert(speechs->GetSpeech(2)->NbOfSegments() == 1);
	assert(string(speechs->GetSpeech(2)->GetSegment(0)->GetSpeakerId()).compare("JON") == 0);
	assert(speechs->GetSpeech(2)->GetSegment(0)->isEmpty());
	assert(speechs->GetSpeech(2)->GetSegment(0)->GetNumberOfFirstToken() == 0);	
	
	//seg4
	assert(speechs->GetSpeech(3)->NbOfSegments() == 1);
	assert(string(speechs->GetSpeech(3)->GetSegment(0)->GetSpeakerId()).compare("NICO") == 0);
	assert(speechs->GetSpeech(3)->GetSegment(0)->GetNumberOfFirstToken() == 1);
	tok = speechs->GetSpeech(3)->GetSegment(0)->GetFirstToken(0);
	assert(string(tok->GetText()).compare("NICO") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("IS") == 0);
	tok = tok->GetNextToken(0);
	assert(string(tok->GetText()).compare("HERE") == 0);
}
