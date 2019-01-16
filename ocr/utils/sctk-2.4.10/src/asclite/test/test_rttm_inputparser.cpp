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
 
#include "test_rttm_inputparser.h" // class's header file

// class constructor
RTTMInputParserTest::RTTMInputParserTest()
{
	Properties::SetProperty("align.optionally", "none");
	Properties::SetProperty("align.fragment_are_correct", "false");
}

// class destructor
RTTMInputParserTest::~RTTMInputParserTest()
{
	// insert your code here
}

/*
 * Launch all test
 */
void RTTMInputParserTest::testAll()
{
	cout << "- test BasicImport" << endl;
    testBasicImport();
	testAlternations();
}

/*
 * Test the import of the Basic file
 */
void RTTMInputParserTest::testBasicImport()
{
	RTTMInputParser* parser = new RTTMInputParser();
	SpeechSet* speechs = parser->loadFile(Properties::GetProperty("dataDirectory") + "/rttmstm.rttm");
	
	assert(speechs->GetNumberOfSpeech() == 2);
	
	assert(speechs->GetSpeech(0)->NbOfSegments() == 1);
    
    assert(speechs->GetSpeech(0)->GetSegment(0)->GetSource() == "ALPHA");
    assert(speechs->GetSpeech(0)->GetSegment(0)->GetSpeakerId() == "OMEGA");
	
	assert(speechs->GetSpeech(0)->GetSegment(0)->ToTopologicalOrderedStruct().size() == 3);
	
    assert(speechs->GetSpeech(0)->GetSegment(0)->ToTopologicalOrderedStruct()[0]->GetText() == "A");
	assert(speechs->GetSpeech(0)->GetSegment(0)->ToTopologicalOrderedStruct()[1]->GetText() == "B");
	assert(speechs->GetSpeech(0)->GetSegment(0)->ToTopologicalOrderedStruct()[2]->GetText() == "C");
    
	assert(speechs->GetSpeech(0)->GetSegment(0)->GetFirstToken(0)->GetText() == "A");
	assert(speechs->GetSpeech(0)->GetSegment(0)->GetLastToken(0)->GetText() == "C");
	
    assert(speechs->GetSpeech(1)->NbOfSegments() == 1);
	
	assert(speechs->GetSpeech(1)->GetSegment(0)->GetSource() == "ALPHA");
    assert(speechs->GetSpeech(1)->GetSegment(0)->GetSpeakerId() == "THETA");
	
	assert(speechs->GetSpeech(1)->GetSegment(0)->ToTopologicalOrderedStruct()[0]->GetText() == "D");
	assert(speechs->GetSpeech(1)->GetSegment(0)->ToTopologicalOrderedStruct()[1]->GetText() == "E");
	assert(speechs->GetSpeech(1)->GetSegment(0)->ToTopologicalOrderedStruct()[2]->GetText() == "F");
	assert(speechs->GetSpeech(1)->GetSegment(0)->ToTopologicalOrderedStruct()[3]->GetText() == "G");
	assert(speechs->GetSpeech(1)->GetSegment(0)->ToTopologicalOrderedStruct()[4]->GetText() == "H");
	assert(speechs->GetSpeech(1)->GetSegment(0)->ToTopologicalOrderedStruct()[5]->GetText() == "I");
	
	assert(speechs->GetSpeech(1)->GetSegment(0)->GetFirstToken(0)->GetText() == "D");
	assert(speechs->GetSpeech(1)->GetSegment(0)->GetLastToken(0)->GetText() == "I");

	delete speechs;
	delete parser;
}

/*
 *   Test the handling of embedded alternations
 */
void RTTMInputParserTest::testAlternations()
{
	RTTMInputParser* parser = new RTTMInputParser();
	SpeechSet* speechs = parser->loadFile(Properties::GetProperty("dataDirectory") + "/rttmAlt.rttm");

//    cout << speechs->GetSpeech(0)->ToString() << endl;
//    cout << speechs->GetSpeech(1)->ToString() << endl;
	
	vector <Token *>toks = speechs->GetSpeech(0)->GetSegment(0)->ToTopologicalOrderedStruct();
	assert(toks.size() == 3);
	assert(toks[0]->GetText().compare("A") == 0 );
	assert(toks[1]->GetText().compare("B") == 0 );
	assert(toks[2]->GetText().compare("B2") == 0 );
	toks.clear();
	
	toks = speechs->GetSpeech(0)->GetSegment(1)->ToTopologicalOrderedStruct();
	assert(toks.size() == 2);
	assert(toks[0]->GetText().compare("S") == 0 );
	assert(toks[1]->GetText().compare("T") == 0 );
	toks.clear();

	toks = speechs->GetSpeech(1)->GetSegment(0)->ToTopologicalOrderedStruct();
	assert(toks.size() == 9);
	assert(toks[0]->GetText().compare("D") == 0 );
	assert(toks[1]->GetText().compare("E") == 0 );
	assert(toks[2]->GetText().compare("F") == 0 );
	assert(toks[3]->GetText().compare("G") == 0 );
	assert(toks[4]->GetText().compare("O") == 0 );
	assert(toks[5]->GetText().compare("GG") == 0 );
	assert(toks[6]->GetText().compare("G2") == 0 );
	assert(toks[7]->GetText().compare("DD") == 0 );
	assert(toks[8]->GetText().compare("D2") == 0 );
		
	toks = speechs->GetSpeech(1)->GetSegment(1)->ToTopologicalOrderedStruct();
	assert(toks.size() == 12);
	assert(toks[0]->GetText().compare("P") == 0 );
	assert(toks[1]->GetText().compare("J") == 0 );
	assert(toks[2]->GetText().compare("Q") == 0 );
	assert(toks[3]->GetText().compare("L") == 0 );
	assert(toks[4]->GetText().compare("M") == 0 );
	assert(toks[5]->GetText().compare("R") == 0 );
	assert(toks[6]->GetText().compare("MM") == 0 );
	assert(toks[7]->GetText().compare("M2") == 0 );
	assert(toks[8]->GetText().compare("LL") == 0 );
	assert(toks[9]->GetText().compare("L2") == 0 );
	assert(toks[10]->GetText().compare("JJ") == 0 );
	assert(toks[11]->GetText().compare("J2") == 0 );
		
	toks.clear();
}
