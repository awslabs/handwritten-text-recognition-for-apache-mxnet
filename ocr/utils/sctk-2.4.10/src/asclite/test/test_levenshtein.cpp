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

#include "test_levenshtein.h" // class's header file

// class constructor
LevenshteinTest::LevenshteinTest()
{
	Properties::SetProperty("align.optionally", "none");
}

// class destructor
LevenshteinTest::~LevenshteinTest()
{
	// insert your code here
}

void LevenshteinTest::TestGAS()
{
	StdBenchmark* bench = new StdBenchmark();
	Levenshtein* laligner = new Levenshtein();
	
	// hyp: a b c
	// ref: a * c
	Properties::SetProperties(bench->GetProperties(2));
	laligner->SetSegments(bench->GetTest(2), NULL, false);
	laligner->Align();
	GraphAlignedSegment* res = laligner->GetResults();
	cout << "Testing single insertion... ";
	cout.flush();
	assert(res->GetNonNullReference(1) == NULL);
	assert(res->GetPreviousNonNullReference(1)->GetText() == "a");
	assert(res->GetPreviousNonNullReference(0) == NULL);
	assert(res->GetNextNonNullReference(1)->GetText() == "c");
	assert(res->GetNextNonNullReference(2) == NULL);
	assert(res->GetNextNonNullReference(6) == NULL);
	cout << "OK." << endl;
	
	// hyp: a b c
	// ref: * * *
	Properties::SetProperties(bench->GetProperties(4));
	laligner->SetSegments(bench->GetTest(4), NULL, false);
	laligner->Align();
	res = laligner->GetResults();
	cout << "Testing insertions only... ";
	assert(res->GetNextNonNullReference(1) == NULL);
	assert(res->GetPreviousNonNullReference(1) == NULL);
	assert(res->GetNextNonNullReference(1) == NULL);
	cout << "OK." << endl;
	
	// hyp: a b c
	// ref: a * c
	Properties::SetProperties(bench->GetProperties(2));
	laligner->SetSegments(bench->GetTest(2), NULL, true);
	laligner->Align();
	res = laligner->GetResults();
	cout << "Testing single insertion Compressed... ";
	cout.flush();
	assert(res->GetNonNullReference(1) == NULL);
	assert(res->GetPreviousNonNullReference(1)->GetText() == "a");
	assert(res->GetPreviousNonNullReference(0) == NULL);
	assert(res->GetNextNonNullReference(1)->GetText() == "c");
	assert(res->GetNextNonNullReference(2) == NULL);
	assert(res->GetNextNonNullReference(6) == NULL);
	cout << "OK." << endl;
	
	// hyp: a b c
	// ref: * * *
	Properties::SetProperties(bench->GetProperties(4));
	laligner->SetSegments(bench->GetTest(4), NULL, true);
	laligner->Align();
	res = laligner->GetResults();
	cout << "Testing insertions only Compressed... ";
	cout.flush();
	assert(res->GetNextNonNullReference(1) == NULL);
	assert(res->GetPreviousNonNullReference(1) == NULL);
	assert(res->GetNextNonNullReference(1) == NULL);
	cout << "OK." << endl;
}

void LevenshteinTest::TestBasicBenchmark()
{
	StdBenchmark* bench = new StdBenchmark();
	Levenshtein* laligner = new Levenshtein();

	for (int i=0 ; i < bench->GetTestSize() ; ++i)
	{
		cout << "    * test " << i << " : ";
		Properties::SetProperties(bench->GetProperties(i));

		if(i == 12)
		{
			SpeakerMatch* pSpeakerMatch = new SpeakerMatch;
			pSpeakerMatch->LoadFile(Properties::GetProperty("dataDirectory") + "/test12.mdalign.csv");
			laligner->SetSegments(bench->GetTest(i), pSpeakerMatch, false);
		}
		else
		{
			laligner->SetSegments(bench->GetTest(i), NULL, false);
		}
			
		laligner->Align();
		GraphAlignedSegment* res = laligner->GetResults();
		bool ok_cost = (laligner->GetCost() == bench->GetCost(i, "std"));
		bool ok_align = (*res == *(bench->GetResult(i)));

		if (ok_cost && ok_align)
		{
			cout << "OK" << endl;
		}
		else
		{
			cout << "Failed. Cost ";

			if (ok_cost)
			{
				cout << "OK";
			}
			else
			{
				cout << " expected " << bench->GetCost(i, "std") << ", got " << laligner->GetCost();
			}
				
			cout << ", Alignement ";
			
			if (ok_align)
			{
				cout << "OK" << endl;
			} 
			else
			{
				cout << "Failed" << endl;
				cout << "Wanted:" << endl;
				cout << bench->GetResult(i)->ToString() << endl;
				cout << "Obtained:" << endl;
				cout << res->ToString() << endl;
			}
		}
	}
}

void LevenshteinTest::TestAll()
{
	cout << "- Test GAS:" << endl;
	TestGAS();
	cout << endl;

  cout << "- Test basic benchmark:" << endl;
  TestBasicBenchmark();
	cout << endl;
}
