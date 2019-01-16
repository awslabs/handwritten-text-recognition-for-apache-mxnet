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
 
#ifndef TEST_UNIT
#define TEST_UNIT 

#include "stdinc.h"
#include "asctools.h"
#include "test_token.h"
#include "test_segment.h"
#include "test_speech.h"
#include "test_ctm_inputparser.h"
#include "test_stm_inputparser.h"
#include "test_trn_inputparser.h"
#include "test_rttm_inputparser.h"
#include "test_ctmstm_segmentor.h"
#include "test_trntrn_segmentor.h"
#include "test_graphalignedtoken.h"
#include "test_graphalignedsegment.h"
#include "test_graph.h"
#include "test_speechset.h"
#include "tokenalignment_test.h"
#include "test_levenshtein.h"
#include "test_properties.h"
#include "perf_benchmark.h"
#include "alignment_test.h"
#include "alignedsegment_test.h"
#include "alignedspeech_test.h"

void UnitTest();
void RunBenchmark(string bench_name, int repeat);
void die_usage(string message);

map<string, Benchmark*> benchmarks;

int main(int argc, char** argv)
{    
  bool bench_mode = false;
  int repeat = 1;
  vector<string> args;
  Properties::Initialize();
  Properties::SetProperty("dataDirectory","../testfiles");
  
  for(int i=1 ; i < argc ; i++)
  {
    if (string(argv[i]).compare("-b") == 0 || string(argv[i]).compare("--bench") == 0)
    {
      bench_mode = true;
    } 
    else if(string(argv[i]).compare("-r") == 0 || string(argv[i]).compare("--repeat") == 0)
    {
      if (i+1 < argc)
      {
        repeat = atoi(argv[i+1]);
        if (repeat == -1)
        {
          die_usage("repeat_time need to be an integer");
        }
      } 
      else
      {
        die_usage("-r need an integer as argument");
      }
      i++;
    } 
    else if(string(argv[i]).compare("-d") == 0 || string(argv[i]).compare("--datadir") == 0)
    {
      if (i+1 < argc)
      {
         Properties::SetProperty("dataDirectory",argv[i+1]);
      } 
      else
      {
        die_usage("-d needs a directory name");
      }
      i++;
    } 
    else
    {
      args.push_back(string(argv[i]));
    }
  }

  if (bench_mode)
  {
    benchmarks["std"] = new StdBenchmark;
    benchmarks["perf"] = new PerfBenchmark;
    for (uint i=0 ; i < args.size() ; i++)
    {
      RunBenchmark(args[i], repeat);
    }
  } 
  else
  {
    UnitTest();
  }

	#ifdef WIN32
	cout << "Press Enter to continue." << endl;
	getchar();
	#endif
		
    return 0;
}
void UnitTest()
{
	cout << "  +-------------------------------+" << endl;
	cout << "  | Start unit tests on asclite ! |" << endl;
	cout << "  +-------------------------------+" << endl;

	//test GraphAlignedToken
	cout << "Start test GraphAlignedToken..." << endl;
	GraphAlignedTokenTest* t_gat = new GraphAlignedTokenTest;
	t_gat->testAll();
	cout << "test GraphAlignedToken pass without failure" << endl;
	cout << endl;
	//delete t_gat;

	//test Token
	cout << "Start test Token..." << endl;
	TestToken* t_token = new TestToken;
	t_token->TestAll();
	cout << "test Token pass without failure" << endl;
	cout << endl;
	//delete t_token;

	//test Segment
	cout << "Start test Segment..." << endl;
	TestSegment* t_segment = new TestSegment;
	t_segment->testAll();
	cout << "test Segment pass without failure" << endl;
	cout << endl;
	//delete t_segment;

	//test Speech
	cout << "Start test Speech..." << endl;
	TestSpeech* t_speech = new TestSpeech;
	t_speech->testAll();
	cout << "test Speech pass without failure" << endl;
	cout << endl;
	//delete t_speech;

	//test SpeechSet
	cout << "Start test SpeechSet..." << endl;
	SpeechSetTest* t_speechSet = new SpeechSetTest;
	t_speechSet->testAll();
	cout << "test SpeechSet pass without failure" << endl;
	cout << endl;
	//delete t_speechSet;

	//test CTM importer
	cout << "Start test CTM Importer..." << endl;
	CTMInputParserTest* t_ctmparser = new CTMInputParserTest;
	t_ctmparser->testAll();
	cout << "test CTM Importer pass without failure" << endl;
	cout << endl;
	//delete t_ctmparser;

	//test STM importer
	cout << "Start test STM Importer..." << endl;
	STMInputParserTest* t_stmparser = new STMInputParserTest;
	t_stmparser->testAll();
	cout << "test STM Importer pass without failure" << endl;
	cout << endl;
	//delete t_stmparser;

	//test TRN importer
	cout << "Start test TRN Importer..." << endl;
	TRNInputParserTest* t_trnparser = new TRNInputParserTest;
	t_trnparser->testAll();
	cout << "test TRN Importer pass without failure" << endl;
	cout << endl;
	//delete t_trnparser;

	//test RTTM importer
	cout << "Start test RTTM Importer..." << endl;
	RTTMInputParserTest* t_rttmparser = new RTTMInputParserTest;
	t_rttmparser->testAll();
	cout << "test RTTM Importer pass without failure" << endl;
	cout << endl;
	//delete t_rttmparser;

	//test CTMSTMSegmentor
	cout << "Start test CTMSTMRTTMSegmentor..." << endl;
	CTMSTMRTTMSegmentorTest* t_ctm_stm_segmentor = new CTMSTMRTTMSegmentorTest;
	t_ctm_stm_segmentor->testAll();
	cout << "test CTMSTMRTTMSegmentor pass without failure" << endl;
	cout << endl;
	//delete t_ctm_stm_segmentor;

	//test TRNTRNSegmentor
	cout << "Start test TRNTRNSegmentor..." << endl;
	TRNTRNSegmentorTest* t_trn_trn_segmentor = new TRNTRNSegmentorTest;
	t_trn_trn_segmentor->testAll();
	cout << "test TRNTRNSegmentor pass without failure" << endl;
	cout << endl;
	//delete t_trn_trn_segmentor;

	//test GraphAlignedSegment
	cout << "Start test GraphAlignedSegment..." << endl;
	GraphAlignedSegmentTest* t_gas = new GraphAlignedSegmentTest;
	t_gas->testAll();
	cout << "test GraphAlignedSegment pass without failure" << endl;
	cout << endl;
	//delete t_gas;

	//test Graph
	cout << "Starting test Graph..." << endl;
	TestGraph* pTestGraph = new TestGraph;
	if(pTestGraph->TestAll())
	cout << "Test Graph: Completed!" << endl << endl;
	else
	cout << "Test Graph: FAILED!" << endl << endl;
	//delete pTestGraph;

	//test TokenAlignment
	cout << "Starting TokenAlignment tests..." << endl;
	TokenAlignmentTest* pTokenAlignmentTest = new TokenAlignmentTest;
	pTokenAlignmentTest->TestAll();
	cout << "TokenAlignment tests completed without failure!" << endl << endl;
	//delete pTokenAlignmentTest;

	//test AlignedSegment
	cout << "Starting AlignedSegment tests..." << endl;
	AlignedSegmentTest* pAlignedSegmentTest = new AlignedSegmentTest;
	pAlignedSegmentTest->TestAll();
	cout << "AlignedSegment tests completed without failure!" << endl << endl;
	//delete pAlignedSegmentTest;

	// test AlignedSpeech
	cout << "Starting AlignedSpeech tests..." << endl;
	AlignedSpeechTest* pAlignedSpeechTest = new AlignedSpeechTest;
	pAlignedSpeechTest->TestAll();
	cout << "AlignedSpeech tests completed without failure!" << endl << endl;
	//delete pAlignedSpeechTest;

	//test Alignment
	cout << "Starting Alignment tests..." << endl;
	AlignmentTest* pAlignmentTest = new AlignmentTest;
	pAlignmentTest->TestAll();
	cout << "Alignment tests completed without failure!" << endl << endl;		
	//delete pAlignmentTest;

	//test Properties
	cout << "Starting test Properties..." << endl;
	PropertiesTest* t_prop = new PropertiesTest;
	t_prop->testAll();
	cout << "test Properties pass without failure" << endl;
	cout << endl;
	//delete t_prop;

	//test Levenshtein
	cout << "Starting test Levenshtein..." << endl;
	LevenshteinTest* t_lev = new LevenshteinTest;
	t_lev->TestAll();
	cout << "test Levenshtein pass without failure" << endl;
	cout << endl;
	//delete t_lev;
}

void RunBenchmark(string bench_name, int repeat)
{
	double time_before, duration;
	Benchmark* bench = benchmarks[bench_name];
	
	if (!bench)
	{
		cout << "Benchmark : " << bench_name << " doesn't exist !!!" << endl;
		exit(-1);
	}
	
	double bench_before = timerStart();

	printf("+--------------------------------------------+\n");  
	printf("| Benchmark : %10s                     |\n", bench_name.c_str());
	
	time_before = timerStart();
	Levenshtein* laligner = new Levenshtein();
	duration = timerEnd(time_before);
	
	printf("| Levenshtein creation: %2.3f ms             |\n", duration);
	printf("| Repetition factor   : %5d                |\n", repeat);
	printf("+---------+----------+------------+----------+\n");
	printf("| test    | segments | alignement | results  |\n");
	printf("+---------+----------+------------+----------+\n");
  
	for (int i=0 ; i < bench->GetTestSize() - 1 ; i++)
	{
		double t_segs, d_segs, t_align, d_align, t_res, d_res;
		d_segs = d_align = d_res = 0.0;
			
		for (int j=0 ; j < repeat ; j++)
		{
			Properties::SetProperties(bench->GetProperties(i));
			t_segs = timerStart();
			laligner->SetSegments(bench->GetTest(i), NULL, false);
			d_segs += timerEnd(t_segs);
			t_align = timerStart();
			laligner->Align();
			d_align += timerEnd(t_align);
			t_res = timerStart();
			laligner->GetResults();
			d_res += timerEnd(t_res);
		}
		
		printf("| test %2d | %2.3f ms |  %2.3f ms  | %2.3f ms |\n", i, d_segs/repeat, d_align/repeat, d_res/repeat);
  }
	
	printf("+---------+----------+------------+----------+\n");
	printf("| total   | %2.3f ms                         |\n", timerEnd(bench_before)/repeat);
	printf("+---------+----------------------------------+\n");
	cout << endl;
}

void die_usage(string message)
{
  cout << "Usage: asclite_test [-b|--bench bench_name] [-r|--repeat repeat_time] [-d|--datadir directoryNameOfTestFiles] [bench_name ...]" << endl;
  
  cout << "       -d Default is '../testfiles'" << endl;
  cout << endl;
  cout << "  " << message << endl;
  exit(-1);
}

#endif
