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

#include "perf_benchmark.h" // class's header file
#include "speech.h"

// class destructor
PerfBenchmark::~PerfBenchmark()
{
	map<string, vector<int> >::iterator i, ei;

	i = costs.begin();
	ei = costs.end();

	while(i != ei)
	{
		i->second.clear();
		i++;
	}

	costs.clear();
	props.clear();
}

// class constructor
PerfBenchmark::PerfBenchmark()
{
	// Common variables
	Segment* t_seg;
	SegmentsGroup* t_segs;

	//1) Simple test in 2 dimention but with large segment
	// two differents same sentences (3 tokens long)
	// ref : a b ....
	// hyp : a c ....
	t_segs = new SegmentsGroup();
	t_seg = CreateSampleSegment(200, string("b"));
	t_segs->AddReference(t_seg);

	t_seg = CreateSampleSegment(200, string("c"));
	t_segs->AddHypothesis(t_seg);

	tests.push_back(t_segs);
	props.clear();
	properties.push_back(props);
	costs["std"].push_back(0);

	//2) Simple test in 3 dimention but with large segment
	// 3 differents sentences
	// ref : a b ....
	// hyp : a c ....
	// hyp : a d ....
	t_segs = new SegmentsGroup();
	t_seg = CreateSampleSegment(100, string("b"));
	t_segs->AddReference(t_seg);

	t_seg = CreateSampleSegment(50, string("c"));
	t_segs->AddHypothesis(t_seg);

	t_seg = CreateSampleSegment(50, string("d"));
	t_segs->AddHypothesis(t_seg);

	tests.push_back(t_segs);
	props.clear();
	properties.push_back(props);
	costs["std"].push_back(0);	

	//3) Simple test in 4 dimention but with large segment
	// two exact same sentences (3 tokens long)
	// ref : a b b b b ... 50x ... b b b b b b b b b b b b b b b
	// hyp : a c c c c c c c c c c c c c c c c c c c
	// hyp : a d d d d d d d d d d d d d d d d d d d
	// hyp : a e e e e e e e e e
	// hyp : a f
	// hyp : a g
	t_segs = new SegmentsGroup();
	t_seg = CreateSampleSegment(50, string("b"));
	t_segs->AddReference(t_seg);

	t_seg = CreateSampleSegment(20, string("c"));
	t_segs->AddHypothesis(t_seg);

	t_seg = CreateSampleSegment(20, string("d"));
	t_segs->AddHypothesis(t_seg);

	t_seg = CreateSampleSegment(10, string("e"));
	t_segs->AddHypothesis(t_seg);

	t_seg = CreateSampleSegment(2, string("f"));
	t_segs->AddHypothesis(t_seg);

	t_seg = CreateSampleSegment(2, string("g"));
	t_segs->AddHypothesis(t_seg);

	tests.push_back(t_segs);
	props.clear();
	properties.push_back(props);
	costs["std"].push_back(0);	
}

/**
 * Access the "type" cost no index
 */
int PerfBenchmark::GetCost(int index, string type)
{
	return costs[type][index];
}

Segment* PerfBenchmark::CreateSampleSegment(int nb, string hop)
{
	SpeechSet* speechSet = new SpeechSet();
	Speech* speech = new Speech(speechSet);
	Segment* t_seg = Segment::CreateWithEndTime(0, nb*1000, speech);
	Token* f_tok = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	Token* p_tok = f_tok;
	Token* t_tok;
	
	for (int i=1 ; i < nb ; i++)
	{
		t_tok = Token::CreateWithEndTime(1000*i, 1000*i + 1000, t_seg, hop, p_tok, NULL);
		p_tok = t_tok;
	}
	
	t_seg->AddFirstToken(f_tok);
	t_seg->AddLastToken(p_tok);
	return t_seg;
}
