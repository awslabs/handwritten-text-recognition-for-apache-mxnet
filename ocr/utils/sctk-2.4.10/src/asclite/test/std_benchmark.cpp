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

#include "std_benchmark.h" // class's header file
#include "speech.h"

const int StdBenchmark::REF_DIM = 1;
const int StdBenchmark::HYP_DIM = 0;

void StdBenchmark::CreateSimpleSegment(string text1, string text2, string text3, bool isRef)
{
	int dim = (isRef) ? REF_DIM : HYP_DIM;

	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, text1);
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, text2, t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, text3, t_tok1, NULL);  
	a_tok1->SetToken(dim, t_tok0);
	a_tok2->SetToken(dim, t_tok1);
	a_tok3->SetToken(dim, t_tok2);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);

	(isRef) ? g_segs->AddReference(t_seg) : g_segs->AddHypothesis(t_seg);
}

void StdBenchmark::CreateSimpleAlignment(string _ref1, string _ref2, string _ref3, string _hyp1, string _hyp2, string _hyp3)
{
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok1 = new GraphAlignedToken(2);
	a_tok2 = new GraphAlignedToken(2);
	a_tok3 = new GraphAlignedToken(2);
	CreateSimpleSegment(_ref1, _ref2, _ref3, true);
	CreateSimpleSegment(_hyp1, _hyp2, _hyp3, false);

	tests.push_back(g_segs);
}

// class constructor
StdBenchmark::StdBenchmark()
{
	SpeechSet* speechSet = new SpeechSet();
	speech = new Speech(speechSet);

	// 0: basic test
	// two exact same sentences (3 tokens long)
	// ref: a b c
	// hyp: a b c
	CreateSimpleAlignment("a", "b", "c", "a", "b", "c");
	props.clear();
	properties.push_back(props);
	costs["std"].push_back(0);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	results.push_back(t_ali_toks);

	// 1: basic test
	// one mistake in 2 sentences (3 tokens long)
	// ref: a f c
	// hyp: a b c
	CreateSimpleAlignment("a", "f", "c", "a", "b", "c");
	props.clear();
	properties.push_back(props);
	costs["std"].push_back(400);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	results.push_back(t_ali_toks);

	// 2: basic test
	// one insertion in 2 sentences (3 tokens long)
	// ref : a * c
	// hyp : a b c
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok1 = new GraphAlignedToken(2);
	a_tok2 = new GraphAlignedToken(2);
	a_tok3 = new GraphAlignedToken(2);

	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok0, NULL);
	a_tok1->SetToken(REF_DIM, t_tok0);
	a_tok2->SetToken(REF_DIM, NULL);
	a_tok3->SetToken(REF_DIM, t_tok1);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok1);
	g_segs->AddReference(t_seg);

	CreateSimpleSegment("a", "b", "c", false);

	tests.push_back(g_segs);

	properties.push_back(props);
	costs["std"].push_back(300);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	results.push_back(t_ali_toks);

	// 3: basic test
	// one deletion in 2 sentences (3 tokens long)
	// ref : a b c
	// hyp : a * c
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok1 = new GraphAlignedToken(2);
	a_tok2 = new GraphAlignedToken(2);
	a_tok3 = new GraphAlignedToken(2);

	CreateSimpleSegment("a", "b", "c", true);

	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok0, NULL);
	a_tok1->SetToken(HYP_DIM, t_tok0);
	a_tok2->SetToken(HYP_DIM, NULL);
	a_tok3->SetToken(HYP_DIM, t_tok1);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok1);
	g_segs->AddHypothesis(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);
	costs["std"].push_back(300);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	results.push_back(t_ali_toks);

	// 4: Basic test
	// All insertions of 3 tokens
	// ref: * * *
	// hyp: a b c
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok1 = new GraphAlignedToken(2);
	a_tok2 = new GraphAlignedToken(2);
	a_tok3 = new GraphAlignedToken(2);

	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	a_tok1->SetToken(REF_DIM, NULL);
	a_tok2->SetToken(REF_DIM, NULL);
	a_tok3->SetToken(REF_DIM, NULL);
	g_segs->AddReference(t_seg);

	CreateSimpleSegment("a", "b", "c", false);

	tests.push_back(g_segs);

	properties.push_back(props);
	costs["std"].push_back(900);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	results.push_back(t_ali_toks);


	// 5: Basic test
	// All deletions of 3 tokens
	// ref: a b c
	// hyp: * * *
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok1 = new GraphAlignedToken(2);
	a_tok2 = new GraphAlignedToken(2);
	a_tok3 = new GraphAlignedToken(2);

	CreateSimpleSegment("a", "b", "c", true);

	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	a_tok1->SetToken(HYP_DIM, NULL);
	a_tok2->SetToken(HYP_DIM, NULL);
	a_tok3->SetToken(HYP_DIM, NULL);
	g_segs->AddHypothesis(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);
	costs["std"].push_back(900);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	results.push_back(t_ali_toks);


	// 6: multi dimension align
	// multi dimension alignment of 5 tokens 2Hyp, 2Ref
	// Ref1 : a b c e f * * * * *
	// Ref2 : * * * * * a b c e f
	// Hyp1 : a b c e f * * * * *
	// Hyp2 : * * * * * a b c e f
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(2);
	a_tok0 = new GraphAlignedToken(4);
	a_tok1 = new GraphAlignedToken(4);
	a_tok2 = new GraphAlignedToken(4);
	a_tok3 = new GraphAlignedToken(4);
	a_tok4 = new GraphAlignedToken(4);
	a_tok5 = new GraphAlignedToken(4);
	a_tok6 = new GraphAlignedToken(4);
	a_tok7 = new GraphAlignedToken(4);
	a_tok8 = new GraphAlignedToken(4);
	a_tok9 = new GraphAlignedToken(4);

	// hyp: a b c e f * * * * *
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(2000, 3000, t_seg, "e", t_tok2, NULL);
	t_tok4 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok3, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok4);
	a_tok0->SetToken(0, t_tok0);
	a_tok1->SetToken(0, t_tok1);
	a_tok2->SetToken(0, t_tok2);
	a_tok3->SetToken(0, t_tok3);
	a_tok4->SetToken(0, t_tok4);
	a_tok5->SetToken(0, NULL);
	a_tok6->SetToken(0, NULL);
	a_tok7->SetToken(0, NULL);
	a_tok8->SetToken(0, NULL);
	a_tok9->SetToken(0, NULL);
	g_segs->AddHypothesis(t_seg);

	// hyp: * * * * * a b c e f
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(2000, 3000, t_seg, "e", t_tok2, NULL);
	t_tok4 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok3, NULL);
	a_tok0->SetToken(1, NULL);
	a_tok1->SetToken(1, NULL);
	a_tok2->SetToken(1, NULL);
	a_tok3->SetToken(1, NULL);
	a_tok4->SetToken(1, NULL);
	a_tok5->SetToken(1, t_tok0);
	a_tok6->SetToken(1, t_tok1);
	a_tok7->SetToken(1, t_tok2);
	a_tok8->SetToken(1, t_tok3);
	a_tok9->SetToken(1, t_tok4);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok4);
	g_segs->AddHypothesis(t_seg);

	// ref: a b c e f * * * * *
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(2000, 3000, t_seg, "e", t_tok2, NULL);
	t_tok4 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok3, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok4);
	a_tok0->SetToken(2, t_tok0);
	a_tok1->SetToken(2, t_tok1);
	a_tok2->SetToken(2, t_tok2);
	a_tok3->SetToken(2, t_tok3);
	a_tok4->SetToken(2, t_tok4);
	a_tok5->SetToken(2, NULL);
	a_tok6->SetToken(2, NULL);
	a_tok7->SetToken(2, NULL);
	a_tok8->SetToken(2, NULL);
	a_tok9->SetToken(2, NULL);
	g_segs->AddReference(t_seg);

	// ref: * * * * * a b c e f
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(2000, 3000, t_seg, "e", t_tok2, NULL);
	t_tok4 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok3, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok4);
	a_tok0->SetToken(3, NULL);
	a_tok1->SetToken(3, NULL);
	a_tok2->SetToken(3, NULL);
	a_tok3->SetToken(3, NULL);
	a_tok4->SetToken(3, NULL);
	a_tok5->SetToken(3, t_tok0);
	a_tok6->SetToken(3, t_tok1);
	a_tok7->SetToken(3, t_tok2);
	a_tok8->SetToken(3, t_tok3);
	a_tok9->SetToken(3, t_tok4);
	g_segs->AddReference(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);
	costs["std"].push_back(0);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok9);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok8);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok7);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok6);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok5);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok4);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok0); 
	results.push_back(t_ali_toks);

	// 7: multi dimension align 2
	// multi dimension alignment of 3 tokens 1Hyp, 2Ref
	// Ref1 : a b c
	// Ref2 : d e f
	// Hyp1 : a b c d e {c/f}
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok0 = new GraphAlignedToken(3);
	a_tok1 = new GraphAlignedToken(3);
	a_tok2 = new GraphAlignedToken(3);
	a_tok3 = new GraphAlignedToken(3);
	a_tok4 = new GraphAlignedToken(3);
	a_tok5 = new GraphAlignedToken(3);

	// hyp: a b c d e {c/f}
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(2000, 3000, t_seg, "d", t_tok2, NULL);
	t_tok4 = Token::CreateWithEndTime(2000, 3000, t_seg, "e", t_tok3, NULL);
	t_tok5 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok4, NULL);
	t_tok6 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok4, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok5);
	t_seg->AddLastToken(t_tok6);
	a_tok0->SetToken(0, t_tok0);
	a_tok1->SetToken(0, t_tok1);
	a_tok2->SetToken(0, t_tok2);
	a_tok3->SetToken(0, t_tok3);
	a_tok4->SetToken(0, t_tok4);
	a_tok5->SetToken(0, t_tok6);// <-- important f is align not c
	g_segs->AddHypothesis(t_seg);

	// Ref1 : a b c
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(1, t_tok0);
	a_tok1->SetToken(1, t_tok1);
	a_tok2->SetToken(1, t_tok2);
	a_tok3->SetToken(1, NULL);
	a_tok4->SetToken(1, NULL);
	a_tok5->SetToken(1, NULL);
	g_segs->AddReference(t_seg);

	// Ref2 : d e f
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "d");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "e", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(2, NULL);
	a_tok1->SetToken(2, NULL);
	a_tok2->SetToken(2, NULL);
	a_tok3->SetToken(2, t_tok0);
	a_tok4->SetToken(2, t_tok1);
	a_tok5->SetToken(2, t_tok2);
	g_segs->AddReference(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);	
	costs["std"].push_back(0);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok5);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok4);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok0);
	results.push_back(t_ali_toks);

	// 8: multi dimension align 3
	// multi dimension alignment of 3 tokens 1Hyp, 2Ref
	// Ref1 : a b c
	// Ref2 : c e f
	// Hyp1 : a b c d e {c/f}
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok0 = new GraphAlignedToken(3);
	a_tok1 = new GraphAlignedToken(3);
	a_tok2 = new GraphAlignedToken(3);
	a_tok3 = new GraphAlignedToken(3);
	a_tok4 = new GraphAlignedToken(3);
	a_tok5 = new GraphAlignedToken(3);

	// Hyp1 : a b c d e {c/f}
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(2000, 3000, t_seg, "d", t_tok2, NULL);
	t_tok4 = Token::CreateWithEndTime(2000, 3000, t_seg, "e", t_tok3, NULL);
	t_tok5 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok4, NULL);
	t_tok6 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok4, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok5);
	t_seg->AddLastToken(t_tok6);
	a_tok0->SetToken(0, t_tok0);
	a_tok1->SetToken(0, t_tok1);
	a_tok2->SetToken(0, t_tok2);
	a_tok3->SetToken(0, t_tok3);
	a_tok4->SetToken(0, t_tok4);
	a_tok5->SetToken(0, t_tok6);// <-- important f is align not c
	g_segs->AddHypothesis(t_seg);

	// Ref1 : a b c
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(1, t_tok0);
	a_tok1->SetToken(1, t_tok1);
	a_tok2->SetToken(1, t_tok2);
	a_tok3->SetToken(1, NULL);
	a_tok4->SetToken(1, NULL);
	a_tok5->SetToken(1, NULL);
	g_segs->AddReference(t_seg);

	// Ref2 : c e f
	t_seg = Segment::CreateWithEndTime(0, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "c");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "e", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "f", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(2, NULL);
	a_tok1->SetToken(2, NULL);
	a_tok2->SetToken(2, NULL);
	a_tok3->SetToken(2, t_tok0);
	a_tok4->SetToken(2, t_tok1);
	a_tok5->SetToken(2, t_tok2);
	g_segs->AddReference(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);	
	costs["std"].push_back(400);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok5);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok4);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok0);
	results.push_back(t_ali_toks); 

	// 9: multi dimension alignment designed to test the time-pruning option turned off
	// multi dimension alignment of 3 tokens 1Hyp, 2Ref
	// Ref1 : a b c
	// Ref2 :     d
	// Hyp1 : d a b c
	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok0 = new GraphAlignedToken(3);
	a_tok1 = new GraphAlignedToken(3);
	a_tok2 = new GraphAlignedToken(3);
	a_tok3 = new GraphAlignedToken(3);
	a_tok4 = new GraphAlignedToken(3);
	a_tok5 = new GraphAlignedToken(3);

	// Hyp1 : a b c 
	t_seg = Segment::CreateWithEndTime(0, 6000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "d");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "a", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "b", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(5000, 6000, t_seg, "c", t_tok2, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok3);
	a_tok0->SetToken(0, t_tok0);
	a_tok1->SetToken(0, t_tok1);
	a_tok2->SetToken(0, t_tok2);
	a_tok3->SetToken(0, t_tok3);
	g_segs->AddHypothesis(t_seg);

	// Ref1 : a b c
	t_seg = Segment::CreateWithEndTime(0, 3000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(1, NULL);
	a_tok1->SetToken(1, t_tok0);
	a_tok2->SetToken(1, t_tok1);
	a_tok3->SetToken(1, t_tok2);
	g_segs->AddReference(t_seg);

	// Ref2 : d
	t_seg = Segment::CreateWithEndTime(2000, 3000, speech);
	t_tok0 = Token::CreateWithEndTime(2000, 3000, t_seg, "d");
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok0);
	a_tok0->SetToken(2, t_tok0);
	a_tok1->SetToken(2, NULL);
	a_tok2->SetToken(2, NULL);
	a_tok3->SetToken(2, NULL);
	g_segs->AddReference(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);	
	costs["std"].push_back(0);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok0);
	results.push_back(t_ali_toks); 

	// 10: multi dimension alignment designed to test the time-pruning option turned ON
	// multi dimension alignment of 3 tokens 1Hyp, 2Ref
	// Ref1 : a b c
	// Ref2 :     d
	// Hyp1 : d a b c

	props.clear();
	props[string("align.timepruneoptimization")] = string("true");	
	props[string("align.timepruneoptimizationthreshold")] = string("0");
	props[string("align.timewordoptimization")] = string("true");	
	props[string("align.timewordoptimizationthreshold")] = string("0");
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok0 = new GraphAlignedToken(3);
	a_tok1 = new GraphAlignedToken(3);
	a_tok2 = new GraphAlignedToken(3);
	a_tok3 = new GraphAlignedToken(3);
	a_tok4 = new GraphAlignedToken(3);
	a_tok5 = new GraphAlignedToken(3);

	// Hyp1 : d a b c 
	t_seg = Segment::CreateWithEndTime(0, 6000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "d");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "a", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "b", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(5000, 6000, t_seg, "c", t_tok2, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok3);
	a_tok0->SetToken(0, t_tok0);
	a_tok1->SetToken(0, t_tok1);
	a_tok2->SetToken(0, t_tok2);
	a_tok3->SetToken(0, NULL);
	a_tok4->SetToken(0, t_tok3);
	g_segs->AddHypothesis(t_seg);

	// Ref1 : a b c
	t_seg = Segment::CreateWithEndTime(0, 3000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(1, NULL);
	a_tok1->SetToken(1, t_tok0);
	a_tok2->SetToken(1, t_tok1);
	a_tok3->SetToken(1, t_tok2);
	a_tok4->SetToken(1, NULL);
	g_segs->AddReference(t_seg);

	// Ref2 : d
	t_seg = Segment::CreateWithEndTime(2000, 6000, speech);
	t_tok0 = Token::CreateWithEndTime(2000, 6000, t_seg, "d");
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok0);
	a_tok0->SetToken(2, NULL);
	a_tok1->SetToken(2, NULL);
	a_tok2->SetToken(2, NULL);
	a_tok3->SetToken(2, NULL);
	a_tok4->SetToken(2, t_tok0);
	g_segs->AddReference(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);	
	costs["std"].push_back(1000);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok4);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok0);
	results.push_back(t_ali_toks); 

	// 11: multi dimension alignment designed to test the time-pruning option turned OFF
	// multi dimension alignment of 3 tokens 1Hyp, 2Ref
	// Ref1 : a b c
	// Ref2 :     d
	// Hyp1 : d a b c

	props.clear();
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(1);
	a_tok0 = new GraphAlignedToken(3);
	a_tok1 = new GraphAlignedToken(3);
	a_tok2 = new GraphAlignedToken(3);
	a_tok3 = new GraphAlignedToken(3);

	// Hyp1 : d a b c 
	t_seg = Segment::CreateWithEndTime(0, 6000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "d");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "a", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "b", t_tok1, NULL);
	t_tok3 = Token::CreateWithEndTime(5000, 6000, t_seg, "c", t_tok2, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok3);
	a_tok0->SetToken(0, t_tok0);
	a_tok1->SetToken(0, t_tok1);
	a_tok2->SetToken(0, t_tok2);
	a_tok3->SetToken(0, t_tok3);
	g_segs->AddHypothesis(t_seg);

	// Ref1 : a b c
	t_seg = Segment::CreateWithEndTime(0, 3000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "b", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(1, NULL);
	a_tok1->SetToken(1, t_tok0);
	a_tok2->SetToken(1, t_tok1);
	a_tok3->SetToken(1, t_tok2);
	g_segs->AddReference(t_seg);

	// Ref2 : d
	t_seg = Segment::CreateWithEndTime(2000, 6000, speech);
	t_tok0 = Token::CreateWithEndTime(2000, 6000, t_seg, "d");
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok0);
	a_tok0->SetToken(2, t_tok0);
	a_tok1->SetToken(2, NULL);
	a_tok2->SetToken(2, NULL);
	a_tok3->SetToken(2, NULL);
	g_segs->AddReference(t_seg);

	tests.push_back(g_segs);

	properties.push_back(props);	
	costs["std"].push_back(0);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok0);
	results.push_back(t_ali_toks); 
	
	// 12: multi dimension align 2 + Speaker Optimization
	// Ref1 <=> Hyp1 and Ref2 <=> Hyp2
	// Hyp1 : * a d b *
	// Hyp2 : d * * * c
	// Ref1 : * * d b *
	// Ref2 : * a * * c 
	// Costs: 3 1 0 0 0 = 4
	props.clear();
	props[string("align.speakeroptimization")] = string("true");
	props[string("dataDirectory")] = string("../testfiles");
	
	g_segs = new SegmentsGroup();
	t_ali_toks = new GraphAlignedSegment(2);
	a_tok0 = new GraphAlignedToken(4);
	a_tok1 = new GraphAlignedToken(4);
	a_tok2 = new GraphAlignedToken(4);
	a_tok3 = new GraphAlignedToken(4);
	a_tok4 = new GraphAlignedToken(4);
	
	// Hyp1 : * a d b *
	t_seg = Segment::CreateWithEndTime(0, 3000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(1000, 2000, t_seg, "d", t_tok0, NULL);
	t_tok2 = Token::CreateWithEndTime(2000, 3000, t_seg, "b", t_tok1, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok2);
	a_tok0->SetToken(0, NULL);
	a_tok1->SetToken(0, t_tok0);
	a_tok2->SetToken(0, t_tok1);
	a_tok3->SetToken(0, t_tok2);
	a_tok4->SetToken(0, NULL);
	t_seg->SetSource("FileA");
	t_seg->SetChannel("1");
	t_seg->SetSpeakerId("Speaker_alpha");
	g_segs->AddHypothesis(t_seg);
	
	// Hyp2 : d * * * c
	t_seg = Segment::CreateWithEndTime(1000, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(1000, 2000, t_seg, "d");
	t_tok1 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok0, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok1);
	a_tok0->SetToken(1, t_tok0);
	a_tok1->SetToken(1, NULL);
	a_tok2->SetToken(1, NULL);
	a_tok3->SetToken(1, NULL);
	a_tok4->SetToken(1, t_tok1);
	t_seg->SetSource("FileA");
	t_seg->SetChannel("1");
	t_seg->SetSpeakerId("Speaker_beta");
	g_segs->AddHypothesis(t_seg);
	
	// Ref1 : * * d b *
	t_seg = Segment::CreateWithEndTime(1000, 5000, speech);
	t_tok0 = Token::CreateWithEndTime(1000, 2000, t_seg, "d");
	t_tok1 = Token::CreateWithEndTime(2000, 3000, t_seg, "b", t_tok0, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok1);
	a_tok0->SetToken(2, NULL);
	a_tok1->SetToken(2, NULL);
	a_tok2->SetToken(2, t_tok0);
	a_tok3->SetToken(2, t_tok1);
	a_tok4->SetToken(2, NULL);
	t_seg->SetSource("FileA");
	t_seg->SetChannel("1");
	t_seg->SetSpeakerId("Speaker_gamma");
	g_segs->AddReference(t_seg);
	
	// Ref2 : * a * * c
	t_seg = Segment::CreateWithEndTime(0, 3000, speech);
	t_tok0 = Token::CreateWithEndTime(0, 1000, t_seg, "a");
	t_tok1 = Token::CreateWithEndTime(2000, 3000, t_seg, "c", t_tok0, NULL);
	t_seg->AddFirstToken(t_tok0);
	t_seg->AddLastToken(t_tok1);
	a_tok0->SetToken(3, NULL);
	a_tok1->SetToken(3, t_tok0);
	a_tok2->SetToken(3, NULL);
	a_tok3->SetToken(3, NULL);
	a_tok4->SetToken(3, t_tok1);
	t_seg->SetSource("FileA");
	t_seg->SetChannel("1");
	t_seg->SetSpeakerId("Speaker_delta");
	g_segs->AddReference(t_seg);
	
	tests.push_back(g_segs);

	properties.push_back(props);
	costs["std"].push_back(400);

	t_ali_toks->AddFrontGraphAlignedToken(a_tok4);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok3);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok2);
	t_ali_toks->AddFrontGraphAlignedToken(a_tok1);  	
	t_ali_toks->AddFrontGraphAlignedToken(a_tok0); 
	results.push_back(t_ali_toks);
}

// class destructor
StdBenchmark::~StdBenchmark()
{
	delete t_tok0;
	delete t_tok1;
	delete t_tok2;
	delete t_tok3;
	delete t_tok4;
	delete t_tok5;
	delete t_tok6;
	delete t_seg;
	delete g_segs;
	delete a_tok1;
	delete a_tok2;
	delete a_tok3;
	delete a_tok4;
	delete a_tok5;
	delete a_tok6;
	delete a_tok7;
	delete a_tok8;
	delete a_tok9;
	delete a_tok0;
	delete t_ali_toks;	
	delete speech;
}

/**
* Access the "type" cost no index
 */
int StdBenchmark::GetCost(int index, string type)
{
	return costs[type][index];
}
