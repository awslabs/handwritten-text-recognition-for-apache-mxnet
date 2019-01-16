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

#ifndef STDBENCHMARK_H
#define STDBENCHMARK_H

#include "benchmark.h"
#include "speechset.h"

/**
 * This class build some classic test and results,
 * the Advanced Leveshtein algo should produce.
 */
class StdBenchmark : public Benchmark
{
	public:
		// class constructor
		StdBenchmark();
		// class destructor
		~StdBenchmark();
		/**
		 * Access the standard cost no index
		 */
		int GetCost(int index, string type);
		
	private:
    /**
     * Contain the collection of min cost
     * for the various cost methods
     */
    map<string, vector<int> > costs;
	Token* t_tok0;
	Token* t_tok1;
	Token* t_tok2;
	Token* t_tok3;
	Token* t_tok4;
	Token* t_tok5;
	Token* t_tok6;
	Segment* t_seg;
	Speech* speech;
	SegmentsGroup* g_segs;
	GraphAlignedToken* a_tok1;
	GraphAlignedToken* a_tok2;
	GraphAlignedToken* a_tok3;
	GraphAlignedToken* a_tok4;
	GraphAlignedToken* a_tok5;
	GraphAlignedToken* a_tok6;
	GraphAlignedToken* a_tok7;
	GraphAlignedToken* a_tok8;
	GraphAlignedToken* a_tok9;
	GraphAlignedToken* a_tok0;
	GraphAlignedSegment* t_ali_toks;
	map<string, string> props;

	static const int REF_DIM;
	static const int HYP_DIM;

	void CreateSimpleSegment(string text1, string text2, string text3, bool isRef);
	void CreateSimpleAlignment(string ref1, string ref2, string ref3, string hyp1, string hyp2, string hyp3);
};

#endif // STDBENCHMARK_H
