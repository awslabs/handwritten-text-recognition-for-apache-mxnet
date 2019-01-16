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

#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "stdinc.h"
#include "segment.h"
#include "graphalignedsegment.h"
#include "segmentsgroup.h"

/**
 * This class represent an abstract benchmark structure and 
 * methods to access the test collections.
 * 
 * @author <a href="mailto:nradde@nist.gov">Nicolas Radde</a>
 * @todo Solve the exposure problem of GraphAligneSegment
 */
class Benchmark
{
	public:
		// class constructor
		Benchmark();
		// class destructor
		~Benchmark();
		/**
		 * Access the tests no index
		 */
		SegmentsGroup* GetTest(int index);
		/**
		 * Access the properties no index
		 */
		map<string, string> GetProperties(int index);
		/**
		 * Access the results no index
		 */
		GraphAlignedSegment* GetResult(int index);
		/**
		 * Get the number of tests
		 */
		int GetTestSize();
		
	protected:
    /**
     * Contain the collection of tests as
     * a vector of vector<Segments*>
     */
    vector<SegmentsGroup* > tests;
    /**
     * Contain the collection of results as
     * a vector of vector<GraphAlignedToken*>
     */
    vector<GraphAlignedSegment*> results;
    /**
     * Contain the Properties for each test
     */
    vector<map<string, string> > properties;
    
    
};

#endif // BENCHMARK_H
