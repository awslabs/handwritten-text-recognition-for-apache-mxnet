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
 
#include "benchmark.h" // class's header file

// class constructor
Benchmark::Benchmark()
{
	// insert your code here
}

// class destructor
Benchmark::~Benchmark()
{
	// insert your code here
}

/**
 * Access the tests no index
 */
SegmentsGroup* Benchmark::GetTest(int index)
{
  return tests[index];
}

/**
 * Access the tests no index
 */
map<string, string> Benchmark::GetProperties(int index)
{
  return properties[index];
}

/**
 * Access the results no index
 */
GraphAlignedSegment* Benchmark::GetResult(int index)
{
  return results[index];
}

/**
 * Get the number of tests
 */
int Benchmark::GetTestSize()
{
  return tests.size();  
}
