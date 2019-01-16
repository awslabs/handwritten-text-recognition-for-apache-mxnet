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

#ifndef TEST_CTMSTMRTTM_SEGMENTOR_H
#define TEST_CTMSTMRTTM_SEGMENTOR_H

#include "stdinc.h"
#include "ctmstmrttm_segmentor.h"
#include "speech.h"
#include "segment.h"

/**
 * Common testing for the CTM_STM implementation of 
 * the segmentor.
 */
class CTMSTMRTTMSegmentorTest
{
	public:
		// class constructor
		CTMSTMRTTMSegmentorTest();
		// class destructor
		~CTMSTMRTTMSegmentorTest();
		void testBasicIteration();
		void testSimpleRefOverlapIteration(); 
		void testSequentialSegments();
		void testEmptySegmentOverlapCaseIteration();
		void testRefShorterThanHypCase();
    void testAll();
};

#endif // TEST_CTMSTM_SEGMENTOR_H
