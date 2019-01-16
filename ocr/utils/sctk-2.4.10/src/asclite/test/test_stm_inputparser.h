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

#ifndef TEST_STM_INPUTPARSER_H
#define TEST_STM_INPUTPARSER_H

#include "stdinc.h"
#include "segment.h"
#include "speech.h"
#include "token.h"
#include "stm_inputparser.h"

/**
 * Test the STM input parser
 */
class STMInputParserTest
{
	public:
		// class constructor
		STMInputParserTest();
		// class destructor
		~STMInputParserTest();
		/**
		 * Launch all the test of STM input parser
		 */
		void testAll();
		/**
		 * Test the import of the Basic file
		 */
		void testBasicImport();
};

#endif // TEST_STM_INPUTPARSER_H
