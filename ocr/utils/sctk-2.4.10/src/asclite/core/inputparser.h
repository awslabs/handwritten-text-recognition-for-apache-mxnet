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

#ifndef INPUTPARSER_H
#define INPUTPARSER_H

#include "stdinc.h"
#include "token.h"
#include "segment.h"
#include "speech.h"
#include "speechset.h"

/**
 * This class represent a abstract input parser.
 * the main method is loadFile()
 */
class InputParser
{
	public:
		// class constructor
		InputParser() {	m_bOneTokenPerSegment = false; }
		// class destructor
		virtual ~InputParser() { }
		/**
		 * Load the select file and create the 
		 * Speech/Segment/Token structure with it.
		 */
		virtual SpeechSet* loadFile(const string& name)=0;
		
		int ParseString(const string& chaine)  { return static_cast<int>(floor( ( atof(chaine.c_str()) * 1000 ) + 0.5)); }
		
		void SetOneTokenPerSegment(const bool& _bool) { m_bOneTokenPerSegment = _bool; }
		bool isOneTokenPerSegment() { return m_bOneTokenPerSegment; }
		
	protected:
        /**
         * Build a Token with a text input
         */
        Token* BuildToken(const int& start, const int& dur, const string& text_to_parse, Segment* parent);
		
		bool m_bOneTokenPerSegment;
};

#endif // INPUTPARSER_H
