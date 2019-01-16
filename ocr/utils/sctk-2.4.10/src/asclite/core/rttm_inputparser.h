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

#ifndef RTTM_INPUTPARSER_H
#define RTTM_INPUTPARSER_H

#include "linestyle_inputparser.h"
#include "logger.h"

/**
 * Implementation of a RTTM parser for asclite.
 */
class RTTMInputParser : public LineStyleInputParser
{
	public:
		// class constructor
		RTTMInputParser() {}
		// class destructor
		virtual ~RTTMInputParser() {}
		/**
		 * Load the named file into a Speech element.
		 * Create a segment for each RTTM line
		 */
		SpeechSet* loadFile(const string& name);
		
		SpeechSet* loadFileSpeaker(const string& name);
		SpeechSet* loadFileLexeme(const string& name);
  
  private:
        static const string IGNORE_TIME_SEGMENT_IN_SCORING;
        static Logger* logger;
        /**
          * Attach all Last Tokens of seg1 to all First tokens of seg2
          */
        void Attach(Segment* seg1, Segment* seg2);
};

#endif // RTTM_INPUTPARSER_H
