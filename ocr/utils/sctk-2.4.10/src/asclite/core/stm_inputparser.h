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

#ifndef STM_INPUTPARSER_H
#define STM_INPUTPARSER_H

#include "linestyle_inputparser.h" // inheriting class's header file

/**
 * Load the named file into a speech element.
 */
class STMInputParser : public LineStyleInputParser
{
	public:
		// class constructor
		STMInputParser() {}
		// class destructor
		virtual ~STMInputParser() {}
		/*
		static const int STM_SOURCE_SIZE_MAX = 100;
		static const int STM_CHANNEL_SIZE_MAX = 10;
		static const int STM_SPKR_SIZE_MAX = 100;
		static const int STM_LUR_SIZE_MAX = 100;
		*/		
		/*
         * Load the named file into a Speech element.
         * Create a segment for each STM line
         */
        SpeechSet* loadFile(const string& name);
    private:
    	bool CompareToISGCaseInsensitive(char str[]);
    	void LowerCase(char str[]);
    
        static const string IGNORE_TIME_SEGMENT_IN_SCORING;
        static Logger* logger;
        /**
         * Attach all Last Tokens of seg1 to all First tokens of seg2
         */
        void Attach(Segment* seg1, Segment* seg2);
};

#endif // STM_INPUTPARSER_H
