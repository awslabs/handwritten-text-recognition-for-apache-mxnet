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

#ifndef CTMSTMRTTM_SEGMENTOR_H
#define CTMSTMRTTM_SEGMENTOR_H

#include "segmentor.h" // inheriting class's header file
#include "logger.h"

/**
 * Implementation of a segmentor that take rttm hyp and stm reference.
 */
class CTMSTMRTTMSegmentor : public Segmentor
{
	public:
		// class constructor
		CTMSTMRTTMSegmentor() {}
		// class destructor
		~CTMSTMRTTMSegmentor();
		/**
		 * Reset the segmentor with the references and hypothesis
		 * If the references are the same as before only the iteration is initialised.
		 */
		void Reset(SpeechSet* references, SpeechSet* hypothesis);
		/**
		 * Return true if the segmentor have more segments to process.
		 * This method is not time consuming and can be call many time.
		 */
		bool HasNext() { return(currentSegmentRef != NULL); }
		/**
		 * Return the next group of segments to process.
		 * This method is time consuming and will return a different result at each call.
		 */
		SegmentsGroup* Next();
		
		void ResetGeneric(map<string, SpeechSet*> &mapspeechSet);
		SegmentsGroup* NextGeneric();
	private:
        /**
         * The list of all the source file
         */
        set<string> sourceList;
        /**
         * The list of all the channel
         */
        map<string, set<string> > channelList;
        /**
         * Loop to return the last segment occurance into an overlaping loop.
         */
        Segment* GetLastOverlapingSegment(/*int startTime, */SpeechSet* speechs);
        Segment* GetLastOverlapingSegmentGeneric(/*int startTime*/);
        /**
         * Return the first segment that occur after the given time
         */
        Segment* GetFirstSegment(const int& startTime, SpeechSet* speechs);
        Segment* GetFirstSegmentGeneric(const int& startTime);
        /**
         * The logger
         */
        static Logger* logger;
        //-----------------------
        // Position attributes
        //-----------------------
        Segment* currentSegmentRef;
        string currentSource;
        string currentChannel;
};

#endif // CTMSTMRTTM_SEGMENTOR_H
