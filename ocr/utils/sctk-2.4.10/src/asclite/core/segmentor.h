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
 
#ifndef SEGMENTOR_H
#define SEGMENTOR_H

#include "stdinc.h"
#include "speech.h"
#include "segmentsgroup.h"
#include "speechset.h"

/**
 * Generic interface to a segmentor.
 * The role of a segmentor is to iterate on set of speech to retrieve
 * a set of segment to be aligned together.
 */
class Segmentor
{
	public:
		// class constructor
		Segmentor();
		// class destructor
		virtual ~Segmentor() {}
		/**
		 * Reset the segmentor with the references and hypothesis
		 * If the references are the same as before only the iteration is initialised.
		 */
		virtual void Reset(SpeechSet* references, SpeechSet* hypothesis)=0;
		virtual void ResetGeneric(map<string, SpeechSet*> & mapspeechSet)=0;
		/**
		 * Return true if the segmentor have more segments to process.
		 * This method is not time consuming and can be call many time.
		 */
		virtual bool HasNext()=0;
		/**
		 * Return the next group of segments to process.
		 * This method is time consuming and will return a different result at each call.
		 */
		virtual SegmentsGroup* Next()=0;
		virtual SegmentsGroup* NextGeneric()=0;
	protected:
        /**
         * references
         */
        SpeechSet* refs;
        /**
         * Hypothesis
         */
        SpeechSet* hyps;
        
        vector<SpeechSet*> m_VectSpeechSet;
};

#endif // SEGMENTOR_H
