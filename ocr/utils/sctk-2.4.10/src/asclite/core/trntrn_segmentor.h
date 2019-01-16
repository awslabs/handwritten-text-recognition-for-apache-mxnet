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
 
#ifndef TRNTRN_SEGMENTOR_H
#define TRNTRN_SEGMENTOR_H

#include "segmentor.h" // inheriting class's header file
#include "logger.h"

/**
 * Segmentor for reference trn and hypothesis trn file.
 */
class TRNTRNSegmentor : public Segmentor
{
	public:
		// class constructor
		TRNTRNSegmentor() {}
		// class destructor
		~TRNTRNSegmentor() { uteranceList.clear(); }
		/**
		 * Reset the segmentor with the references and hypothesis
		 * If the references are the same as before only the iteration is initialised.
		 */
		void Reset(SpeechSet* references, SpeechSet* hypothesis);
		void ResetGeneric(map<string, SpeechSet*> &mapspeechSet) { }
		
		/**
		 * Return true if the segmentor have more segments to process.
		 * This method is not time consuming and can be call many time.
		 */
		bool HasNext() { return (currentUterance != ""); }
		/**
		 * Return the next group of segments to process.
		 * This method is time consuming and will return a different result at each call.
		 */
		SegmentsGroup* Next();
		SegmentsGroup* NextGeneric() { return NULL; }

  private:
    static Logger* logger;
		/**
     * The list of all the uteranceid
     */
    set<string> uteranceList;
    //-----------------------
    // Position attributes
    //-----------------------
    string currentUterance;
};

#endif // TRNTRN_SEGMENTOR_H
