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
 
#ifndef SPKRAUTOOVERLAP_H
#define SPKRAUTOOVERLAP_H

#include "checker.h"
#include "logger.h"

/**
 * Check and Validate the overlaping of a speaker with himself
 */
class SpkrAutoOverlap : public Checker
{
	public:
		// class constructor
		SpkrAutoOverlap() {}
		// class destructor
		~SpkrAutoOverlap() {}
		
		bool isProcessAllSpeechSet() { return false; }
		
		unsigned long int ProcessSingleSpeech(Speech* speech);
		unsigned long int ProcessSpeechSet(SpeechSet* ref, map<string, SpeechSet*> &hyp) { return 0; }
		
		void LoadFile(const string& filename) { }
		
	private:
        static Logger* logger;
};

#endif
