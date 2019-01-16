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
 
#ifndef FILTER_H
#define FILTER_H

#include "stdinc.h"
#include "speech.h"
#include "speechset.h"

/**
 * Abstract interface to a Filter.
 */
class Filter
{
	public:
		// class constructor
		Filter();
		// class destructor
		virtual ~Filter();
		
		virtual bool isProcessAllSpeechSet() = 0;
		virtual unsigned long int ProcessSingleSpeech(Speech* speech) = 0;
		virtual unsigned long int ProcessSpeechSet(SpeechSet* ref, map<string, SpeechSet*> &hyp) = 0;
		
		virtual void LoadFile(const string& filename) = 0;
};

#endif // FILTER_H
