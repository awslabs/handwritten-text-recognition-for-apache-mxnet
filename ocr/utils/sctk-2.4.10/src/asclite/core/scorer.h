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

#ifndef SCORER_H
#define SCORER_H

#include "stdinc.h"
#include "alignment.h"
#include "speakermatch.h"

/**
 * Interface to a scorer implementation.
 * A Scorer is an object which will give a quantification to
 * a couple of Token for all Token in an Alignement.
 */
class Scorer
{
	public:
		// class constructor
		Scorer();
		// class destructor
		virtual ~Scorer();
		/**
		 * Launch the scoring on the Alignement object.
		 */
		virtual void Score(Alignment* alignement, SpeakerMatch *speakerMatch)=0;
};

#endif // SCORER_H
