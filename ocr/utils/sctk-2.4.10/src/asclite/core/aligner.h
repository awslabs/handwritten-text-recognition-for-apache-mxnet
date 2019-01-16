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

#ifndef ALIGNER_H
#define ALIGNER_H

#include "stdinc.h"
#include "segment.h"
#include "alignment.h"
#include "graphalignedsegment.h"
#include "segmentsgroup.h"
#include "speakermatch.h"

/**
 * This class is a generic definition for an Aligner.
 * It contain an align methods and a method to retrieve the results
 * in an Alignement object.
 */
class Aligner
{
	public:
		// class constructor
		Aligner();
		// class destructor
		virtual ~Aligner();
		/**
		 * Align the segments with the implemented algo.
		 */
		virtual void Align()=0;
		/**
		 * Retrieve the results.
		 */
		virtual GraphAlignedSegment* GetResults()=0;
		/**
		 * Initialise the Aligner to work with this set of segments
		 */
		virtual void SetSegments(SegmentsGroup* segmentsGroup, SpeakerMatch* pSpeakerMatch, const bool& useCompArray)=0;
};

#endif // ALIGNER_H
