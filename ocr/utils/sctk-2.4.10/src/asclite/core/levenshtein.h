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

#ifndef LEVENSHTEIN_H
#define LEVENSHTEIN_H

#include "aligner.h"
#include "graph.h"
#include "speakermatch.h"

/**
 * Implementation of an Aligner with the Levenshtein algorithm
 */
class Levenshtein : public Aligner
{
	public:
		// class constructor
		Levenshtein();
		// class destructor
		~Levenshtein();
		/**
		 * Align the segments with the Levenshtein algorithm.
		 */
		virtual void Align() { graph->FillGraph(); }
		/**
         * Retrieve the results.
         */
        virtual GraphAlignedSegment* GetResults() { return graph->RetrieveAlignment(); }
		/**
		 * Initialise the Aligner to work with this set of segments
		 */
		virtual void SetSegments(SegmentsGroup* segmentsGroup, SpeakerMatch* pSpeakerMatch, const bool& useCompArray);
        /**
         * Return the minimal cost of the graph
         */
        int GetCost() { return graph->GetBestCost(); }
    
    private:
        /**
         * Implementation of the Graph
         */
        Graph* graph;
		
        /**
         * Where the ref begin in the list of Segments
         */
        size_t refSepIndex;
    
        /** Logger */
        static Logger* logger;
};

#endif // LEVENSHTEIN_H
