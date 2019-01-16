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

#ifndef ALIGNMENT_H
#define ALIGNMENT_H

#include "stdinc.h"
#include "alignedspeech.h"
#include "graphalignedsegment.h"
#include "segmentsgroup.h"
#include "logger.h"

class AlignedSpeechIterator;

/**
 * This object represent the alignements for a test set.
 */
class Alignment
{		
    public:
        Alignment();
        ~Alignment();
	
        /**
         * Add a system name to the alignment
         */
        void AddSystem(const string& filename, const string& system) { systemFilenames.push_back(filename); systems.push_back(system); }
        /**
         * Return the nb of systems this alignment contain
         */
        size_t GetNbOfSystems() const { return systems.size(); }
        /**
         * Return the system nb index
         */
        string GetSystem(const size_t& index) const { return systems[index]; }
        /**
         * Return the system filename nb index
         */
        string GetSystemFilename(const size_t& index) const { return systemFilenames[index]; }
        
        /**
         * Add a Graph Aligned Segment into the Alignment structure.
         * 
         * @param gas The graphAlignedSegment to add
         * @param hyp_key the key to id the hypothesis into the alignment
         */ 
        void AddGraphAlignedSegment(GraphAlignedSegment* gas, const string& hyp_key, SegmentsGroup* segmentsGroup);
	
        /** 
         * Retrieve or create the AlignedSpeech associated with the given reference Speech. 
         *
         * @param referenceSpeech the reference Speech associated with the AlignedSpeech to be retrieved
         * @param doCreate if true, the method will create a new AlignedSpeech if none already exist for the given Speech.
         */
        AlignedSpeech* GetOrCreateAlignedSpeechFor(Speech* referenceSpeech, const bool& doCreate);
	
        /** Retrieves an iterator over the AlignedSpeeches contained in this Alignment */
        AlignedSpeechIterator* AlignedSpeeches();
	
        /** Returns a string representation of this Alignment. */
        string ToString();
	
        friend class AlignedSpeechIterator;
		
		//ulint GetsSegGrpID() { return s_seggrp_id; }
			
    private:
        map< Speech* , AlignedSpeech* > m_references;
        vector<string> systems;
        vector<string> systemFilenames;
	
        Segment* LookRight(const size_t& gatIndex, Token** nextNonNullRef, GraphAlignedSegment* gas);
		
		/**
         * the logger
         */
        static Logger* logger;
};

#endif
