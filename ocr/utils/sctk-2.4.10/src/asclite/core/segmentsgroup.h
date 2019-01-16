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
 
#ifndef SEGMENTSGROUP_H
#define SEGMENTSGROUP_H

#include "token.h"
#include "segment.h"
#include "id.h"
#include "logger.h"

/**
 * This class represent a group of segments with the information
 *  about ref/hyp if this information exist.
 */
class SegmentsGroup
{
	public:
		// class constructor
		SegmentsGroup() { s_id = ID::GetID(); }
		// class destructor
		~SegmentsGroup();
		/**
		 * Add an Hypothesis into the SegmentGroup.
		 */
		void AddHypothesis(Segment* hypothesis);
		/**
		 * Add a list of Hypothesis into the SegmentGroup.
		 * This list should be under the same Speech object.
		 */
		void AddHypothesis(const vector<Segment*> & hypothesis) { hypothesiss.push_back(hypothesis); }
		/**
		 * Add a reference segment into the SegmentGroup.
		 */
		void AddReference(Segment* reference);
		/**
		 * Add a list of reference segment into the SegmentGroup.
		 * This list should be under the same Speech object.
		 */
		void AddReference(const vector<Segment*> & reference) { references.push_back(reference); }
		/**
		 * Return the Segments for the reference nb index
		 */
		vector<Segment*> GetReference(const size_t& index) { return references[index]; }
		/**
		 * Return the Segments for the hypothesis nb index
		 */
		vector<Segment*> GetHypothesis(const size_t& index) { return hypothesiss[index]; }
		/**
		 * Return the List of Token by topological order for the Reference no i
		 */
		vector<Token*> ToTopologicalOrderedStructRef(const size_t& index);
		/**
		 * Return the List of Token by topological order for the Hypothesis no i
		 */
		vector<Token*> ToTopologicalOrderedStructHyp(const size_t& index);
		/**
         * Return the number of references
         */
        size_t GetNumberOfReferences() { return references.size(); }
        /**
         * Return the number of hypothesis
         */
        size_t GetNumberOfHypothesis() { return hypothesiss.size(); }
        /**
         * Return a possible Reference for the given Hyp token.
         * The Segment returned is the first that match the token time
         * This method should always return a Segment as an Reference should always contain a Segment at a given time.
         */
        Segment* GetRefSegmentByTime(Token* token);
        /**
         * Return true if the Group contains segment to ignore in scoring
         */
        bool isIgnoreInScoring();
        /**
         * Get the guessed difficulty number of this alignement
         * linear number > with difficulty
         */
        ullint GetDifficultyNumber();
		
		/*
		 * Return a String definint the difficulty number 
		 */
		string GetDifficultyString();
		
		/** Returns a string representation of this SegmentGroup. */
		string ToString();
		
		/** Log display Alignment */
		void LoggingAlignment(const bool& bgeneric, const string& type);
		
		ulint GetsID() { return s_id; }
		
		int GetTotalDuration();
    
		int GetMinTime();
		
		int GetMaxTime();
    
	private:
		/**
		 * Vector that contain all the segments (ref)
		 */
		vector<vector<Segment*> > references;
		
		/**
		 * Vector that contain all the segments (hyp)
		 */
		vector<vector<Segment*> > hypothesiss;
		
		/** the logger */
        static Logger* logger;
		
		ulint s_id;
};

#endif // SEGMENTSGROUP_H
