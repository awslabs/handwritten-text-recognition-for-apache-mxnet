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
 
#ifndef GRAPH_H
#define GRAPH_H

#include "stdinc.h"

#include "token.h"
#include "segment.h"
#include "graphalignedtoken.h"
#include "graphalignedsegment.h"
#include "segmentsgroup.h"
#include "logger.h"

#include "linestyle_inputparser.h"
#include "graph_coordinate.h"
#include "arraylevenshteinmatrix.h"
#include "compressedlevenshteinmatrix.h"
#include "speakermatch.h"

/**
 * Inplementation of the Levenshtein Distance Algorithm in Multi-Dimension
 */
class Graph
{
    // Members
	private:
		
		/**
		 * Database for the optimization speaker alignment 
		 */
		SpeakerMatch* m_pSpeakerMatch;
		
		int m_typeCostModel; // 1: regular levenshtein
							 // 2: time based

        int m_CostTransition;		
		/** Cost Insertion and Deletion */
        int m_CostInsertion;
		/** Cost Optionally Insertion and Deletion */
        int m_CostOptionally;
        /** Cost for Correct but not in the same speaker */
        int m_CostCorrectNonSpeaker;
		/** Cost for Adaptive */
        int m_CostAdaptive;
        /** process or not optionnaly token for reference */
        bool m_useOptForRef;
        /** process or not optionnaly token for hypothesis */
        bool m_useOptForHyp;
        /** Number of dimension */
        bool m_bCompressedArray;
		/** Cost Transition */
		size_t m_Dimension;
		/** Deep for each dimension, deep is the number of element on the dimension */
		size_t* m_TabDimensionDeep;
		/** Position of the first ref into the dimension array
		 *  H ... H R R .... R
		 *          ^
	     *     m_IndexRef
	     */		
		size_t m_IndexRef;
		/**
		 *  true: not using optimization for Hyp and Ref cost computation
		 *  false: using optimization for Hyp and Ref cost computation
		 */
		bool m_HypRefStatus;
		
        
		/** map associating nodes (via their coordinates) to cost */
		LevenshteinMatrix* m_MapCost;
		
		/** Array of Hyps and Refs */
		vector<Token*>* m_TabVecHypRef;
		/** Map token to index into vector */
		map<Token*, size_t>* m_TabMapTokenIndex;
        /** List of first tokens of the segments */
		list<Token*>* m_TabFirstTokens;
		/** List of last tokens of the segments */
		list<Token*>* m_TabLastTokens;
		/** Optimization for the graph */
		bool m_bPruneOptimization;
		bool m_bWordOptimization;
		bool m_bSpeakerOptimization;
		bool m_bAdaptiveCostOptimization;
		bool m_bWordAlignCostOptimization;
		
		/** optimization gap */
		int m_PruneOptimizationThreshold;
		/** optimization Insert/Del */
		int m_WordOptimizationThreshold;
		
		int m_EstimatedMaxCost;
		
		int m_MaxDurationSegmentGroup;
        
        /** Logger */
		static Logger* logger;
        
        // Caching
        /** Caching for transition cost */
        list<size_t>*** m_TabCacheDimPreviousIndex;

		int m_TimeBasedSafeDivider;
		
		size_t m_NbThreads;
        
    // Methods
	public:
        /** Constructor */
		Graph() {}
		/** Constructor with the number of dimension */
		Graph(SegmentsGroup* _segmentsGroup, SpeakerMatch* _pSpeakerMatch, const int& _typeCost, const int& _costTrans, const int& _costIns, const int& _costOpt, const int& _costCorrectNonSpeaker, const int& _costAdaptive, const bool& _optIns, const bool& _optDel, const bool& _bCompressedArray);
		/** Destructor */
		~Graph();
		
		/** Set the deep of one dimension */
		void SetDimensionDeep(const size_t& dim, const size_t& deep);
		/** Set the position of the first ref */
		void SetIndexRef(const size_t& ind);
		/** Set the dimension */
		void SetDimension(const size_t& dim);
	
        /** Returns the deep of one dimension */
		size_t GetDimensionDeep(const size_t& d) { return m_TabDimensionDeep[d]; }
		/** Returns the number of dimensions */
		size_t GetDimension() { return m_Dimension; }
		
		/** Return true if the dimension is a reference, false if not */
		bool IsReference(const size_t& dim) { return(dim >= m_IndexRef); }
		/** Return true if the dimension is a hypothesis, false if not */
		bool IsHypothesis(const size_t& dim) { return(dim < m_IndexRef); }
		
		// Debug methods
		/** Size of the map */
		size_t SizeMap() { return m_MapCost->GetNumberOfCalculatedCosts(); }
		
		/** returns cost of insertion */
		int GetCostInsertion(const bool& optionally) { return optionally ? m_CostOptionally : m_CostInsertion; }
		/** returns cost of transition */
		int GetCostTransition() { return m_CostTransition; }
		int GetCostTransitionWordBased(Token* pToken1, Token* pToken2);
		int GetCostTransitionTimeBased(Token* pToken1, Token* pToken2);
		
		int GetCostAdaptive(Token* pToken1, Token* pToken2);
		
		int GetCostWordAlign(Token* pToken1, Token* pToken2);
		
		/** Calculate the cost for the coordinate */
        int CalculateCost(size_t* curcoord);
		
		/** Fill the graph with cost */
		void FillGraph();
		
		/** Return the best (min) cost at the end of the graph */
		int GetBestCost();
		
		/** Returns the alignment */
		GraphAlignedSegment* RetrieveAlignment();
		
		/** Print the Levenshtein array */
		void PrintLevenshteinArray();
		
	private:
		/** Create the list of starting coordinates */
		void StartingCoordinates(GraphCoordinateList& listStart);
	
		/** is the one of the last possible coordinates ? */
		bool isEndingCoordinate(size_t* coord);
	
		/** List the previous coordinates */
		void PreviousCoordinates(GraphCoordinateList& listPrev, size_t* coord) { m_HypRefStatus ? PreviousCoordinatesHypRef(listPrev, coord) : PreviousCoordinatesGeneric(listPrev, coord); }
		/** List the previous coordinates optimized for Hyp-Ref constraints */
		void PreviousCoordinatesHypRef(GraphCoordinateList& listPrev, size_t* coord);
		/** List the previous coordinates generic way to compute */
		void PreviousCoordinatesGeneric(GraphCoordinateList& listPrev, size_t* coord);
	
        /** returns the list of previous indexes */
		void PreviousIndexes(list<size_t>& listPrev, const size_t& dim, const size_t& index);
		
		/** returns the cost between 2 coordinates */
		int GetTransitionCost(size_t* coordcurr, size_t* coordprev) { return m_HypRefStatus ? GetTransitionCostHypRef(coordcurr, coordprev) : GetTransitionCostGeneric(coordcurr, coordprev); }
		/** returns the cost between 2 coordinates for Hyp-Ref constraints */
		int GetTransitionCostHypRef(size_t* coordcurr, size_t* coordprev);
		/** returns the cost between 2 coordinates generic way to compute */
		int GetTransitionCostGeneric(size_t* coordcurr, size_t* coordprev);
		
		int GetTransitionCostHypRefWordBased(size_t* coordcurr, size_t* coordprev);
		int GetTransitionCostGenericWordBased(size_t* coordcurr, size_t* coordprev);
		int GetTransitionCostHypRefTimeBased(size_t* coordcurr, size_t* coordprev);
		int GetTransitionCostGenericTimeBased(size_t* coordcurr, size_t* coordprev);
		
		
		/** Returns the best previous coordinate */
		size_t* GetBestCoordinateAndCost(size_t* coordcurr);
		
		/** Check if the Hyp or Ref is empty */
		bool isHypRefEmpty(const size_t& hr) { return(m_TabLastTokens[hr].empty()); }
		
		/** Returns the number of coordinates which have changed */
		size_t NumberChanged(size_t* coord1, size_t* coord2);
		
		/** returns true if the transition/Insertion/Deletion is allowed */
		bool ValidateTransitionInsertionDeletion(size_t* coordcurr, size_t* coordprev);
		
		/** Use the optimization */
		void SetGraphOptimization();
		
//		bool isMultiThreaded() { return(m_NbThreads>1); }
};

#endif
