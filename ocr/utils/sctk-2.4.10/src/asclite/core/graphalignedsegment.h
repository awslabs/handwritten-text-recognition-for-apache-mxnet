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

#ifndef GRAPHALIGNEDSEGMENT_H
#define GRAPHALIGNEDSEGMENT_H

#include "stdinc.h"

#include "graphalignedtoken.h"
#include "segment.h"
#include "token.h"
#include "logger.h"

/**
 * The result produce by the Graph after a Levenshtein alignment
 */
class GraphAlignedSegment
{
	private:
		/** container of the alignedtokens */
		vector<GraphAlignedToken*> m_vGraphAlignedTokens;
		/** Hyp-Ref Index */
		size_t m_HypRefIndex;
		
		static Logger* m_pLogger;
	public:
		/** class constructor */
		GraphAlignedSegment(const size_t& _HypRefIndex);
		/** class destructor */
		~GraphAlignedSegment();
				
		/** Return the indexed element of the GraphAlignedSegment */
		GraphAlignedToken* GetGraphAlignedToken(const size_t& index);
		/** Return the nb of GraphAlignedToken */
		size_t GetNbOfGraphAlignedToken() { return m_vGraphAlignedTokens.size(); }
		/** Add a GraphAlignedToken into the structure to the back */
		void AddFrontGraphAlignedToken(GraphAlignedToken* _pGToken) { m_vGraphAlignedTokens.push_back(_pGToken); }
		/** Returns a string representation of this GraphAlignedSegment. */
		string ToStringAddnChar(const string& chara, const int& num);
		string ToStringAddText(const string& text, const int& maxc);
		string ToString();
		/** Retrieves the non-null reference for the (gatIndex + 1)th 
		 * GraphAlignedToken (or NULL if no such reference exists for that 
		 * GraphAlignedToken) 
		 */
		Token* GetNonNullReference(const size_t& gatIndex);
		/** Retrieves the non-null hypothesis for the (gatIndex + 1)th 
		 * GraphAlignedToken (or NULL if no such hypothesis exists for that 
		 * GraphAlignedToken) 
		 */
		Token* GetNonNullHypothesis(const size_t& gatIndex);
		
		/** Retrieves the precedent non-null reference that occured before 
		 * (gatIndex +1) in this GraphAlignedSegment (or NULL if such reference
		 * does not exist)
		 */
		Token* GetPreviousNonNullReference(const size_t& gatIndex);
		
		/** Retrieves the next non-null reference that occurs after 
		 * (gatIndex +1) in this GraphAlignedSegment (or NULL if such reference
		 * does not exist)
		 */
		Token* GetNextNonNullReference(const size_t& gatIndex);
		
		void LoggingAlignment(const ulint& seggrpid);
		
		/**
		 * Redefine the == operator to go throw all the object for the comparison
		 */
		bool operator ==(const GraphAlignedSegment & gas) const;
		bool operator !=(const GraphAlignedSegment & gas) const;
};

#endif // GRAPHALIGNEDSEGMENT_H
