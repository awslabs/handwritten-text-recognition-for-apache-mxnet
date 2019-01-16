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
 
#ifndef ALIGNEDSEGMENT_H
#define ALIGNEDSEGMENT_H

#include "stdinc.h"
#include "segment.h"
#include "tokenalignment.h"

/**
 * Represent the collection of TokenAlignment for a segment.
 */
class AlignedSegment
{
    public:
        AlignedSegment(Segment* referenceSegment);
        ~AlignedSegment();

        /**
         * Retrieve the (index+1)th token alignment.
         */
        TokenAlignment* GetTokenAlignmentAt(size_t index) { return m_tokenAlignments[index]; }

        /**
         * Return the count of TokenAlignment
         */
        size_t GetTokenAlignmentCount() { return m_tokenAlignments.size(); }

        /**
         * Add a TokenAlignment to the AlignedSegment
         */
        void AddTokenAlignment(TokenAlignment* ta);

        /**
         * Retrieve the reference segment associated with this AlignedSegment.
         */
        Segment* GetReferenceSegment() { return m_referenceSegment; }

        /** Specifies that the given hypothesis Token is the alignment for the system
         * identified with the given key for the given reference Token.
         * @param reference the reference Token
         * @param hypKey the string identifying the system which output is being aligned
         * @param hypothesis the system Token being aligned to the reference
         * @return an int < 0 if the addition was not successful, 0 if the addition 
         *					had already been done previously
         */
        int AddTokenAlignment(Token* reference, const string& hypKey, Token* hypothesis);

        /** Retrieves the TokenAlignment associated to the specified reference Token
         * or NULL if no such TokenAlignment exists.
         * @param reference the reference Token which associated TokenAlignment is to be retrieved
         */
        TokenAlignment* GetTokenAlignmentFor(Token* reference) { return GetTokenAlignmentFor(reference, false); }

        /** Returns a string representation of this AlignedSegment */
        string ToString();
		
		void SetSegGrpID(ulint _id) { m_SegGrpID = _id; }
		
		ulint GetSegGrpID() { return m_SegGrpID; };
	
    private:
        vector< TokenAlignment* > m_tokenAlignments;
        Segment* m_referenceSegment;
        map< Token*, TokenAlignment* > m_refToAlignments;
        ulint m_SegGrpID;
        
        TokenAlignment* GetTokenAlignmentFor(Token* ref, const bool& create);
};

#endif
