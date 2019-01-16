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

#ifndef ALIGNEDSPEECH_H
#define ALIGNEDSPEECH_H

#include "stdinc.h"
#include "alignedsegment.h"
#include "speech.h"
#include "segment.h"

class AlignedSegmentIterator;

/** 
 * Represents an aligned system output.
 */
class AlignedSpeech
{	
	friend class AlignedSegmentIterator;
    
    public:
        AlignedSpeech(Speech* referenceSpeech);
        ~AlignedSpeech();
            
        /** Retrieve or create the AlignedSegment associated with the given Segment. 
         * @param segment the Segment associated with the AlignedSegment to be retrieved
         * @param doCreate if true, the method will create a new AlignedSegment if none already exist for the given segment.
         */
        AlignedSegment* GetOrCreateAlignedSegmentFor(Segment* segment, const bool& doCreate);
        
        /** Retrieves an iterator over all the AlignedSegments contained in this AlignedSpeech. */
        AlignedSegmentIterator* AlignedSegments();
        
        /** Retrieves the associated reference Speech. */
        Speech* GetReferenceSpeech() { return m_speech; }
        
        /** Returns a string representation of this AlignedSpeech. */
        string ToString();
	
    private:
            map< Segment*,  AlignedSegment* > m_segments;
            Speech* m_speech;
};

#endif
