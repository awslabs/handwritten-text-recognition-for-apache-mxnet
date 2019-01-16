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

/** 
 * Represents an aligned system output.
 */

#include "alignedspeech.h"
#include "alignedsegmentiterator.h"

AlignedSpeech::AlignedSpeech(Speech* speech)
{
	m_segments = map <Segment*,  AlignedSegment* >();
	m_speech = speech;
}

AlignedSpeech::~AlignedSpeech()
{
	map< Segment*,  AlignedSegment* >::iterator i, ei;
	
	i = m_segments.begin();
	ei = m_segments.end();
	
	while(i != ei)
	{
		AlignedSegment* ptr_elt = i->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++i;
	}
	
	m_segments.clear();

}

AlignedSegment* AlignedSpeech::GetOrCreateAlignedSegmentFor(Segment* segment, const bool& doCreate)
{
	// segment to be inserted need to be part of the same reference speech
	if(segment == NULL || m_speech != segment->GetParentSpeech())
    {
		return NULL;
	}
	
	AlignedSegment* result = m_segments[segment];
    
	if(result == NULL && doCreate)
    {
		result = new AlignedSegment(segment);
		m_segments[segment] = result;
	}
    
	return result;
}

AlignedSegmentIterator* AlignedSpeech::AlignedSegments()
{
	return new AlignedSegmentIterator(this);
}

string AlignedSpeech::ToString()
{
	AlignedSegmentIterator* alSegments = AlignedSegments();
	AlignedSegment* alSeg;
	string result = "<AlignedSpeech>\n";
    
	while(alSegments->Current(&alSeg))
    {
		result += alSeg->ToString() + "\n";
	}
    
	result += "</AlignedSpeech>";
    
	delete alSegments;
    
	return result;
}
