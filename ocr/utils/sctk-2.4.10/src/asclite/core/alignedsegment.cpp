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
 * Represent the collection of TokenAlignment for a segment.
 */

#include "alignedsegment.h"

AlignedSegment::AlignedSegment(Segment* referenceSegment)
{
    m_tokenAlignments = vector< TokenAlignment* >();
    m_refToAlignments = map<Token*, TokenAlignment*>();
    m_referenceSegment = referenceSegment;
}

AlignedSegment::~AlignedSegment()
{
	for(size_t i=0; i< m_tokenAlignments.size(); ++i)
	{
		TokenAlignment* ptr_elt = m_tokenAlignments[i];
		
		if(ptr_elt)
			delete ptr_elt;
	}
	
	m_tokenAlignments.clear();
	m_refToAlignments.clear();
}

string AlignedSegment::ToString()
{
	size_t nbTokAli = GetTokenAlignmentCount();
	string result = "<AlignedSegment>\n\tref:\n\t" + m_referenceSegment->ToString() + "\n";
    
	for(size_t i = 0; i < nbTokAli; ++i)
    {
		result += GetTokenAlignmentAt(i)->ToString() + "\n";
	}
    
	result += "</AlignedSegment>";
	return result;
}

TokenAlignment* AlignedSegment::GetTokenAlignmentFor(Token* reference, const bool& create)
{	
	TokenAlignment* ta = m_refToAlignments[reference];
	
	// if we don't want to create a new TA or one already exists for reference, return it
	if(!create || ta != NULL)
    {
		return ta;
	}
	
	ta = new TokenAlignment(reference);		
	m_tokenAlignments.push_back(ta);
	
	// don't add ta to the map if it's not associated to a non-NULL reference
	if(reference != NULL)
    {
		m_refToAlignments[reference] = ta;
	}
	
	return ta;
}

int AlignedSegment::AddTokenAlignment(Token* reference, const string& hypKey, Token* hypothesis) 
{
	if(reference != NULL && reference->GetParentSegment() != m_referenceSegment) 
    {
		return -1;
	}
	
	return GetTokenAlignmentFor(reference, true)->AddAlignmentFor(hypKey, hypothesis);
}
