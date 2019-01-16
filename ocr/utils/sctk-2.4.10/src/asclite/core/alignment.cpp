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
 * This object represent the alignements for a test set.
 */

#include "alignment.h"
#include "token.h"
#include "segment.h"
#include "alignedspeechiterator.h"

Logger* Alignment::logger = Logger::getLogger();

Alignment::Alignment()
{
	m_references = map< Speech*, AlignedSpeech* >();
}

Alignment::~Alignment()
{
	map< Speech* , AlignedSpeech* >::iterator i, ei;
	
	i = m_references.begin();
	ei = m_references.end();
	
	while(i !=ei)
	{
		AlignedSpeech* ptr_elt = i->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++i;
	}
	
	m_references.clear();
	systems.clear();
}

string Alignment::ToString()
{
	string result = "Alignment:\n";
	AlignedSpeech* reference;
	AlignedSpeechIterator* references = AlignedSpeeches();
    
	while(references->Current(&reference))
    {
		result += reference->ToString() + "\n";
	}
    
	return result;
}

AlignedSpeech* Alignment::GetOrCreateAlignedSpeechFor(Speech* referenceSpeech, const bool& doCreate)
{
	AlignedSpeech* result = m_references[referenceSpeech];
    
	if(result == NULL && doCreate)
    {
		result = new AlignedSpeech(referenceSpeech);
		m_references[referenceSpeech] = result;
	}
    
	return result;
}

void Alignment::AddGraphAlignedSegment(GraphAlignedSegment* gas, const string& hyp_key, SegmentsGroup* segmentsGroup)
{
	if(gas == NULL)
		return;
		
	Token* ref = NULL;
	Token* hyp = NULL;
    Token* lastNonNullRef = NULL;
	Token* nextNonNullRef = NULL;
	Speech* refSpeech = NULL;
	Segment* refSegment = NULL;
	AlignedSpeech* alignedSpeech = NULL;
	AlignedSegment* alignedSegment = NULL;
	bool didLookRight = false;
	bool didLookLeft = false;
    	
	for(size_t i = 0; i < gas->GetNbOfGraphAlignedToken(); ++i)
	{
		refSegment = NULL;
		ref = gas->GetNonNullReference(i);
		hyp = gas->GetNonNullHypothesis(i);
		
		// handling insertions!
		if(ref == NULL)
		{		
			didLookLeft = false;
			didLookRight = false;
			
			do 
            {
                refSegment = NULL;
				
                //cout << "pouet: " << hyp->ToString() << endl;
				
				// first retrieve a ref segment most likely associated with the hyp token
				if( (lastNonNullRef != NULL) && !didLookLeft )
				{ // first look left for a ref token		
					refSegment = lastNonNullRef->GetParentSegment();
					didLookLeft = true;
					//cout << "Look left" << endl;
				} 
                else if (!didLookRight)
                { // if no ref token on the left and we didnt already look right => look right				
					refSegment = LookRight(i, &nextNonNullRef, gas);
					didLookRight = true;
					//cout << "Look right" << endl;
				}
				
				if(refSegment == NULL) 
                {
					if(didLookRight) 
                    {
						 //at this point, we need to get segment by time
						refSegment = segmentsGroup->GetRefSegmentByTime(hyp);
						//cout << "Get segment by time" << endl;
					} 
                    else 
                    {
						// we used the left token to find a segment but no match, look right then
						refSegment = LookRight(i, &nextNonNullRef, gas);					
						didLookRight = true;
						//cout << "Look right again" << endl;
					}				
				}
				
				//cout << "hyp{" << hyp->GetStartTime() << "," << hyp->GetEndTime()<< "}, ref:" <<  refSegment->ToString()<< endl;
			}
			while (!hyp->OverlapWith(refSegment) && hyp->GetStartTime() >= 0 && hyp->GetEndTime() >= 0);
		}
		else 
		{ // we have a ref token (not an insertion): use it! :)
			refSegment = ref->GetParentSegment();
			lastNonNullRef = ref;
		}
		
		refSpeech = refSegment->GetParentSpeech();
		alignedSpeech = this->GetOrCreateAlignedSpeechFor(refSpeech, true);
		alignedSegment = alignedSpeech->GetOrCreateAlignedSegmentFor(refSegment, true);
		alignedSegment->AddTokenAlignment(ref, hyp_key, hyp);
		alignedSegment->SetSegGrpID(segmentsGroup->GetsID());
	}
	
	//Case where the references are empty and there is no entry for the hyp
	if (gas->GetNbOfGraphAlignedToken() == 0 && segmentsGroup->GetNumberOfReferences() != 0)
	{
        for (size_t i=0 ; i < segmentsGroup->GetNumberOfReferences() ; ++i)
        {
            vector<Segment*> refsSeg = segmentsGroup->GetReference(i);
            
            for (size_t j=0 ; j < refsSeg.size() ; ++j)
            {
                refSpeech = refsSeg[j]->GetParentSpeech();
                alignedSpeech = this->GetOrCreateAlignedSpeechFor(refSpeech, true);
                alignedSpeech->GetOrCreateAlignedSegmentFor(refsSeg[j], true);
            }
        }
    }
}

Segment* Alignment::LookRight(const size_t& gatIndex, Token** nextNonNullRef, GraphAlignedSegment* gas)
{	
	*nextNonNullRef = gas->GetNextNonNullReference(gatIndex);
	
	if(*nextNonNullRef != NULL)
    { // use ref token on the right
		return (*nextNonNullRef)->GetParentSegment();					
	}
	
	return NULL;
}

AlignedSpeechIterator* Alignment::AlignedSpeeches()
{
	return new AlignedSpeechIterator(this);
}
