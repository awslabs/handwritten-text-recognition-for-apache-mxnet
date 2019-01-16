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
 * Internal representation of an hypothesis file or a reference file.
 */

#include "speech.h" // class's header file
#include "speechset.h"

Logger* Speech::m_pLogger = Logger::getLogger();

// class constructor
Speech::Speech() {}

// class constructor
Speech::Speech(SpeechSet* parent)
{
	parentSet = parent;
}

Speech::Speech(const vector<Segment *>& segments)
{
	m_segments = vector<Segment *>(segments);
}

// class destructor
Speech::~Speech()
{
	for(size_t i=0; i<m_segments.size(); ++i)
	{
		Segment* ptr_elt = m_segments[i];
		
		if(ptr_elt)
			delete ptr_elt;
	}
	
	m_segments.clear();
}

bool Speech::PerformCaseSensitiveAlignment()
{
	return parentSet->PerformCaseSensitiveAlignment();
}

bool Speech::AreFragmentsCorrect()
{
	return parentSet->AreFragmentsCorrect();
}

bool Speech::UseOptionallyDeletable()
{
	return parentSet->UseOptionallyDeletable();
}

/**
 * Retrieves the SpeechSet in which this Speech is located 
 */
SpeechSet* Speech::GetParentSpeechSet()
{
  return parentSet;
}

/**
* Return the next Segment starting at the specified time for
* the specified source and channel. If the time is in the middle of
* A segment return the segment itself.
*/
Segment* Speech::NextSegment(const int& time, const string& source, const string& channel)
{
	int last_endTime = 0;	
	
	for (size_t i=0 ; i < m_segments.size() ; ++i)
	{
        Segment *currentSegment = m_segments[i];
		
		if (currentSegment->GetSource() == source && currentSegment->GetChannel() == channel)
		{
			if (currentSegment->GetEndTime() > time && last_endTime <= time)
			{
				return currentSegment;
			}
            
			last_endTime = currentSegment->GetEndTime();
		}
	}
    
	return NULL;
}

/**
 * Return the segments of this speech by the given time
 */
vector<Segment*> Speech::GetSegmentsByTime(const int& start, const int& end, const string& source, const string& channel)
{
	vector<Segment*> res;
    
	for (size_t i=0 ; i < m_segments.size() ; ++i)
	{
		Segment *currentSegment = m_segments[i];
		if (currentSegment->GetSource() == source && currentSegment->GetChannel() == channel)
		{
			int s_mid = ((currentSegment->GetEndTime()+currentSegment->GetStartTime()) / 2);
            
			if (s_mid >= start && s_mid < end)
			{
				res.push_back(currentSegment);
			}
		}
	}
	return res;
}

string Speech::ToString()
{
	std::ostringstream oss;
	oss << "<Speech NbOfSegments='" << NbOfSegments() << "'>";
	
	for(size_t i = 0; i < NbOfSegments(); ++i)
    {
        Segment *seg = GetSegment(i);
        oss << seg->ToString() << endl;
    }	
	oss << "</Speech>";
	return oss.str();
}

void Speech::RemoveSegment(Segment* currentSegment)
{	
	list<Token*> listPreviousTokenofFirstToken;
	list<Token*> listNextTokenofLastToken;
		
	// Remove links from the previous tokens of the first tokens of the segment
	for(size_t f=0; f<currentSegment->GetNumberOfFirstToken(); ++f)
	{
		Token* firstToken = currentSegment->GetFirstToken(f);
		
		if(firstToken)
		{
			for(size_t p=0; p<firstToken->GetNbOfPrecTokens(); ++p)
			{
				Token* previousTokenofFirstToken = firstToken->GetPrecToken(p);
				listPreviousTokenofFirstToken.push_back(previousTokenofFirstToken);
				previousTokenofFirstToken->UnlinkNextToken(firstToken);
			}
		}
	}
	
	// Remove links from the next tokens of the last tokens of the segment
	for(size_t l=0; l<currentSegment->GetNumberOfLastToken(); ++l)
	{
		Token* lastToken = currentSegment->GetLastToken(l);
		
		if(lastToken)
		{
			for(size_t n=0; n<lastToken->GetNbOfNextTokens(); ++n)
			{
				Token* nextTokenofLastToken = lastToken->GetNextToken(n);
				listNextTokenofLastToken.push_back(nextTokenofLastToken);
				nextTokenofLastToken->UnlinkPrevToken(lastToken);
			}
		}
	}
		
	// Re-attach the tokens
	list<Token*>::iterator prev  = listPreviousTokenofFirstToken.begin();
	list<Token*>::iterator eprev = listPreviousTokenofFirstToken.end();
	list<Token*>::iterator next  = listNextTokenofLastToken.begin();
	list<Token*>::iterator enext = listNextTokenofLastToken.end();
	
	while(prev != eprev)
	{
		while(next != enext)
		{
			(*prev)->AddNextToken(*next);
			(*next)->AddPrecToken(*prev);
			
			++next;
		}
		
		++prev;
	}
    
    listPreviousTokenofFirstToken.clear();
    listNextTokenofLastToken.clear();
  	  	  
    // Remove Segment from vector
    vector<Segment*>::iterator SegIter = m_segments.begin();
    	
	while (SegIter != m_segments.end() && (*SegIter) != currentSegment)
		++SegIter;

	if (SegIter == m_segments.end())
    {
		LOG_FATAL(m_pLogger, "Speech::RemoveSegment(), the segment is not at the right spot!!");
		exit(E_INVALID);
	}
	
	m_segments.erase(SegIter);
    
    // destroy! the segment now
    delete currentSegment;
}

int Speech::GetMinTokensTime()
{
	int MinTime = -1;
	
	for(size_t spj=0; spj<NbOfSegments(); ++spj)
	{
		int tmpmin = GetSegment(spj)->GetMinTokensTime();
		
		if ( (MinTime == -1) || (tmpmin < MinTime) )
			MinTime = tmpmin;
	}
	
	return MinTime;
}

int Speech::GetMaxTokensTime()
{
	int MaxTime = -1;
	
	for(size_t spj=0; spj<NbOfSegments(); ++spj)
	{
		int tmpmax = GetSegment(spj)->GetMaxTokensTime();
		
		if ( (MaxTime == -1) || (tmpmax > MaxTime) )
			MaxTime = tmpmax;
	}
	
	return MaxTime;
}
