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
 * Internal representation of a segment.
 * A segment is a list of Token.
 */

#include "segment.h" // class's header file
#include "speech.h"
#include "speechset.h"

Logger* Segment::logger = Logger::getLogger();

Segment* Segment::CreateWithDuration(const int& _startTime, const int& _duration, Speech* parent)
{
	Segment* segment = new Segment();
	segment->speech = parent;
	return (Segment*) segment->InitWithDuration(_startTime, _duration);
}

Segment* Segment::CreateWithEndTime(const int& _startTime, const int& _endTime, Speech* parent)
{
	Segment* segment = new Segment();
	segment->speech = parent;
	return (Segment*) segment->InitWithEndTime(_startTime, _endTime);
}

bool Segment::AreStartTimeAndEndTimeValid(const int& _startTime, const int& _endTime)
{
	if(!TimedObject::AreStartTimeAndEndTimeValid(_startTime, _endTime))
		return false;
    
	return _startTime <= GetLowestTokenStartTime() && _endTime >= GetHighestTokenEndTime();
}

/**
 * Merge all Segments in one segments
 */
Segment* Segment::Merge(const vector<Segment*> & segments)
{
	int start=INT_MAX, end=0;
    Token* t_start = NULL;
    Token* t_end = NULL;
	Segment* currentSegment = NULL;
  
    for (size_t i=0 ; i < segments.size() ; ++i)
    {
		currentSegment = segments[i];
    
        if (currentSegment->GetStartTime() < start)
        {
            t_start = currentSegment->f_token;
            start = currentSegment->GetStartTime();
        }
    
        if (currentSegment->GetEndTime() > end)
        {
            t_end = currentSegment->l_token;
            end = currentSegment->GetEndTime();
        }
    }
    
    Segment* seg = Segment::CreateWithEndTime(start, end, currentSegment->GetParentSpeech());
    seg->f_token = t_start;
    seg->l_token = t_end;
    
    return seg;
}

int Segment::GetLowestTokenStartTime()
{
	int lowestStartTime = 0;	
  
    for(size_t i = 0; i < f_token->GetNbOfNextTokens() ; ++i) 
    {
        int sTime = f_token->GetNextToken(i)->GetStartTime();
	  
        if(sTime < lowestStartTime)
            lowestStartTime = sTime;
    }
	
    return lowestStartTime;
}

int Segment::GetHighestTokenEndTime()
{
	int highestEndTime = 0;
  
    for(size_t i = 0; i < l_token->GetNbOfNextTokens() ; ++i) 
    {
        int eTime = l_token->GetNextToken(i)->GetEndTime();
        
        if(eTime < highestEndTime)
            highestEndTime = eTime;
	}
	
    return highestEndTime;
}

// class constructor
Segment::Segment()
{
	//f_token = new Token();
    f_token = Token::CreateWithEndTime(0, 0, this);
	l_token = Token::CreateWithEndTime(0, 0, this);
	ignoreSegmentInScoring = false;
}

// class destructor
Segment::~Segment()
{
	vector<Token*> listAllTokens = ToTopologicalOrderedStruct();
	
	for(size_t i=0; i<listAllTokens.size(); ++i)
	{
		Token* ptr_elt = listAllTokens[i];
		
		if(ptr_elt)
			delete ptr_elt;
	}
	
	listAllTokens.clear();
	
	delete f_token;
	delete l_token;
}

string Segment::ToString() 
{
	std::ostringstream oss;
	oss << "<Segment " << TimedObject::ToString() << ">\n\tspkrId='" << GetSpeakerId()
		<< "' src='" << GetSource() << "' chnl='" << GetChannel() << "'" << endl;
	vector<Token*> tokens = ToTopologicalOrderedStruct();
	Token* token;
	
	for(size_t i = 0; i < tokens.size(); ++i) 
    {
		token = tokens[i];
        
		if(token != NULL)
			oss << "\t" << token->ToString() << endl;
	}	
	
	oss << "</Segment>";
	return oss.str();
}

string Segment::ToStringAsLine()
{
	std::ostringstream oss;
	oss << "<Segment " << TimedObject::ToString() << " spkrId='" << GetSpeakerId()
		<< "' src='" << GetSource() << "' chnl='" << GetChannel() << "'";
	vector<Token*> tokens = ToTopologicalOrderedStruct();
	Token* token;
	
	for(size_t i = 0; i < tokens.size(); ++i)
    {
		token = tokens[i];
        
		if(token != NULL)
            oss << " " << token->ToString();
	}	
	
	return oss.str();
}

// sets the value of token
void Segment::AddFirstToken(Token* token)
{
	if (token->GetStartTime() != -1 && (token->GetStartTime() < this->GetStartTime()))
	{
		char buffer[BUFFER_SIZE];
		sprintf(buffer, "Segment: Tried to add a token with start time before begin of segment file ='%s' tkn='%s' src=%s chnl=%s st=%d et=%d seg: st=%d et=%d",
				GetParentSpeech()->GetParentSpeechSet()->GetSourceFileName().c_str(),
				token->GetText().c_str(), GetSource().c_str(), GetChannel().c_str(), token->GetStartTime(), token->GetEndTime(), GetStartTime(), GetEndTime());
		LOG_FATAL(logger, buffer);
		exit(E_INVALID);
    }
    else
    {
        f_token->AddNextToken(token);
    }
}

// sets the value of token
void Segment::AddLastToken(Token* token)
{
    if (token->GetEndTime() != -1 && (token->GetEndTime() > this->GetEndTime()))
	{
		char buffer[BUFFER_SIZE];
		sprintf(buffer, "Segment: Tried to add a token with start time before begin of segment file ='%s' tkn='%s' src=%s chnl=%s st=%d et=%d seg: st=%d et=%d",
				GetParentSpeech()->GetParentSpeechSet()->GetSourceFileName().c_str(),
				token->GetText().c_str(), GetSource().c_str(), GetChannel().c_str(), token->GetStartTime(), token->GetEndTime(), GetStartTime(), GetEndTime());
		LOG_FATAL(logger, buffer);
		exit(E_INVALID);
    }
    else
    {
        l_token->AddNextToken(token);
    }
}

/**
 * Output a plan version of the segment.
 * Put them on a topological order.
 */
vector<Token*> Segment::ToTopologicalOrderedStruct()
{
	vector<Token*> res;
	
	for(size_t i = 0 ; i < f_token->GetNbOfNextTokens() ; ++i)
    {
        Token* token = f_token->GetNextToken(i);
        res.push_back(token);
        ToTopologicalOrderedStruct(token, &res);
    }
	
    return res;
}

/**
 * Recurs methods to compute a topological order of a graph
 */
void Segment::ToTopologicalOrderedStruct(Token* start, vector<Token*>* doneNode)
{
    if (!isLastToken(start))
    {
        for(size_t i = 0 ; i < start->GetNbOfNextTokens() ; ++i)
        {
            Token* token = start->GetNextToken(i);
            
            if (find(doneNode->begin(), doneNode->end(), token) == doneNode->end())
            {
                doneNode->push_back(token);
                ToTopologicalOrderedStruct(token, doneNode);
            }
        }
    }
}

/**
 * Return if the token is on the list of the First tokens.
 */
bool Segment::isFirstToken(Token* token)
{
	for(size_t i = 0 ; i < f_token->GetNbOfNextTokens() ; ++i)
        if (f_token->GetNextToken(i) == token)
            return true;
    
    return false;
}

/**
 * Return if the token is on the list of the last tokens.
 */
bool Segment::isLastToken(Token* token)
{
	for(size_t i = 0 ; i < l_token->GetNbOfNextTokens() ; ++i)
        if (l_token->GetNextToken(i) == token)
            return true;
  
    return false;
}

/**
 * Return the parent Speech 
 */
Speech* Segment::GetParentSpeech()
{
	return speech;
}

/** Determines if case is taken into account to align Tokens part of this Speech. */
bool Segment::PerformCaseSensitiveAlignment() 
{ 
    return speech->PerformCaseSensitiveAlignment(); 
}
		
/** Determines if fragments are considered as correct when aligning Tokens part of this Speech. */
bool Segment::AreFragmentsCorrect() 
{ 
    return speech->AreFragmentsCorrect(); 
}

/** Determines if optionally deletable Tokens need to be accounted for. */
bool Segment::UseOptionallyDeletable()
{ 
    return speech->UseOptionallyDeletable(); 
}

/** Replaces the token with a linked list of tokens.  The initial token is NOT deleted **/
void Segment:: ReplaceTokenWith(Token *token, const vector<Token*> & startTokens, const vector<Token*> & endTokens)
{
	// Store a list of prev tokens to avoid continual adding of tokens!!!
	vector<Token *>prevTokens;
	for (size_t p=0; p<token->GetNbOfPrecTokens(); ++p)
		prevTokens.push_back(token->GetPrecToken(p));

	// Do the link
	for (size_t s = 0; s<startTokens.size(); ++s)
		for (size_t p=0; p<prevTokens.size(); ++p)
			prevTokens[p]->LinkTokens(startTokens[s]);		
			
	// Link in the Starts
	if (isFirstToken(token))
		for (size_t s = 0; s<startTokens.size(); ++s)
			AddFirstToken(startTokens[s]);
	
	// Store a list of next tokens to avoid continual adding of tokens!!!
	vector<Token *>nextTokens;
	for (size_t n=0; n<token->GetNbOfNextTokens(); ++n)
		nextTokens.push_back(token->GetNextToken(n));
			
	// Do the link
	for (size_t e = 0; e<endTokens.size(); ++e)
		for (size_t n=0; n<nextTokens.size(); ++n)
			endTokens[e]->LinkTokens(nextTokens[n]);					
	
	// Link in the End
	if (isLastToken(token))
		for (size_t e = 0; e<endTokens.size(); ++e)
			AddLastToken(endTokens[e]);
		
	// Unlink the token --- With the assumption that something has replaced IT
	if (isFirstToken(token))
		f_token->UnlinkNextToken(token);
	
	if(isLastToken(token))
		l_token->UnlinkNextToken(token);
	
	while(token->GetNbOfPrecTokens() > 0)
		token->GetPrecToken(0)->UnlinkTokens(token);
	
	while(token->GetNbOfNextTokens() > 0)
		token->UnlinkTokens(token->GetNextToken(0));
		
	// cleanup
	prevTokens.clear();
	nextTokens.clear();
}

int Segment::GetMinTokensTime()
{
	int minTime = -1;
	vector<Token*> vectok = ToTopologicalOrderedStruct();
				
	for(size_t veci=0; veci<vectok.size(); ++veci)
		if( (minTime == -1) || (vectok[veci]->GetStartTime() < minTime) )
			minTime = vectok[veci]->GetStartTime();
		
	vectok.clear();
	
	return minTime;
}

int Segment::GetMaxTokensTime()
{
	int maxTime = -1;
	vector<Token*> vectok = ToTopologicalOrderedStruct();
				
	for(size_t veci=0; veci<vectok.size(); ++veci)
		if( (maxTime == -1) || (vectok[veci]->GetEndTime() > maxTime) )
			maxTime = vectok[veci]->GetStartTime();
		
	vectok.clear();
	
	return maxTime;
}

void  Segment::SetTokensOptionallyDeletable()
{
	vector<Token*> vectok = ToTopologicalOrderedStruct();
			
	for(size_t veci=0; veci<vectok.size(); ++veci)
	{
		 vectok[veci]->BecomeOptionallyDeletable();
	}
	
	vectok.clear();
}
