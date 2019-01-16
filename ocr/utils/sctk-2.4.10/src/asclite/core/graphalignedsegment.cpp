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
 * The result produce by the Graph after a Levenshtein alignment
 */

#include "graphalignedsegment.h"

Logger* GraphAlignedSegment::m_pLogger = Logger::getLogger();

/** class constructor */
GraphAlignedSegment::GraphAlignedSegment(const size_t& _HypRefIndex) : m_HypRefIndex(_HypRefIndex)
{
}

/** class destructor */
GraphAlignedSegment::~GraphAlignedSegment()
{
	vector<GraphAlignedToken*>::iterator i, ei;
	
	i = m_vGraphAlignedTokens.begin();
	ei = m_vGraphAlignedTokens.end();
	
	while(i != ei)
	{
		GraphAlignedToken* elt_ptr = *i;
			
		if(elt_ptr)
			delete elt_ptr;
		
		++i;
	}
	
	m_vGraphAlignedTokens.clear();
}

/** Return the indexed element of the GraphAlignedSegment */
GraphAlignedToken* GraphAlignedSegment::GetGraphAlignedToken(const size_t& index)
{
	if(index < GetNbOfGraphAlignedToken())
    {
		return m_vGraphAlignedTokens[GetNbOfGraphAlignedToken() - index - 1];
	}
    else
    {
        return NULL;
	}
}

string GraphAlignedSegment::ToStringAddnChar(const string& chara, const int& num)
{
	ostringstream outstr;
	
	for(int i=0; i<num; ++i)
		outstr << chara;
	
	return outstr.str();
}

string GraphAlignedSegment::ToStringAddText(const string& text, const int& maxc)
{
	ostringstream outstr;
	
	if (text.length() > 20)
	{
        outstr << "...";
		outstr << ToStringAddnChar(" ", maxc-3);
    }
	else
	{
		outstr << " " << text << " ";
		outstr << ToStringAddnChar(" ", maxc - text.length());
	}
	
	return outstr.str();
}

/** Returns a string representation of this GraphAlignedSegment. */
string GraphAlignedSegment::ToString()
{
	ostringstream osstr;
	size_t i, nbrGAT, GATdim;
	Token* token = NULL;
	long int refSepIndex = m_HypRefIndex - 1;
	
	nbrGAT = GetNbOfGraphAlignedToken();
    
    if (nbrGAT == 0)
    {
        osstr << "!!! No aligned tokens in this graphalignedsegment !!!" << endl;
        return osstr.str();
    }
    
	GATdim = GetGraphAlignedToken(0)->GetDimension();
	
    int k = GATdim - 1;
	
	int* maxsize = new int[nbrGAT];
	
	for(i = 0; i < nbrGAT; ++i)
	{
		int maxsizei = 0;
		
		for(size_t j = 0; j < GATdim ; ++j)
		{
			token = GetGraphAlignedToken(i)->GetToken(j);
			
			if(token != NULL)
			{
				maxsizei = max(maxsizei, (int) token->GetText().length());
            }
			else
			{
				maxsizei = max(maxsizei, 1);
            }
        }
		
		maxsize[i] = maxsizei;
	}
	
	for(long int j = GATdim - 1; j >=  0; --j)
	{
		if(j == refSepIndex)
		{
			osstr << "|-----|";			
			
			for(i=0; i<nbrGAT; ++i)
			{
				osstr << ToStringAddnChar("-", maxsize[i]+2);
				osstr << "|";
			}
			
			osstr << endl;
		}
		else if(j == k)
		{
			osstr << ",-----";
			
			for(i=0; i<nbrGAT; ++i)
			{
				osstr << ToStringAddnChar("-", maxsize[i]+3);
			}
			
			osstr << "." << endl;
		}
		
		osstr << "| " << ((j > refSepIndex) ? "ref" : "hyp") <<  " |";		
		
		for(i = 0; i < nbrGAT; ++i)
		{
			token = GetGraphAlignedToken(i)->GetToken(j);
			
			if(token != NULL)
			{
				osstr << ToStringAddText(token->GetText(), maxsize[i]);
            }
			else
			{
				osstr << ToStringAddText("*", maxsize[i]);
            }
			
			osstr << "|";
		}
		
		osstr << endl;
	}
	
	osstr << "`-----";
	
	for(i=0; i<nbrGAT; ++i)
		osstr << ToStringAddnChar("-", maxsize[i]+3);
	
	osstr << "'" << endl;
	
	return osstr.str();
}

void GraphAlignedSegment::LoggingAlignment(const ulint& seggrpid)
{
	char buffer[BUFFER_SIZE];
	sprintf(buffer, "%li", seggrpid);
	string seggrpsid = string(buffer);

	for(size_t i=0; i < GetNbOfGraphAlignedToken(); ++i)
	{
		GraphAlignedToken* gat = GetGraphAlignedToken(i);
		
		string alignoutput = "";
		string File = "";
		string Channel = "";
		
		for(size_t j=0; j<gat->GetDimension(); ++j)
		{
			Token* gat_token = gat->GetToken(j);
			
			if(gat_token != NULL)
			{
				File = gat_token->GetParentSegment()->GetSource();
				Channel = gat_token->GetParentSegment()->GetChannel();
				alignoutput += "," + gat_token->GetCSVInformation();
			}
			else
			{
				alignoutput += ",,,,,,,,,,,";
			}
		}
				
		LOG_ALIGN(m_pLogger, "YES," + seggrpsid + "," + File + "," + Channel + "," + alignoutput);
	}
}

Token* GraphAlignedSegment::GetNonNullReference(const size_t& gatIndex)
{
	GraphAlignedToken* aGAT = GetGraphAlignedToken(gatIndex);

	if(aGAT	!= NULL)
	{
		Token* outToken;
		
		for(size_t i=m_HypRefIndex; i<aGAT->GetDimension(); ++i)
        {
			outToken = aGAT->GetToken(i);
			
            if(outToken != NULL)
				return outToken;
		}
	}
	
	return NULL;
}

Token* GraphAlignedSegment::GetNonNullHypothesis(const size_t& gatIndex)
{
	GraphAlignedToken* aGAT = GetGraphAlignedToken(gatIndex);

	if(aGAT	!= NULL)
	{
		Token* outToken;
		
		for(size_t i=0; i<m_HypRefIndex; ++i)
        {
			outToken = aGAT->GetToken(i);
            
			if(outToken != NULL)
				return outToken;
		}
	}
	
	return NULL;
}

Token* GraphAlignedSegment::GetPreviousNonNullReference(const size_t& gatIndex)
{
	if (gatIndex == 0)
		return NULL;
    
	Token* result = NULL;
    
	for (long int i = gatIndex - 1; i >= 0; --i)
    {
		result = GetNonNullReference(i);
        
		if (result != NULL)
			return result;
	}
    
	return result;
}

Token* GraphAlignedSegment::GetNextNonNullReference(const size_t& gatIndex)
{
	size_t size = GetNbOfGraphAlignedToken();
    
	if (gatIndex >= size)
		return NULL;
    
	Token* result = NULL;
    
	for (size_t i = gatIndex + 1; i < size; ++i)
    {
		result = GetNonNullReference(i);
        
		if (result != NULL)
			return result;
	}
    
	return result;
}

bool GraphAlignedSegment::operator ==(const GraphAlignedSegment & gas) const
{	
    size_t size = m_vGraphAlignedTokens.size();

    if (size != gas.m_vGraphAlignedTokens.size())
        return false;

    if (m_HypRefIndex != gas.m_HypRefIndex)
        return false;

    for (size_t i=0 ; i < size ; ++i)
        if (*m_vGraphAlignedTokens[i] != *(gas.m_vGraphAlignedTokens[i]))
            return false;
    
    return true;
}

bool GraphAlignedSegment::operator !=(const GraphAlignedSegment & gas) const
{
	return !(gas == *this);
}
