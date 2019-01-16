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
 
/** Class to aligned tokens returned by the graph */
 
#include "graphalignedtoken.h"

/** Constructor with the number of dimension */
GraphAlignedToken::GraphAlignedToken(const size_t& _dimension)
{
	m_Dimension = _dimension;
	m_TabAlignedTokens = new Token* [m_Dimension];
	
	for(size_t i = 0; i < m_Dimension; ++i)
	{
		m_TabAlignedTokens[i] = NULL;
	}
}

/** Destructor */
GraphAlignedToken::~GraphAlignedToken()
{
	if(m_TabAlignedTokens)
		delete [] m_TabAlignedTokens;
}

/** Set the pointer of a token */
void GraphAlignedToken::SetToken(const size_t& dim, Token* token)
{
	if(dim < GetDimension())
	{
		m_TabAlignedTokens[dim] = token;
    }
	else
	{
		printf("GraphAlignedToken::SetToken()\nInvalid dimension (%li), max: %li\nExiting!\n", dim, m_Dimension);
		exit(E_INVALID);
	}
}

string GraphAlignedToken::ToString()
{
	string result = "[";
	Token* token;
	
	for (size_t i = 0; i < m_Dimension; ++i)
    {
		token = m_TabAlignedTokens[i];
		result += ((token != NULL) ? token->GetText() : "*") + " ";
	}
	
	result += "]";
	
	return result;
}

/**
 * Redefine the == operator to go through all the object for the comparison
 */
bool GraphAlignedToken::operator ==(const GraphAlignedToken & gat) const
{	
    if(m_Dimension != gat.m_Dimension)
        return false;

    for (size_t i=0 ; i < m_Dimension ; ++i)
    {
        Token* left = m_TabAlignedTokens[i];
        Token* right = gat.m_TabAlignedTokens[i];

        if (left == NULL) 
        {
            if(right != NULL) 
                return false;
        } 
        else if (!left->Equals(right)) 
        {
            return false;
        }
    }
    
    return true;
}

bool GraphAlignedToken::operator !=(const GraphAlignedToken & gat) const
{
	if(m_Dimension != gat.m_Dimension)
        return true;
	
    for (size_t i=0 ; i < m_Dimension ; ++i)
    {
        Token* left = m_TabAlignedTokens[i];
        Token* right = gat.m_TabAlignedTokens[i];

        if (left == NULL) 
        {
            if(right != NULL) 
                return true;
        } 
        else if (!left->Equals(right)) 
        {
            return true;
        }
    }
    
    return false;
}
