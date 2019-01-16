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

#include "tokenalignment.h"

// Initialize the static variables
const TokenAlignment::AlignmentResult TokenAlignment::CORRECT(string("C"), string("Correct Alignment"));
const TokenAlignment::AlignmentResult TokenAlignment::SUBSTITUTION(string("S"), string("Substitution: The hypothesis token aligns to a reference token but was not correctly recognized."));
const TokenAlignment::AlignmentResult TokenAlignment::SPEAKERSUB(string("P"), string("Speaker Substitution: The hypothesis token aligns to a reference token, the text matches but the speaker assignment was incorrect."));
const TokenAlignment::AlignmentResult TokenAlignment::DELETION(string("D"), string("Deletion: A reference token was not recognized by the system output."));
const TokenAlignment::AlignmentResult TokenAlignment::INSERTION(string("I"), string("Insertion: The system output a token that didn't exist in the reference."));
const TokenAlignment::AlignmentResult TokenAlignment::REFERENCE(string("R"), string("This token is a reference token."));
const TokenAlignment::AlignmentResult TokenAlignment::INVALID_SYSTEM(string("X"), string("ERROR: No result exists for the specified system name."));
const TokenAlignment::AlignmentResult TokenAlignment::UNAVAILABLE(string("U"), string("The evaluation result for this token hasn't been computed yet."));

const string TokenAlignment::REFERENCE_KEY("REFERENCE");

TokenAlignment::TokenAlignment(Token* refToken) 
{	
	m_alignmentEvaluations[TokenAlignment::REFERENCE_KEY] = new AlignmentEvaluation(refToken, TokenAlignment::REFERENCE);
}

TokenAlignment::~TokenAlignment()
{
	map< string, AlignmentEvaluation* >::iterator i, ei;
	
	i = m_alignmentEvaluations.begin();
	ei = m_alignmentEvaluations.end();
	
	while(i != ei)
	{
		AlignmentEvaluation* ptr_elt = i->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++i;
	}
	
	m_alignmentEvaluations.clear();
}


int TokenAlignment::AddAlignmentFor(const string & hypothesisKey, Token* hypothesisToken) 
{
	// an alignment already exists for this key, don't do anything
	if(m_alignmentEvaluations[hypothesisKey] != NULL)
		return 0;
	
	m_alignmentEvaluations[hypothesisKey] = new AlignmentEvaluation(hypothesisToken);
	return 1;
}

TokenAlignment::AlignmentResult TokenAlignment::GetResultFor(const string & system)
{
	AlignmentEvaluation* res = GetAlignmentFor(system);
	return (res != NULL) ? res->GetResult() : TokenAlignment::INVALID_SYSTEM;
}

Token* TokenAlignment::GetTokenFor(const string & system) 
{
	AlignmentEvaluation* res = GetAlignmentFor(system);
	return (res != NULL) ? res->GetToken() : NULL;
}

Token* TokenAlignment::GetReferenceToken() 
{
	return m_alignmentEvaluations[TokenAlignment::REFERENCE_KEY]->GetToken();
}

string TokenAlignment::ToString() 
{
	t_alignmentMap::iterator
    iter = m_alignmentEvaluations.begin(),
    iter_end = m_alignmentEvaluations.end();
	
	string result;
	
	while( iter != iter_end )
    {
		result += iter->first + " | " + ((AlignmentEvaluation *)iter->second)->ToString() + '\n';
		++iter;
	}
	
	return result;
}

TokenAlignment::AlignmentEvaluation::AlignmentEvaluation(Token* token, const AlignmentResult& result) 
{
	m_token = token;
	m_result = result;
}

string TokenAlignment::AlignmentEvaluation::ToString()
{
	return ((m_token != NULL) ? m_token->GetText() : "*") + " | " + m_result.GetShortName();
}

bool TokenAlignment::AlignmentEvaluation::Equals(AlignmentEvaluation* other)
{
	if(this == other)
		return true;
	if(other == NULL)
		return false;
	return m_token->Equals(other->m_token) && m_result.GetShortName() == other->m_result.GetShortName();
}

TokenAlignment::AlignmentResult::AlignmentResult(const string& shortName, const string& description)
{
	m_shortName = shortName;
	m_description = description;
}
