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
 * This class represent a group of segments with the information
 *  about ref/hyp if this information exist.
 */
 
#include "segmentsgroup.h"

Logger* SegmentsGroup::logger = Logger::getLogger();

// class destructor
SegmentsGroup::~SegmentsGroup()
{
	size_t i;
	
	for(i=0; i<references.size(); ++i)
		references[i].clear();
	
	references.clear();
	
	for(i=0; i<hypothesiss.size(); ++i)
		hypothesiss[i].clear();
	
	hypothesiss.clear();
}

/**
 * Add a reference segment into the SegmentGroup.
 */
void SegmentsGroup::AddReference(Segment* reference)
{
    vector<Segment*> temp;
    temp.push_back(reference);
	references.push_back(temp);
}

/*
 * Add an Hypothesis into the SegmentGroup.
 */
void SegmentsGroup::AddHypothesis(Segment* hypothesis)
{
    vector<Segment*> temp;
    temp.push_back(hypothesis);
	hypothesiss.push_back(temp);
}

/**
 * Return true if the Group contains segment to ignore in scoring
 */
bool SegmentsGroup::isIgnoreInScoring()
{
    for (size_t i=0 ; i < references.size() ; ++i)
    {
        for (size_t j=0 ; j < references[i].size() ; ++j)
        {
            if (references[i][j]->isSegmentExcludeFromScoring())
                return true;
        }
    }
  
    return false;
}

/**
 * Return a possible Reference for the given Hyp token.
 * The Segment returned is the first that match the token time
 * This method should always return a Segment as an Reference should always contain a Segment at a given time.
 */
Segment* SegmentsGroup::GetRefSegmentByTime(Token* token)
{
    for(size_t i=0; i < references.size(); ++i)
    {
        for(size_t j=0; j < references[i].size(); ++j)
        {
			if(references[i][j]->OverlapWith(token))
                return references[i][j];
        }
    }
    
    //Hey why are you here ?
    LOG_FATAL(logger, "Warning: This token don't have a reference segment for himself:");
    cerr << token->GetText() <<" (" << token->GetStartTime() << "," << token->GetEndTime() << ")"<< endl;
    cerr << "from segment(" << token->GetParentSegment()->GetStartTime() << "," << token->GetParentSegment()->GetEndTime() << ") ";
    cerr << token->GetParentSegment()->GetSource() << ", " << token->GetParentSegment()->GetChannel() << ", " << token->GetParentSegment()->GetSpeakerId() << endl;
    exit(E_INVALID);
	
    return NULL;
}


string SegmentsGroup::GetDifficultyString()
{
	string res;
	char buff1[100];
    
	sprintf(buff1, "REF:%lu(", (ulint) references.size());
	res.append(buff1);
    for (size_t i=0 ; i < references.size() ; ++i)
    {
		size_t temp_size = 0;
        for (size_t j=0 ; j < references[i].size() ; ++j)
        {
            temp_size += references[i][j]->ToTopologicalOrderedStruct().size();
		}
		sprintf(buff1,"%lu", (ulint) temp_size);
		if (i != 0) res.append(",");
		res.append(buff1);
    }
	sprintf(buff1, ") HYP:%lu(", (ulint) hypothesiss.size());
	res.append(buff1);

    for (size_t i=0 ; i < hypothesiss.size() ; ++i)
    {
		size_t temp_size = 0;
        
        for (size_t j=0 ; j < hypothesiss[i].size() ; ++j)
        {
            temp_size += hypothesiss[i][j]->ToTopologicalOrderedStruct().size();
        }
    
		if (i != 0) res.append(",");
		sprintf(buff1,"%lu", (ulint) temp_size);
		res.append(buff1);
    }
	res.append(")");
    
    return res;
}

ullint SegmentsGroup::GetDifficultyNumber()
{
    ullint res = 1;
    
    for (size_t i=0 ; i < references.size() ; ++i)
    {
        ullint temp_size = 0;
    
        for (size_t j=0 ; j < references[i].size() ; ++j)
        {
            temp_size += (ullint) references[i][j]->ToTopologicalOrderedStruct().size();
		}
        
		res = res * (temp_size+1);
    }
    
    for (size_t i=0 ; i < hypothesiss.size() ; ++i)
    {
		ullint temp_size = 0;
        
        for (size_t j=0 ; j < hypothesiss[i].size() ; ++j)
        {
            temp_size += (ullint) hypothesiss[i][j]->ToTopologicalOrderedStruct().size();
        }
    
		res = res * (temp_size+1);
    }
    
    return res;
}


/**
 * Return the List of Token by topological order for the Reference no i
 */
vector<Token*> SegmentsGroup::ToTopologicalOrderedStructRef(const size_t& index)
{
    vector<Token*> tokens;
	
	tokens.push_back(NULL);
	
	for (size_t i=0 ; i < references[index].size() ; ++i)
    {
        vector<Token*> temp = references[index][i]->ToTopologicalOrderedStruct();

		for (size_t j=0; j<temp.size(); ++j)
			tokens.push_back(temp[j]);
    }
	
	tokens.push_back(NULL);
	
    return tokens;
}

/**
 * Return the List of Token by topological order for the Hypothesis no i
 */
vector<Token*> SegmentsGroup::ToTopologicalOrderedStructHyp(const size_t& index)
{
    vector<Token*> tokens;
	
	tokens.push_back(NULL);
	
    for (size_t i=0 ; i < hypothesiss[index].size() ; ++i)
    {
        vector<Token*> temp = hypothesiss[index][i]->ToTopologicalOrderedStruct();
                
		for (size_t j=0; j<temp.size(); ++j)
			tokens.push_back(temp[j]); 
    }
	
	tokens.push_back(NULL); 
	
    return tokens;
}

/** Returns a string representation of this SegmentGroup. */
string SegmentsGroup::ToString()
{
	ostringstream osstr;
	
    osstr << "References:" << endl;
	
    for (size_t i = 0 ; i < references.size() ; ++i)
    {
        for (size_t j = 0 ; j < references[i].size() ; ++j)
        {
            vector<Token*> temp = references[i][j]->ToTopologicalOrderedStruct();
            osstr << "  spkr=" << i << " j=" << j << " #tokens=" << temp.size() << " ";
            osstr << "[";
			
        for (size_t k=0 ; k < temp.size() ; ++k)
        {
            osstr << " " << temp[k]->ToString();
        }
			
        osstr << " ]" << endl;
        }
	}
	
	osstr << "Hypotheses:" << endl;
	
    for (size_t i = 0 ; i < hypothesiss.size() ; ++i)
    {
        for (size_t j = 0 ; j < hypothesiss[i].size() ; ++j)
        {
            vector<Token*> temp = hypothesiss[i][j]->ToTopologicalOrderedStruct();
            osstr << "  spkr=" << i << " j=" << j << " #tokens=" << temp.size() << " ";;
            osstr << "[";
			
            for (size_t k=0 ; k < temp.size() ; ++k)
            {
                osstr << " " << temp[k]->ToString();
            }
			
            osstr << " ]" << endl;
        }
    }
	
	return osstr.str();
}

/** Log display Alignment */
void SegmentsGroup::LoggingAlignment(const bool& bgeneric, const string& type)
{
	char buffer1[BUFFER_SIZE];
	sprintf(buffer1, "%li", s_id);
	string seggrpsid = string(buffer1);
	
	//References
	for (size_t i = 0 ; i < references.size() ; ++i)
    {
        for (size_t j = 0 ; j < references[i].size() ; ++j)
        {
            vector<Token*> listTkn = references[i][j]->ToTopologicalOrderedStruct();
			
			if(!listTkn.empty())
			{
				vector<Token*>::iterator  k=listTkn.begin();
				vector<Token*>::iterator ek=listTkn.end();
				
				string csvHyp = ",,,,,,,,,,";
				
				if(bgeneric)
					csvHyp = "";
					
				string csvRef = ",,,,,,,,,,";
				
				while(k != ek)
				{
					csvRef = (*k)->GetCSVInformation();
					string File = (*k)->GetParentSegment()->GetSource();
					string Channel = (*k)->GetParentSegment()->GetChannel();
					LOG_ALIGN(logger, type + "," + seggrpsid + "," + File + "," + Channel + ",," + csvRef + "," + csvHyp);
					++k;
				}
			}
			else
			{
				string csvHyp = ",,,,,,,,,,";
				
				if(bgeneric)
					csvHyp = "";
					
				string csvRef = ",,,,,,,,,,";
				
				string segbt = "";
	
				if(references[i][j]->GetStartTime() >= 0)
				{
					char buffer [BUFFER_SIZE];
					sprintf(buffer, "%.3f", ((double)(references[i][j]->GetStartTime()))/1000.0);
					segbt = string(buffer);
				}
				
				string seget = "";
				
				if(references[i][j]->GetEndTime() >= 0)
				{
					char buffer [BUFFER_SIZE];
					sprintf(buffer, "%.3f", ((double)(references[i][j]->GetEndTime()))/1000.0);
					seget = string(buffer);
				}
				
				string spkr = references[i][j]->GetSpeakerId();
				
				char buffer2[BUFFER_SIZE];
				sprintf(buffer2, "%li", references[i][j]->GetsID());
				string segsid = string(buffer2);
				
				string File = references[i][j]->GetSource();
				string Channel = references[i][j]->GetChannel();
				
				csvRef = segsid + "," + segbt + "," + seget + "," + spkr + ",,,,,,,";
				
				LOG_ALIGN(logger, type + "," + seggrpsid + "," + File + "," + Channel + ",," + csvRef + "," + csvHyp);
			}
		}
	}
	
	//Hypotheses
	for (size_t i = 0 ; i < hypothesiss.size() ; ++i)
    {
        for (size_t j = 0 ; j < hypothesiss[i].size() ; ++j)
        {
            vector<Token*> listTkn = hypothesiss[i][j]->ToTopologicalOrderedStruct();
			
			if(!listTkn.empty())
			{
				vector<Token*>::iterator  k=listTkn.begin();
				vector<Token*>::iterator ek=listTkn.end();
				string csvHyp = ",,,,,,,,,,";
				string csvRef = ",,,,,,,,,,";
				
				while(k != ek)
				{
					csvHyp = (*k)->GetCSVInformation();
					string File = (*k)->GetParentSegment()->GetSource();
					string Channel = (*k)->GetParentSegment()->GetChannel();
					LOG_ALIGN(logger, type + "," + seggrpsid + "," + File + "," + Channel + ",," + csvRef + "," + csvHyp);
					++k;
				}
			}
			else
			{
				string csvHyp = ",,,,,,,,,,";
				string csvRef = ",,,,,,,,,,";
				
				string segbt = "";
	
				if(hypothesiss[i][j]->GetStartTime() >= 0)
				{
					char buffer [BUFFER_SIZE];
					sprintf(buffer, "%.3f", ((double)(hypothesiss[i][j]->GetStartTime()))/1000.0);
					segbt = string(buffer);
				}
				
				string seget = "";
				
				if(hypothesiss[i][j]->GetEndTime() >= 0)
				{
					char buffer [BUFFER_SIZE];
					sprintf(buffer, "%.3f", ((double)(hypothesiss[i][j]->GetEndTime()))/1000.0);
					seget = string(buffer);
				}
				
				string spkr = hypothesiss[i][j]->GetSpeakerId();
				
				char buffer2[BUFFER_SIZE];
				sprintf(buffer2, "%li", hypothesiss[i][j]->GetsID());
				string segsid = string(buffer2);
				
				string File = hypothesiss[i][j]->GetSource();
				string Channel = hypothesiss[i][j]->GetChannel();
				
				csvHyp = segsid + "," + segbt + "," + seget + "," + spkr + ",,,,,,,";
				
				LOG_ALIGN(logger, type + "," + seggrpsid + "," + File + "," + Channel + ",," + csvRef + "," + csvHyp);
			}
        }
    }
}

int SegmentsGroup::GetMinTime()
{
	int minBTSG = INT_MAX;
		
	for(size_t i=0; i<GetNumberOfReferences(); ++i)
	{
		vector<Segment*> vecSeg = GetReference(i);
		
		for(size_t j=0; j<vecSeg.size(); ++j)
		{
			if(vecSeg[j]->GetStartTime() < minBTSG)
				minBTSG = vecSeg[j]->GetStartTime();
		}
	}
	
	for(size_t i=0; i<GetNumberOfHypothesis(); ++i)
	{
		vector<Segment*> vecSeg = GetHypothesis(i);
		
		for(size_t j=0; j<vecSeg.size(); ++j)
		{
			if(vecSeg[j]->GetStartTime() < minBTSG)
				minBTSG = vecSeg[j]->GetStartTime();
		}
	}
	
	return(minBTSG);
}

int SegmentsGroup::GetMaxTime()
{
	int maxETSG = 0;
		
	for(size_t i=0; i<GetNumberOfReferences(); ++i)
	{
		vector<Segment*> vecSeg = GetReference(i);
		
		for(size_t j=0; j<vecSeg.size(); ++j)
		{
			if(vecSeg[j]->GetEndTime() > maxETSG)
				maxETSG = vecSeg[j]->GetEndTime();
		}
	}
	
	for(size_t i=0; i<GetNumberOfHypothesis(); ++i)
	{
		vector<Segment*> vecSeg = GetHypothesis(i);
		
		for(size_t j=0; j<vecSeg.size(); ++j)
		{
			if(vecSeg[j]->GetEndTime() > maxETSG)
				maxETSG = vecSeg[j]->GetEndTime();
		}
	}
	
	return(maxETSG);
}

int SegmentsGroup::GetTotalDuration()
{
	int minBTSG = GetMinTime();
	int maxETSG = GetMaxTime();
	int diff = maxETSG - minBTSG;
	
	if(diff <= 0)
	{
		LOG_FATAL(logger, "Duration of the Segment group is negative or equal 0");
		exit(E_INVALID);
	}
	
	return(diff);
}
