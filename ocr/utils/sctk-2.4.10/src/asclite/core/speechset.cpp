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
 * A speech set represent all the data from one source (reference, participant)
 * it's a collection of Speech(Speaker) from one source.
 */
 
#include "speechset.h" // class's header file

Logger* SpeechSet::logger = Logger::getLogger();

// class constructor
SpeechSet::SpeechSet(const string& sourceFileName)
{
	ref = false;
	hyp = false;
	gen = false;
	UpdatePropertiesIfNeeded(true);
	fileName = sourceFileName;
	titleName = sourceFileName;
}

void SpeechSet::UpdatePropertiesIfNeeded(const bool& force)
{
	if(force || Properties::IsDirty())
    {
		case_sensitive = (Properties::GetProperty("align.case_sensitive") == "true");
		fragments_are_correct = (Properties::GetProperty("align.fragment_are_correct") == "true");
		string optionally = Properties::GetProperty("align.optionally");
		optionally_deletable = (optionally == "both");
        
		if(!optionally_deletable)
			optionally_deletable = ( (IsRef()) ? optionally == "ref" : optionally == "hyp" );
	}
}

bool SpeechSet::PerformCaseSensitiveAlignment()
{
	UpdatePropertiesIfNeeded(false);
	return case_sensitive;
}

bool SpeechSet::AreFragmentsCorrect()
{
	UpdatePropertiesIfNeeded(false);
	return fragments_are_correct;
}

bool SpeechSet::UseOptionallyDeletable()
{
	UpdatePropertiesIfNeeded(false);
	return optionally_deletable;
}


// class destructor
SpeechSet::~SpeechSet()
{
	m_VectCategoryLabel.clear();
	
	vector<Speech*>::iterator i, ei;

	i = speeches.begin();
	ei = speeches.end();
	
	while(i != ei)
	{
		Speech* ptr_elt = *i;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++i;
	}
	
	speeches.clear();
}

void SpeechSet::AddLabelCategory(const string& type, const string& id, const string& title, const string& desc)
{
	stcCategoryLabel stcCategoryLabel1;
	stcCategoryLabel1.type = type;
	stcCategoryLabel1.id = id;
	stcCategoryLabel1.title = title;
	stcCategoryLabel1.desc = desc;
	m_VectCategoryLabel.push_back(stcCategoryLabel1);
}

/**
 * Set the hyp/ref status of this set
 */
void SpeechSet::SetOrigin(const string& status)
{
    if(status == "ref")
    {
        ref = true;
		hyp = false;
		gen = false;
    }
    else if(status == "hyp")
    {
        hyp = true;
		ref= false;
		gen = false;
    }
    else if(status == "gen")
    {
    	hyp = true;
		ref= false;
		gen = true;
    }
    else
    {
        LOG_WARN(logger, "The status of the SpeechSet dont exist (must be 'ref' or 'hyp') and was: "+status);
    }
	
    UpdatePropertiesIfNeeded(true);
}

bool SpeechSet::HasInterSegmentGap()
{
	for(size_t spseti = 0; spseti < GetNumberOfSpeech(); ++spseti)
		for(size_t i=0; i<GetSpeech(spseti)->NbOfSegments(); ++i)
			if(GetSpeech(spseti)->GetSegment(i)->GetSpeakerId().compare(string("inter_segment_gap")) == 0)
				return true;
					
	return false;
}

int SpeechSet::GetMinTokensTime()
{
	int MinTime = -1;
	
	for(size_t spseti = 0; spseti < GetNumberOfSpeech(); ++spseti)
	{
		int tmpmin = GetSpeech(spseti)->GetMinTokensTime();
		
		if ( (MinTime == -1) || (tmpmin < MinTime) )
			MinTime = tmpmin;
	}
	
	return MinTime;
}

int SpeechSet::GetMaxTokensTime()
{
	int MaxTime = -1;
	
	for(size_t spseti = 0; spseti < GetNumberOfSpeech(); ++spseti)
	{
		int tmpmax = GetSpeech(spseti)->GetMaxTokensTime();
		
		if ( (MaxTime == -1) || (tmpmax > MaxTime) )
			MaxTime = tmpmax;
	}
	
	return MaxTime;
}
