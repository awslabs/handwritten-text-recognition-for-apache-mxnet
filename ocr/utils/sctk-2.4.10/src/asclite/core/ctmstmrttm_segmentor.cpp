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
 * Implementation of a segmentor that take rttm hyp and rttm reference.
 */

#include "ctmstmrttm_segmentor.h" // class's header file

Logger* CTMSTMRTTMSegmentor::logger = Logger::getLogger();

// class destructor
CTMSTMRTTMSegmentor::~CTMSTMRTTMSegmentor()
{
	sourceList.clear();

	map<string, set<string> >::iterator i, ei;
	
	i = channelList.begin();
	ei = channelList.end();
	
	while(i != ei)
	{
		i->second.clear();
		++i;
	}
  
	channelList.clear();
	
	m_VectSpeechSet.clear();
}

void CTMSTMRTTMSegmentor::Reset(SpeechSet* references, SpeechSet* hypothesis)
{
    if (references != refs)
    {
        refs = references;
        //init the source and Channel list
        sourceList.clear();
        channelList.clear();
    
        for (size_t i=0 ; i < references->GetNumberOfSpeech(); ++i)
        {
            for (size_t j=0 ; j < references->GetSpeech(i)->NbOfSegments() ; ++j)
            {
                Segment* seg = references->GetSpeech(i)->GetSegment(j);
                sourceList.insert(seg->GetSource());
                channelList[seg->GetSource()].insert(seg->GetChannel());
            }
        }
    }
	
    hyps = hypothesis;
    
    //init the iterator
    currentSource = *(sourceList.begin());
    sourceList.erase(currentSource);
    
    currentChannel = *(channelList[currentSource].begin());
    channelList[currentSource].erase(currentChannel);
    
    LOG_INFO(logger, "Prepare to process source("+currentSource+") channel("+currentChannel+")");

    currentSegmentRef = GetFirstSegment(0, references);
}

void CTMSTMRTTMSegmentor::ResetGeneric(map<string, SpeechSet*> &mapspeechSet)
{
	sourceList.clear();
	channelList.clear();
	m_VectSpeechSet.clear();
        
	map<string, SpeechSet*>::iterator k = mapspeechSet.begin();
	map<string, SpeechSet*>::iterator ek = mapspeechSet.end();
	
	while(k != ek)
	{
		m_VectSpeechSet.push_back(k->second);
		
		for (size_t i=0 ; i < k->second->GetNumberOfSpeech(); ++i)
        {
            for (size_t j=0 ; j < k->second->GetSpeech(i)->NbOfSegments() ; ++j)
            {
                Segment* seg = k->second->GetSpeech(i)->GetSegment(j);
                sourceList.insert(seg->GetSource());
                channelList[seg->GetSource()].insert(seg->GetChannel());
            }
        }
        
		++k;
	}
	
	//init the iterator
    currentSource = *(sourceList.begin());
    sourceList.erase(currentSource);
    
    currentChannel = *(channelList[currentSource].begin());
    channelList[currentSource].erase(currentChannel);
    
    LOG_INFO(logger, "Prepare to process source("+currentSource+") channel("+currentChannel+")");

    currentSegmentRef = GetFirstSegmentGeneric(0);
}

SegmentsGroup* CTMSTMRTTMSegmentor::NextGeneric()
{
	SegmentsGroup* segGroup = new SegmentsGroup();

    Segment* lastOverlapingSeg = GetLastOverlapingSegmentGeneric(/*currentSegmentRef->GetStartTime()*/);
    
    // Retrieve segments between current time and next blank
    for(size_t v=0; v<m_VectSpeechSet.size(); ++v)
	{
		for (size_t i=0 ; i<m_VectSpeechSet[v]->GetNumberOfSpeech(); ++i)
		{
			vector<Segment*> v_segs = m_VectSpeechSet[v]->GetSpeech(i)->GetSegmentsByTime(currentSegmentRef->GetStartTime(), lastOverlapingSeg->GetEndTime(), currentSource, currentChannel);
	
			if (v_segs.size() != 0)
				segGroup->AddHypothesis(v_segs);
		}
	}
	
    //Prepare the next iteration
    currentSegmentRef = GetFirstSegmentGeneric(lastOverlapingSeg->GetEndTime());
	
    if (currentSegmentRef == NULL)
    {
        if (!channelList[currentSource].empty())
        {
            currentChannel = *(channelList[currentSource].begin());
            LOG_INFO(logger, "Prepare to process source("+currentSource+") channel("+currentChannel+")");
            channelList[currentSource].erase(currentChannel);
            currentSegmentRef = GetFirstSegmentGeneric(0);   
        } 
        else if (!sourceList.empty())
		{
			currentSource = *(sourceList.begin());
			sourceList.erase(currentSource);
			currentChannel = *(channelList[currentSource].begin());
			channelList[currentSource].erase(currentChannel);
			LOG_INFO(logger, "Prepare to process source("+currentSource+") channel("+currentChannel+")");
			currentSegmentRef = GetFirstSegmentGeneric(0);
		}
    }
	    
    return segGroup;
}

SegmentsGroup* CTMSTMRTTMSegmentor::Next()
{
	SegmentsGroup* segGroup = new SegmentsGroup();

    // Retrieve time of the next blank
    Segment* lastOverlapingSeg = GetLastOverlapingSegment(/*currentSegmentRef->GetStartTime(), */refs);
    //cout << "from:" << currentSegment->GetStartTime() << endl;
    //cout << "los seg("<< lastOverlapingSeg->GetStartTime() << "," << lastOverlapingSeg->GetEndTime() << ")" << endl;
    //cout << "Nbr of Speeches: " << refs->GetNumberOfSpeech() << endl;

    // Retrieve segments between current time and next blank
    for (size_t i=0 ; i < refs->GetNumberOfSpeech() ; ++i)
	{
        vector<Segment*> v_segs = refs->GetSpeech(i)->GetSegmentsByTime(currentSegmentRef->GetStartTime(), lastOverlapingSeg->GetEndTime(), currentSource, currentChannel);
        //cout << "nb of ref: " << v_segs.size() << endl;
        if (v_segs.size() != 0)
            segGroup->AddReference(v_segs);
    }
  
    for (size_t i=0 ; i < hyps->GetNumberOfSpeech(); ++i)
	{
        vector<Segment*> v_segs = hyps->GetSpeech(i)->GetSegmentsByTime(currentSegmentRef->GetStartTime(), lastOverlapingSeg->GetEndTime(), currentSource, currentChannel);
        //cout << "nb of hyp: " << v_segs.size() << endl;
        if (v_segs.size() != 0)
            segGroup->AddHypothesis(v_segs);
    }
  
    //Prepare the next iteration
    //cout << lastOverlapingSeg->GetEndTime() << endl;
    currentSegmentRef = GetFirstSegment(lastOverlapingSeg->GetEndTime(), refs);
	
    //cout << "cur seg("<< currentSegment->GetStartTime() << "," << currentSegment->GetEndTime() << ")" << endl;
    if (currentSegmentRef == NULL)
    {
        if (!channelList[currentSource].empty())
        {
            currentChannel = *(channelList[currentSource].begin());
            LOG_INFO(logger, "Prepare to process source("+currentSource+") channel("+currentChannel+")");
            channelList[currentSource].erase(currentChannel);
            currentSegmentRef = GetFirstSegment(0, refs);   
        } 
        else
        {
            if (!sourceList.empty())
            {
                currentSource = *(sourceList.begin());
                sourceList.erase(currentSource);
                currentChannel = *(channelList[currentSource].begin());
                channelList[currentSource].erase(currentChannel);
                LOG_INFO(logger, "Prepare to process source("+currentSource+") channel("+currentChannel+")");
                currentSegmentRef = GetFirstSegment(0, refs);
            }
            /*
            else
            {
                // currentSegment NULL mean end of loop in this case
            }
            */
        }
    }
	    
    return segGroup;
}

Segment* CTMSTMRTTMSegmentor::GetLastOverlapingSegment(/*int startTime, */SpeechSet* speechs)
{
    //find the next time when there is no segments
    Segment* last = currentSegmentRef;
    bool again = true;
    
    while (again)
    {
        again = false;
        
        for (size_t i=0 ; i < speechs->GetNumberOfSpeech() ; ++i)
        {
            //cout << "get next seg start in : " << last->GetEndTime() << endl;
            Segment* t_seg = speechs->GetSpeech(i)->NextSegment(last->GetEndTime(), currentSource, currentChannel);
            
            if (t_seg != NULL)
            {
                //cout << "got seg start on " << t_seg->GetStartTime() << endl;
               if (t_seg->GetStartTime() < last->GetEndTime())
                {
                    last = t_seg;
                    again = true;
                }
            }
        }
    }
    
    return last;
}

Segment* CTMSTMRTTMSegmentor::GetLastOverlapingSegmentGeneric(/*int startTime*/)
{
    //find the next time when there is no segments
    Segment* last = currentSegmentRef;
    bool again = true;
    
    while (again)
    {
        again = false;
        
        for(size_t v=0; v<m_VectSpeechSet.size(); ++v)
		{
			for (size_t i=0 ; i < m_VectSpeechSet[v]->GetNumberOfSpeech() ; ++i)
			{
				Segment* t_seg = m_VectSpeechSet[v]->GetSpeech(i)->NextSegment(last->GetEndTime(), currentSource, currentChannel);
				
				if (t_seg != NULL)
				{
				   if (t_seg->GetStartTime() < last->GetEndTime())
					{
						last = t_seg;
						again = true;
					}
				}
			}
    	}
    }
    
    return last;
}

Segment* CTMSTMRTTMSegmentor::GetFirstSegment(const int& startTime, SpeechSet* speechs)
{
    int min_time = INT_MAX;
	Segment* retSegment = NULL;
	
    for (size_t i=0 ; i < speechs->GetNumberOfSpeech() ; ++i)
	{
        Segment* t_seg = speechs->GetSpeech(i)->NextSegment(startTime, currentSource, currentChannel);
        
        if (t_seg != NULL)
        {
            if (t_seg->GetStartTime() < min_time)
            {
                retSegment = t_seg;
                min_time = t_seg->GetStartTime(); 
            }
        }
    }
		
    return retSegment;
}

Segment* CTMSTMRTTMSegmentor::GetFirstSegmentGeneric(const int& startTime)
{
    int min_time = INT_MAX;
	Segment* retSegment = NULL;
	
	for(size_t v=0; v<m_VectSpeechSet.size(); ++v)
	{
		SpeechSet* speechs = m_VectSpeechSet[v];
		
		for (size_t i=0 ; i < speechs->GetNumberOfSpeech() ; ++i)
		{
			Segment* t_seg = speechs->GetSpeech(i)->NextSegment(startTime, currentSource, currentChannel);
			
			if (t_seg != NULL)
			{
				if (t_seg->GetStartTime() < min_time)
				{
					retSegment = t_seg;
					min_time = t_seg->GetStartTime(); 
				}
			}
		}
    }
		
    return retSegment;
}
