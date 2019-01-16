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

#include "uemfilter.h"

Logger* UEMElement::m_pLogger = Logger::getLogger();
Logger* UEMFilter::m_pLogger = Logger::getLogger();

UEMElement::UEMElement(const string& _file, const string& _channel, const int& _startTime, const int& _endTime)
{
	m_File = _file;
	m_Channel = _channel;
	m_StartTime = _startTime;
	m_EndTime = _endTime;
}

UEMFilter::~UEMFilter()
{
	for(size_t i=0; i<m_VectUEMElements.size(); ++i)
		if(m_VectUEMElements[i])
			delete m_VectUEMElements[i];
	
	m_VectUEMElements.clear();
}

void UEMFilter::FindElement(const string& file, const string& channel, list<UEMElement*>* pList)
{
	pList->clear();

	for(size_t i=0; i<m_VectUEMElements.size(); ++i)
		if(m_VectUEMElements[i])
			if( (file.find(m_VectUEMElements[i]->GetFile(), 0) == 0) && (channel.compare(m_VectUEMElements[i]->GetChannel()) == 0) )
				pList->push_back(m_VectUEMElements[i]);
}

void UEMFilter::LoadFile(const string& filename)
{
	ifstream file;
	string line;
	file.open(filename.c_str(), ifstream::in);
	
	long int lineNum = -1;
    
	if (! file.is_open())
	{ 
		LOG_FATAL(m_pLogger, "Error opening file " + filename); 
		exit(E_LOAD); 
	}
	
	char l_file[BUFFER_SIZE];
	char l_channel[BUFFER_SIZE];
	char l_start[BUFFER_SIZE];
	char l_end[BUFFER_SIZE];
	
	while (getline(file,line,'\n'))
	{
		 ++lineNum;
	
		if (line.find_first_of(";;") == 0)
		{
			//comment so skip (for now)
		}
		else
		{
			int nbArgParsed = 0;
			nbArgParsed = sscanf(line.c_str(), "%s %s %s %s", (char*) &l_file, (char*) &l_channel, (char*) &l_start, (char*) &l_end);
			 
			if(nbArgParsed != 4)
			{
				char buffer[BUFFER_SIZE];
				sprintf(buffer, "Error parsing the line %li in file %s", lineNum, filename.c_str());
				LOG_ERR(m_pLogger, buffer);
			}
			else
			{
				int start_ms = ParseString(string(l_start));
				int end_ms = ParseString(string(l_end));
				
				if(start_ms < end_ms)
				{
					UEMElement* pUEMElement = new UEMElement(string(l_file), string(l_channel), start_ms, end_ms);
					AddUEMElement(pUEMElement);
				}
				else
				{
					char buffer[BUFFER_SIZE];
					sprintf(buffer, "The time is not proper at the line %li in file %s: begin time %s and endtime %s", lineNum, filename.c_str(), l_start, l_end);
					LOG_ERR(m_pLogger, buffer);
				}
			}
		}
	}
    
	LOG_INFO(m_pLogger, "loading of file '" + filename + "' done");
	file.close();
    
	if(isEmpty())
	{
		LOG_FATAL(m_pLogger, "UEM file '" + filename + "' contains no data!");
		exit(E_LOAD);
	}
	
	m_bUseFile = true;
}

unsigned long int UEMFilter::ProcessSingleSpeech(Speech* speech)
{
	ulint nbrerr = 0;
	list<Segment*> listSegmentsToRemove;
	
	// Step 1: checking if the input is proper and listing segments to remove
	if( (speech->GetParentSpeechSet()->IsRef()) || (speech->GetParentSpeechSet()->IsGen()) )
	{
		// It's a Ref so check the bad ones
		for(size_t segindex=0; segindex<speech->NbOfSegments(); ++segindex)
		{
			Segment* pSegment = speech->GetSegment(segindex);
		
			string segFile = pSegment->GetSource();
			string segChannel = pSegment->GetChannel();
			int segStartTime = pSegment->GetStartTime();
			int segEndTime = pSegment->GetEndTime();
			
			list<UEMElement*>* pListUEMElement = new list<UEMElement*>;
			FindElement(segFile, segChannel, pListUEMElement);
			
			if(pListUEMElement->empty())
			{
				char bufferUEM0[BUFFER_SIZE];
				sprintf(bufferUEM0, "UEMFilter - Removing segment in '%s/%s' with times [%d, %d] because nothing has been defined for these file and channel in UEM file", segFile.c_str(), segChannel.c_str(), segStartTime, segEndTime);
				LOG_INFO(m_pLogger, bufferUEM0);
				listSegmentsToRemove.push_back(pSegment);
			}
			else
			{
				list<UEMElement*>::iterator  i = pListUEMElement->begin();
				list<UEMElement*>::iterator ei = pListUEMElement->end();
				bool keep = false;
				
				while(i != ei)
				{
					UEMElement* pUEMElement = *i;
					int uemStartTime = pUEMElement->GetStartTime();
					int uemEndTime = pUEMElement->GetEndTime();
					
					keep = ( (uemStartTime <= segStartTime) && (segEndTime <= uemEndTime) );
					// [     {      }     ]
					// us    ss     se    ue
										
					if(!keep)
					{
						// 1
						// {     [      }     ]
						// ss    us     se    ue
					
						// 2
						// {     [      ]     }
						// ss    us     ue    se
						
						// 3
						// [     {      ]     }
						// us    ss     ue    se
						
						if( /*1*/( (segStartTime < uemStartTime) && (uemStartTime < segEndTime) && (segEndTime < uemEndTime) ) ||
							/*2*/( (segStartTime < uemStartTime) && (uemStartTime < uemEndTime) && (uemEndTime < segEndTime) ) ||
							/*3*/( (uemStartTime < segStartTime) && (segStartTime < uemEndTime) && (uemEndTime < segEndTime) ) )
						{
							++nbrerr;
							char bufferUEM1[BUFFER_SIZE];
							sprintf(bufferUEM1, "UEMFilter - Segment in '%s/%s' has an unproper time [%d, %d] regarding the UEM file with times: (%s/%s) [%d, %d]", segFile.c_str(), segChannel.c_str(), segStartTime, segEndTime,  pUEMElement->GetFile().c_str(), pUEMElement->GetChannel().c_str(), uemStartTime, uemEndTime);
							LOG_ERR(m_pLogger, bufferUEM1);
						}
						
						++i;
					}
					else
					{
						i = ei;
					}
				}
				
				if(!keep)
				{
					char bufferUEM2[BUFFER_SIZE];
					sprintf(bufferUEM2, "UEMFilter - Removing segment in '%s/%s' with times [%d, %d] regarding the UEM file rules", segFile.c_str(), segChannel.c_str(), segStartTime, segEndTime);
					LOG_INFO(m_pLogger, bufferUEM2);
					listSegmentsToRemove.push_back(pSegment);
				}
			}
			
			pListUEMElement->clear();
			delete pListUEMElement;
		}
	}
	else if(speech->GetParentSpeechSet()->IsHyp())
	{
		// It's a Hyp so just remove them regarding the mid point of every token.
		for(size_t segindex=0; segindex<speech->NbOfSegments(); ++segindex)
		{
			Segment* pSegment = speech->GetSegment(segindex);
			int segMidPoint = (pSegment->GetStartTime() + pSegment->GetEndTime())/2;
			string segFile = pSegment->GetSource();
			string segChannel = pSegment->GetChannel();
			
			list<UEMElement*>* pListUEMElement = new list<UEMElement*>;
			FindElement(segFile, segChannel, pListUEMElement);
			
			if(pListUEMElement->empty())
			{
				char bufferUEM0[BUFFER_SIZE];
				sprintf(bufferUEM0, "UEMFilter - Removing segment in '%s/%s' with times [%d, %d] (mid: %d) because nothing has been defined for these file and channel in UEM file", segFile.c_str(), segChannel.c_str(), pSegment->GetStartTime(), pSegment->GetEndTime(), segMidPoint);
				LOG_INFO(m_pLogger, bufferUEM0);
				listSegmentsToRemove.push_back(pSegment);
			}
			else
			{
				list<UEMElement*>::iterator  i = pListUEMElement->begin();
				list<UEMElement*>::iterator ei = pListUEMElement->end();
				bool keep = false;
				
				while(i != ei)
				{
					UEMElement* pUEMElement = *i;
					int uemStartTime = pUEMElement->GetStartTime();
					int uemEndTime = pUEMElement->GetEndTime();
					keep = ( (uemStartTime <= segMidPoint) && (segMidPoint <= uemEndTime) );
					
					if(!keep)
						++i;
					else			
						i = ei;
				}
				
				if(!keep)
				{
					char bufferUEM3[BUFFER_SIZE];
					sprintf(bufferUEM3, "UEMFilter - Removing segment in '%s/%s' with times [%d, %d] (mid: %d) regarding the UEM file rules", segFile.c_str(), segChannel.c_str(), pSegment->GetStartTime(), pSegment->GetEndTime(), segMidPoint);
					LOG_INFO(m_pLogger, bufferUEM3);
					listSegmentsToRemove.push_back(pSegment);
				}
			}
			
			pListUEMElement->clear();
			delete pListUEMElement;
		}
	}
	else
	{
		LOG_FATAL(m_pLogger, "UEMFilter::ProcessSingleSpeech() - Neither Ref nor Hyp - do nothing!");
		// Not defined so... for the moment do nothing
		exit(E_COND);
	}
	
	// Step 2: removing the unwanted segments
	list<Segment*>::iterator  i = listSegmentsToRemove.begin();
	list<Segment*>::iterator ei = listSegmentsToRemove.end();
	
	while(i != ei)
	{
		speech->RemoveSegment(*i);
		++i;
	}
	
	listSegmentsToRemove.clear();
	
	return nbrerr;
}

unsigned long int UEMFilter::ProcessSpeechSet(SpeechSet* references, map<string, SpeechSet*> &hypothesis)
{
	/* Generation of the ISGs */
	if(references->HasInterSegmentGap())
	{
		LOG_INFO(m_pLogger, "UEMFilter: Inter Segment Gap detected on the input - Abording addition of ISGs");
		return 0;
	}
	
	LOG_INFO(m_pLogger, "UEMFilter:  Adding Inter Segment Gaps to references");
	
	ulint nbrerr = 0;
	CTMSTMRTTMSegmentor* pCTMSTMRTTMSegmentor = new CTMSTMRTTMSegmentor();
	SpeechSet* tmppSpeechSet = new SpeechSet();
	Speech* ISGspeech = new Speech(references);
	
	pCTMSTMRTTMSegmentor->Reset(references, tmppSpeechSet);
	
	map<string, map<string, list<int> > > mapListTime;
	map<string, map<string, int > > mapMinRefTime;
	map<string, map<string, int > > mapMaxRefTime;
	map<string, map<string, int > > mapMinHypTime;
	map<string, map<string, int > > mapMaxHypTime;
	map<string, map<string, list<int> > > mapListSGborder;
	
	long int ElmNum = 0;
	long int LinNum = 0;
	
	while (pCTMSTMRTTMSegmentor->HasNext())
	{
		SegmentsGroup* pSG = pCTMSTMRTTMSegmentor->Next();
		int minSG = -1;
		int maxSG = -1;
		string file = "";
		string channel = "";
		
		size_t numRefs = pSG->GetNumberOfReferences();
		
		for(size_t i=0; i<numRefs; ++i)
		{
			vector<Segment*> vecSeg = pSG->GetReference(i);
			
			for(size_t j=0; j<vecSeg.size(); ++j)
			{
				if( (minSG == -1) || (vecSeg[j]->GetStartTime() < minSG) )
					minSG = vecSeg[j]->GetStartTime();
					
				if( (maxSG == -1) || (vecSeg[j]->GetEndTime() > maxSG) )
					maxSG = vecSeg[j]->GetEndTime();
					
				file = vecSeg[j]->GetSource();
				channel = vecSeg[j]->GetChannel();
				
				if(vecSeg[j]->GetSourceLineNum() > LinNum)
					LinNum = vecSeg[j]->GetSourceLineNum();
					
				if(vecSeg[j]->GetSourceElementNum() > ElmNum)
					ElmNum = vecSeg[j]->GetSourceElementNum();
			}
		}
		
		mapListTime[file][channel].push_back(minSG);
		mapListTime[file][channel].push_back(maxSG);
		mapListSGborder[file][channel].push_back(minSG);
		
		char bufferSG[BUFFER_SIZE];
		sprintf(bufferSG, "UEMFilter::ProcessSpeechSet() - SG %ld time for '%s/%s' with times: %d %d", pSG->GetsID(), file.c_str(), channel.c_str(), minSG, maxSG);
		LOG_DEBUG(m_pLogger, bufferSG);
		
		// Min and max used only when the UEM are defined
		if( mapMinRefTime.find(file) == mapMinRefTime.end())
		{
			// no file defined
			mapMinRefTime[file][channel] = minSG;
			mapMaxRefTime[file][channel] = maxSG;
		}
		else if( mapMinRefTime[file].find(channel) == mapMinRefTime[file].end() )
		{
			// file defined by no chennel defined
			mapMinRefTime[file][channel] = minSG;
			mapMaxRefTime[file][channel] = maxSG;
		}
		else
		{
			// file and channel defined
			if(minSG < mapMinRefTime[file][channel])
				mapMinRefTime[file][channel] = minSG;
				
			if(maxSG > mapMaxRefTime[file][channel])
				mapMaxRefTime[file][channel] = maxSG;
		}
		
		mapMinHypTime[file][channel] = mapMinRefTime[file][channel];
		mapMaxHypTime[file][channel] = mapMaxRefTime[file][channel];
		
		char bufferISG[BUFFER_SIZE];
		sprintf(bufferISG, "UEMFilter::ProcessSpeechSet() - Border SG time for '%s/%s' with times: %d %d", file.c_str(), channel.c_str(), mapMinRefTime[file][channel], mapMaxRefTime[file][channel]);
		LOG_DEBUG(m_pLogger, bufferISG);
		
		if(pSG)
			delete pSG;
	}
		
	if(m_bUseFile)
	// UEM file defined
	{
		for(size_t i=0; i<m_VectUEMElements.size(); ++i)
		{
			string file = m_VectUEMElements[i]->GetFile();
			string channel = m_VectUEMElements[i]->GetChannel();
			
			if( mapListTime.find(file) != mapListTime.end() )
				if( mapListTime[file].find(channel) != mapListTime[file].end() )
				{
					mapListTime[file][channel].push_back(m_VectUEMElements[i]->GetStartTime());
					mapListTime[file][channel].push_back(m_VectUEMElements[i]->GetEndTime());
				}
		}
	}
	else
	// no UEM file, will use the hyps
	{	
		map<string, SpeechSet* >::iterator hi  = hypothesis.begin();
		map<string, SpeechSet* >::iterator hei = hypothesis.end();
		
		while(hi != hei)
		{
			SpeechSet* spkset = hi->second;
			
			for(size_t spseti = 0; spseti < spkset->GetNumberOfSpeech(); ++spseti)
			{
				Speech* speh = spkset->GetSpeech(spseti);
				
				for(size_t spj=0; spj<speh->NbOfSegments(); ++spj)
				{
					string file = speh->GetSegment(spj)->GetSource();
					string channel = speh->GetSegment(spj)->GetChannel();
					
					/* checks */
					if( mapMinHypTime.find(file) == mapMinHypTime.end() )
					{
						LOG_FATAL(m_pLogger, "UEMFilter::ProcessSpeechSet() - mapMinHypTime file '"+file+"' not defined");
						exit(E_MISSINFO);
					}
					else if( mapMinHypTime[file].find(channel) == mapMinHypTime[file].end() )
					{
						LOG_FATAL(m_pLogger, "UEMFilter::ProcessSpeechSet() - mapMinHypTime '"+file+"' channel '"+channel+"' not defined");
						exit(E_MISSINFO);
					}
					
					if( mapMaxHypTime.find(file) == mapMaxHypTime.end() )
					{
						LOG_FATAL(m_pLogger, "UEMFilter::ProcessSpeechSet() - mapMaxHypTime file '"+file+"' not defined");
						exit(E_MISSINFO);
					}
					else if( mapMaxHypTime[file].find(channel) == mapMaxHypTime[file].end() )
					{
						LOG_FATAL(m_pLogger, "UEMFilter::ProcessSpeechSet() - mapMaxHypTime '"+file+"' channel'"+channel+"' not defined");
						exit(E_MISSINFO);
					}
					/* end checks */
					
					vector<Token*> vectok = speh->GetSegment(spj)->ToTopologicalOrderedStruct();
					
					for(size_t veci=0; veci<vectok.size(); ++veci)
					{
						if(vectok[veci]->GetStartTime() < mapMinHypTime[file][channel])
							mapMinHypTime[file][channel] = vectok[veci]->GetStartTime();
							
						if(vectok[veci]->GetEndTime() > mapMaxHypTime[file][channel])
							mapMaxHypTime[file][channel] = vectok[veci]->GetEndTime();
					}
						
					vectok.clear();
				}
			}
			
			++hi;
		}
		
		hi  = hypothesis.begin();
		
		while(hi != hei)
		{
			SpeechSet* spkset = hi->second;
			
			for(size_t spseti = 0; spseti < spkset->GetNumberOfSpeech(); ++spseti)
			{
				Speech* speh = spkset->GetSpeech(spseti);
				
				for(size_t spj=0; spj<speh->NbOfSegments(); ++spj)
				{
					string file = speh->GetSegment(spj)->GetSource();
					string channel = speh->GetSegment(spj)->GetChannel();
					
					mapListTime[file][channel].push_front(mapMinHypTime[file][channel]);
					mapListTime[file][channel].push_back(mapMaxHypTime[file][channel]);
				}
			}
			
			++hi;
		}
	}
	
	// Sorting the times regarding file and channel
	map<string, map<string, list<int> > >::iterator mmi = mapListTime.begin();
	map<string, map<string, list<int> > >::iterator mme = mapListTime.end();
	
	while(mmi != mme)
	{
		string file = mmi->first;
	
		map<string, list<int> >::iterator mi = mmi->second.begin();
		map<string, list<int> >::iterator me = mmi->second.end();
		
		while(mi != me)
		{
			string channel = mi->first;
			mi->second.sort();
			
			list<int>::iterator  l = mi->second.begin();
			list<int>::iterator el = mi->second.end();
			
			while(l != el)
			{
				int begintime = (*l);
				
				++l;
				
				if(l == el)
				{
					LOG_FATAL(m_pLogger, "UEMFilter::ProcessSpeechSet() - Invalid list of time");
					exit(E_INVALID);
				}
				
				if(find(mapListSGborder[file][channel].begin(), mapListSGborder[file][channel].end(), begintime) == mapListSGborder[file][channel].end())
				{
					// the time is not a begining of Segment group, so it's a time from Hyp or UEM
					// ISG can be created
					int endtime = (*l);
					
					if(begintime != endtime)
					{
						Segment* Inter_Segment_Gap = Segment::CreateWithEndTime(begintime, endtime, ISGspeech);
						
						++ElmNum;
						++LinNum;
						Inter_Segment_Gap->SetSource(file);
						Inter_Segment_Gap->SetChannel(channel);
						Inter_Segment_Gap->SetSpeakerId(string("inter_segment_gap"));
						Inter_Segment_Gap->SetSourceLineNum(ElmNum);
						Inter_Segment_Gap->SetSourceElementNum(LinNum);
						
						size_t nbSeg = ISGspeech->NbOfSegments();
						
						ostringstream osstr;
						osstr << "(Inter_Segment_Gap-";
						osstr << setw(3) << nouppercase << setfill('0') << nbSeg << ")";
						Inter_Segment_Gap->SetId(osstr.str());
									
						ISGspeech->AddSegment(Inter_Segment_Gap);
						
						char bufferISG[BUFFER_SIZE];
						sprintf(bufferISG, "UEMFilter::ProcessSpeechSet() - Adding ISG for '%s/%s' with times: %d %d", file.c_str(), channel.c_str(), begintime, endtime);
						LOG_DEBUG(m_pLogger, bufferISG);
					}
				}

				++l;
			}

			mi->second.clear();
			++mi;
		}
	
		mmi->second.clear();
		
		mapMinRefTime[file].clear();
		mapMaxRefTime[file].clear();
		mapMinHypTime[file].clear();
		mapMaxHypTime[file].clear();
		mapListSGborder[file].clear();
		
		++mmi;
	}
	
	mapMinRefTime.clear();
	mapMaxRefTime.clear();
	mapMinHypTime.clear();
	mapMaxHypTime.clear();
	mapListSGborder.clear();
	mapListTime.clear();
	
	references->AddSpeech(ISGspeech);
	
	if(pCTMSTMRTTMSegmentor)
		delete pCTMSTMRTTMSegmentor;
		
	if(tmppSpeechSet)
		delete tmppSpeechSet;
	
	return nbrerr;
}

bool UEMFilter::isProcessAllSpeechSet()
{ 
	return (string("true").compare(Properties::GetProperty("filter.uem.isg")) == 0);
}
