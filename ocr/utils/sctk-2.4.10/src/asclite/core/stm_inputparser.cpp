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
 * Load the named file into a speech element.
 */

#include "stm_inputparser.h" // class's header file

const string STMInputParser::IGNORE_TIME_SEGMENT_IN_SCORING = "ignore_time_segment_in_scoring";
Logger* STMInputParser::logger = Logger::getLogger();

bool STMInputParser::CompareToISGCaseInsensitive(char str[])
{
	char tmp[BUFFER_SIZE];
	
	int  i=0;

	while(str[i])
	{
		tmp[i] = tolower(str[i]);
		i++;
	}
	
	string tmpstr(tmp);
	
	return(tmpstr.compare(string("inter_segment_gap")) == 0);
}

void STMInputParser::LowerCase(char str[])
{
	int  i=0;

	while(str[i])
	{
		str[i] = tolower(str[i]);
		i++;
	}
}

/**
 * Load the named file into a Speech element.
 * @todo Finish this :P
 */
SpeechSet* STMInputParser::loadFile(const string& name)
{
	string line;
	ifstream file;
	long int lineNum = -1;
	long int elemNum = -1;
	map<string, int> nbIgnoreForSpkr;
            
	file.open(name.c_str(), ifstream::in);
    
	if (! file.is_open())
	{ 
		LOG_FATAL(logger, "Error opening STM file " + name); 
		exit (E_LOAD); 
	}
	
	SpeechSet* vec = new SpeechSet(name);
	map<string, Speech*> res;
	
	while (getline(file,line,'\n'))
	{
		++lineNum;
        //cout << lineNum << endl;
		//cout << "prec:"<< prec_seg->isEmpty() << endl;
        
		if (line.find_first_of(";;") == 0)
		{
			string line_catlab = string("");
			string line_id = string("");
			string line_title = string("");
			string line_desc = string("");
			string token;
			size_t index = 0;
		
			char* word = strtok(const_cast<char*>(line.c_str()), " \t\r\n");

			while(word != NULL)
			{
				string str = string(word);
				// ignore the first one which is ;;
				if(index == 1) 
				{
					if(str.compare("CATEGORY") == 0)
						line_catlab = string("CATEGORY");
					
					if(str.compare("LABEL") == 0)
						line_catlab = string("LABEL");
				}
				else if(index > 1)
				{
					size_t qfpos = str.find_first_of("\"");
					size_t qlpos = str.find_last_of("\"");
					
					if(qfpos != qlpos)
					{
						token = str;
					}
					else
					{
						if(!token.empty())
							token.append(string(" "));
							
						token.append(str);
					}

					if(qlpos == str.size()-1)
					{
						while(token.find("\"") != string::npos)
							token.erase(token.find("\""), 1);
					
						// String complete
						if(index == 2)
							line_id = token;
						else if(index == 3)
							line_title = token;
						else if(index == 4)
							line_desc = token;

						token = string("");
					}
				}
			
			
				word = strtok(NULL, " \t\r\n");
				++index;
			}
			
			if(line_id != "")
				vec->AddLabelCategory(line_catlab, line_id, line_title, line_desc);
		}
		else
		{
			char f_file[BUFFER_SIZE];
			char channel[BUFFER_SIZE];
			char spkr[BUFFER_SIZE];
			char start_char[BUFFER_SIZE];
			char end_char[BUFFER_SIZE];
			int start, end;
			char lur[BUFFER_SIZE];                   
			int nbArgParsed = 0;
			string lbl = "";
			
			nbArgParsed = sscanf(line.c_str(), "%s %s %s %s %s %s", (char*) &f_file, (char*) &channel, (char*) &spkr, (char*) &start_char, (char*) &end_char, (char*) &lur);
			
			int lblbpos = line.find("<");
			int lblepos = line.find(">");
			
			if(lblbpos != -1)
				lbl = string(line).substr(lblbpos, lblepos-lblbpos+1);
				
			int wordsBegin = lblepos + 1;
			
			start = ParseString(string(start_char));
			end = ParseString(string(end_char));
			            
			if (nbArgParsed < 6 || lur[0] != '<')
            {
                nbArgParsed = sscanf(line.c_str(), "%s %s %s %s %s", (char*) &f_file, (char*) &channel, (char*) &spkr, (char*) &start_char, (char*) &end_char);
				
				start = ParseString(string(start_char));
				end = ParseString(string(end_char));
				
                wordsBegin = line.find(start_char)+1;
                wordsBegin = line.find(end_char, wordsBegin)+string(end_char).size();
                
                if (nbArgParsed < 5)
                {
                    char buffer[BUFFER_SIZE];
                    sprintf(buffer, "Error parsing the line %li in file %s", lineNum, name.c_str());
                    LOG_ERR(logger, buffer);
                }
            }
            
            // Check if it's inter_segment_gap in case insensitive, and if it is
            // convert it in lower case.
            if(CompareToISGCaseInsensitive(spkr))
            {
				char buffer[BUFFER_SIZE];
				sprintf(buffer, "ISG Detected at line %li '%s'", lineNum, spkr);
				LOG_DEBUG(logger, buffer);
            	LowerCase(spkr);
			}
			
			Speech* speech = res[spkr];
            
			if (!speech)
			{
				res[spkr] = new Speech(vec);
				speech = res[spkr];
			}    
                  
			string s_tokens(line, wordsBegin, line.size());
			string temp_tokens = s_tokens;
			Segment* seg;
			transform(temp_tokens.begin(), temp_tokens.end(), temp_tokens.begin(), (int(*)(int))tolower);
            
			if (temp_tokens.find(IGNORE_TIME_SEGMENT_IN_SCORING) != string::npos)
			{
                seg = Segment::CreateWithEndTime(start, end, speech);
				seg->SetSource(string(f_file));
				seg->SetChannel(string(channel));
				seg->SetSpeakerId(string(spkr));
				seg->SetAsSegmentExcludeFromScoring();
                
                if(nbIgnoreForSpkr.find(spkr) == nbIgnoreForSpkr.end())
                {
                    nbIgnoreForSpkr[spkr]=0;								  
				}
                                
				++(nbIgnoreForSpkr[spkr]);
            }
			else
			{
				seg = ParseWords(string(f_file), string(channel), string(spkr), start, end, speech, s_tokens);
				++elemNum;
			}			
			
			seg->SetSourceLineNum(lineNum);
            seg->SetSourceElementNum(elemNum);
            seg->SetLabel(lbl);
			size_t nbSeg = speech->NbOfSegments();
            
            ostringstream osstr;
			osstr << "(" << spkr << "-";
			osstr << setw(3) << nouppercase << setfill('0') << (nbSeg - nbIgnoreForSpkr[spkr]) << ")";
			seg->SetId(osstr.str());
			
//			if (nbSeg != 0)
//				Attach(speech->GetSegment(nbSeg - 1), seg);
            
			speech->AddSegment(seg);
		}
	}
    	
	map<string, Speech*>::iterator  ssp = res.begin();
	map<string, Speech*>::iterator essp = res.end();
	
	while(ssp != essp)
	{
		string _spkr = ssp->first;
		Speech* _speech = ssp->second;
		size_t nbrseg = _speech->NbOfSegments();
		
		if(nbrseg > 1)
		{			
			Segment* prevSeg = NULL;
			Segment* currSeg = NULL;
			
			for(size_t ii=0; ii<nbrseg; ++ii)
			{
				Segment* segalpha = _speech->GetSegment(ii);
				
				if( (segalpha->GetNumberOfFirstToken() != 0) && (segalpha->GetNumberOfLastToken() != 0) )
					currSeg = segalpha;
				
				if(prevSeg && currSeg)
					Attach(prevSeg, currSeg);
				
				if(currSeg)
				{
					prevSeg = currSeg;
					currSeg = NULL;
				}
			}
		}
		
		++ssp;
	}
	
	file.close();
	LOG_INFO(logger, "loading of file '" + name + "' done");
	
	map<string, Speech*>::iterator i = res.begin();
	map<string, Speech*>::iterator ei = res.end();
    
	bool emptyFile = true;
	
	while (i != ei)
	{
		vec->AddSpeech(i->second);
		emptyFile = false;
		++i;
	}
	
	if(emptyFile)
	{
		LOG_FATAL(logger, "STM file '" + name + "' contains no data!");
		exit(E_MISSINFO);
	}
    
	return vec;
}

void STMInputParser::Attach(Segment* seg1, Segment* seg2)
{
    for (size_t i=0 ; i < seg1->GetNumberOfLastToken() ; ++i)
    {
        for (size_t j=0 ; j < seg2->GetNumberOfFirstToken() ; ++j)
        {
            seg1->GetLastToken(i)->AddNextToken(seg2->GetFirstToken(j));
            seg2->GetFirstToken(j)->AddPrecToken(seg1->GetLastToken(i));
        }    
    }
}
