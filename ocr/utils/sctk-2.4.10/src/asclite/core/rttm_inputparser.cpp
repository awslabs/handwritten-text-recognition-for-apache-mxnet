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
 * Implementation of a RTTM parser for asclite.
 */
 
#include "rttm_inputparser.h" // class's header file

Logger* RTTMInputParser::logger = Logger::getLogger();

/**
 * Load the named file into a Speech element.
 * For a RTTM file no Segment are available so one Segment is created for each token.
 * Also no spkr info are available so it's left empty.
 */
 
SpeechSet* RTTMInputParser::loadFile(const string& name)
{
	if(isOneTokenPerSegment())
	{
		LOG_DEBUG(logger, "Load in Lexeme mode file " + name);
		return ExpandAlternationSpeechSet(loadFileLexeme(name));
	}
	else
	{
		LOG_DEBUG(logger, "Load in Speaker mode file " + name);
		return ExpandAlternationSpeechSet(loadFileSpeaker(name));
	}
}
 
SpeechSet* RTTMInputParser::loadFileLexeme(const string& name)
{
	string line;
	ifstream file;
    long int lineNum = -1;
    long int elemNum = -1;
    
	file.open(name.c_str(), ifstream::in);
    	
	if (! file.is_open())
	{ 
		LOG_FATAL(logger, "Error opening RTTM file " + name);
		exit(E_LOAD); 
	}
	
	SpeechSet* vec = new SpeechSet(name);
    map<string, Speech*> res;
    bool use_confidence = true;
    
    while (getline(file,line,'\n'))
    {
        ++lineNum;
		
        if(line.substr(0, 6) == "LEXEME")
        // if Type is 'LEXEME' we take it, unless we drop the line
        {
            char l_type[BUFFER_SIZE];
            char l_file[BUFFER_SIZE];
			char l_channel[BUFFER_SIZE];
            char l_word[BUFFER_SIZE];
            char l_stype[BUFFER_SIZE];
			char l_spkr[BUFFER_SIZE];
            char l_conf[BUFFER_SIZE];  
			char l_start[BUFFER_SIZE]; 
            char l_duration[BUFFER_SIZE];
            char l_slat[BUFFER_SIZE];
			int nbArgParsed = 0;
            
            nbArgParsed = sscanf(line.c_str(), "%s %s %s %s %s %s %s %s %s %s", (char*) &l_type, 
                                                                                (char*) &l_file, 
                                                                                (char*) &l_channel, 
                                                                                (char*) &l_start, 
                                                                                (char*) &l_duration, 
                                                                                (char*) &l_word, 
                                                                                (char*) &l_stype,
                                                                                (char*) &l_spkr,
                                                                                (char*) &l_conf,
                                                                                (char*) &l_slat );
            
            if( ! ( (nbArgParsed == 9) || (nbArgParsed == 10) ) )
            {
                char buffer[BUFFER_SIZE];
                sprintf(buffer, "Error parsing the line %li in file %s", lineNum, name.c_str());
                LOG_ERR(logger, buffer);
            }
            
            string lower_stype = string(l_stype);
            
            for(size_t i=0;i<lower_stype.length(); ++i)
            {
                lower_stype[i] = tolower(lower_stype[i]);
            }
            
            if( lower_stype == string("lex") ||
			    lower_stype == string("fp") ||
				lower_stype == string("frag") ||
				lower_stype == string("un-lex") ||
				lower_stype == string("for-lex") ||
				lower_stype == string("alpha") ||
				lower_stype == string("acronym") ||
				lower_stype == string("interjection") ||
				lower_stype == string("propernoun") )
            // stype is not 'lex', so drop the line
            {
                Speech* speech = res[l_spkr];
                
                if (!speech)
                {
                    res[l_spkr] = new Speech(vec);
                    speech = res[l_spkr];
                }
                
                int start = ParseString(string(l_start));
                int duration = ParseString(string(l_duration));
                float confscr = 0;
                
                if(string(l_conf).compare("<NA>") == 0)
                    use_confidence = false;
                else
                    confscr = atof(l_conf);
				
				string str_word = string(l_word);
				
				if (str_word.find('"', 0) != string::npos)
				{
					char buffer_err[BUFFER_SIZE];
					sprintf(buffer_err, "Error parsing the line %li in file %s, forbidden character detected", lineNum, name.c_str());
					LOG_FATAL(logger, buffer_err);
					exit(E_LOAD); 
				}
			
				bool _OptionallyDeletable = false;
			
				if( lower_stype == string("fp") ||
					lower_stype == string("frag") ||
					lower_stype == string("un-lex") ||
					lower_stype == string("for-lex") )
				{
					//str_word = string("(" + str_word + ")");
					_OptionallyDeletable = true;
				}
                
                Segment* seg = ParseWordsEx(string(l_file), string(l_channel), string(l_spkr), start, start+duration, speech, str_word, use_confidence, confscr, _OptionallyDeletable);
                
                ++elemNum;
                
                seg->SetSourceLineNum(lineNum);
                seg->SetSourceElementNum(elemNum);
				size_t nbSeg = speech->NbOfSegments();
                
                ostringstream osstr;
                osstr << "(" << l_spkr << "-";
                osstr << setw(3) << nouppercase << setfill('0') << nbSeg << ")";
                seg->SetId(osstr.str());
                
                if (nbSeg != 0)
                {
                    Attach(speech->GetSegment(nbSeg - 1), seg);
                }

                speech->AddSegment(seg);
            }
        }
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
		LOG_FATAL(logger, "RTTM file '" + name + "' contains no data!");
		exit(E_MISSINFO);
	}
    
	return vec;
}

SpeechSet* RTTMInputParser::loadFileSpeaker(const string& name)
{
	string line;
	ifstream file;
    long int lineNum = -1;
    //long int elemNum = -1;
    
	file.open(name.c_str(), ifstream::in);
    
	if (! file.is_open())
	{ 
		LOG_FATAL(logger, "Error opening RTTM file " + name); 
		exit(E_LOAD); 
	}
	
	SpeechSet* vec = new SpeechSet(name);
    map<string, Speech*> res;
	
	map<string, Segment*> mapCurrentSegmentbySpkr;
	map<Segment*, Token*> mapPrevTokenbySegment;
	
	mapCurrentSegmentbySpkr.clear();
	mapPrevTokenbySegment.clear();
    
    while (getline(file,line,'\n'))
    {
        ++lineNum;
				
        if (line.substr(0, 2) == ";;")
		{
			//comment so skip (for now)
		}
		else if(line.substr(0, 7) == "SPEAKER")
		{
			//New Segment
			char l_type[BUFFER_SIZE];
            char l_file[BUFFER_SIZE];
			char l_channel[BUFFER_SIZE];
            char l_word[BUFFER_SIZE];
            char l_stype[BUFFER_SIZE];
			char l_spkr[BUFFER_SIZE];
            char l_conf[BUFFER_SIZE];  
			char l_start[BUFFER_SIZE]; 
            char l_duration[BUFFER_SIZE];
			int nbArgParsed = 0;
            
            nbArgParsed = sscanf(line.c_str(), "%s %s %s %s %s %s %s %s %s", (char*) &l_type, 
                                                                             (char*) &l_file, 
                                                                             (char*) &l_channel, 
                                                                             (char*) &l_start, 
                                                                             (char*) &l_duration, 
                                                                             (char*) &l_word, 
                                                                             (char*) &l_stype,
                                                                             (char*) &l_spkr,
                                                                             (char*) &l_conf );
						
			if(nbArgParsed != 9)
            {
                char buffer[BUFFER_SIZE];
                sprintf(buffer, "Error parsing the line %li in file %s", lineNum, name.c_str());
                LOG_ERR(logger, buffer);
            }
			
			Speech* speech = res[l_spkr];
                
			if (!speech)
			{
				res[l_spkr] = new Speech(vec);
				speech = res[l_spkr];
			}
			
			Segment* currentSeg = Segment::CreateWithDuration(ParseString(string(l_start)), ParseString(string(l_duration)), speech);
			
			currentSeg->SetSource(string(l_file));
			currentSeg->SetChannel(string(l_channel));
			currentSeg->SetSpeakerId(string(l_spkr));
			currentSeg->SetSourceLineNum(lineNum);
			currentSeg->SetSourceElementNum(lineNum);
			
			size_t nbSeg = speech->NbOfSegments();
			
			ostringstream osstr;
			osstr << "(" << l_spkr << "-";
			osstr << setw(3) << nouppercase << setfill('0') << nbSeg << ")";
			currentSeg->SetId(osstr.str());
						
			speech->AddSegment(currentSeg);
			mapCurrentSegmentbySpkr[string(l_spkr)] = currentSeg;
			mapPrevTokenbySegment[currentSeg] = NULL;
		}
        else if(line.substr(0, 6) == "LEXEME")
		// if Type is 'LEXEME' we take it, unless we drop the line
        {
            char l_type[BUFFER_SIZE];
            char l_file[BUFFER_SIZE];
			char l_channel[BUFFER_SIZE];
            char l_word[BUFFER_SIZE];
            char l_stype[BUFFER_SIZE];
			char l_spkr[BUFFER_SIZE];
            char l_conf[BUFFER_SIZE];  
			char l_start[BUFFER_SIZE]; 
            char l_duration[BUFFER_SIZE];
			int nbArgParsed = 0;
            
            nbArgParsed = sscanf(line.c_str(), "%s %s %s %s %s %s %s %s %s", (char*) &l_type, 
                                                                             (char*) &l_file, 
                                                                             (char*) &l_channel, 
                                                                             (char*) &l_start, 
                                                                             (char*) &l_duration, 
                                                                             (char*) &l_word, 
                                                                             (char*) &l_stype,
                                                                             (char*) &l_spkr,
                                                                             (char*) &l_conf );
			
            if(nbArgParsed != 9)
            {
                char buffer[BUFFER_SIZE];
                sprintf(buffer, "Error parsing the line %li in file %s", lineNum, name.c_str());
                LOG_ERR(logger, buffer);
            }
            
            string lower_stype = string(l_stype);
            
            for(size_t i=0;i<lower_stype.length(); ++i)
            {
                lower_stype[i] = tolower(lower_stype[i]);
            }
            
            if( lower_stype == string("lex") ||
			    lower_stype == string("fp") ||
				lower_stype == string("frag") ||
				lower_stype == string("un-lex") ||
				lower_stype == string("for-lex") ||
				lower_stype == string("alpha") ||
				lower_stype == string("acronym") ||
				lower_stype == string("interjection") ||
				lower_stype == string("propernoun") )
            // stype is not 'lex', so drop the line
            {
				if( mapCurrentSegmentbySpkr.find(string(l_spkr)) == mapCurrentSegmentbySpkr.end() )
				{
					//Not found
					char buffer[BUFFER_SIZE];
					sprintf(buffer, "No Segment found for speaker %s at line %li", l_spkr, lineNum);
					LOG_ERR(logger, buffer);
				}
				else
				{
					Segment* seg = mapCurrentSegmentbySpkr[string(l_spkr)];
					Token* tkn = mapPrevTokenbySegment[seg];
									
					string str_word = string(l_word);
				
					if( lower_stype == string("fp") ||
						lower_stype == string("frag") ||
						lower_stype == string("un-lex") ||
						lower_stype == string("for-lex") )
					{
						str_word = string("(" + str_word + ")");
					}
					
					Token* tok = BuildToken(ParseString(string(l_start)), ParseString(string(l_duration)), str_word, seg);

					if(string(l_conf).compare("<NA>") != 0)
						tok->SetConfidence(atof(l_conf));
					
					if(tkn == NULL)
					{
						seg->AddFirstToken(tok);
					}
					else
					{
						tkn->AddNextToken(tok);
						tok->AddPrecToken(tkn);
					}	
					
					mapPrevTokenbySegment[seg] = tok;		
				}
            }
		}
    }
	
	map<Segment*, Token*>::iterator  j = mapPrevTokenbySegment.begin();
	map<Segment*, Token*>::iterator ej = mapPrevTokenbySegment.end();
	
	while(j != ej)
	{
		Segment* segmt = j->first;
		Token* tken = j->second;
		
		if(tken != NULL)
			segmt->AddLastToken(tken);
		
		++j;
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
	
	mapCurrentSegmentbySpkr.clear();
	mapPrevTokenbySegment.clear();
	
    file.close();
    LOG_INFO(logger, "loading of file '" + name + "' done");
    
    map<string, Speech*>::iterator i = res.begin();
	map<string, Speech*>::iterator ei = res.end();
    
	while (i != ei)
	{
		vec->AddSpeech(i->second);
		++i;
	}
    
	return vec;
}


void RTTMInputParser::Attach(Segment* seg1, Segment* seg2)
{
	for (size_t i=0; i < seg1->GetNumberOfLastToken(); ++i)
    {
        for (size_t j=0; j < seg2->GetNumberOfFirstToken(); ++j)
        {
			seg1->GetLastToken(i)->AddNextToken(seg2->GetFirstToken(j));
            seg2->GetFirstToken(j)->AddPrecToken(seg1->GetLastToken(i));
        }    
    }
}
