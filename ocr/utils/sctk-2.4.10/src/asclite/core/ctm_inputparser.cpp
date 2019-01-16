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
 * Implementation of a CTM parser for asclite.
 */

#include "ctm_inputparser.h" // class's header file

Logger* CTMInputParser::logger = Logger::getLogger();

/*
 * Load the named file into a Speech element.
 * For a CTM file no Segment are available so
 * one Segment is created for each token.
 * Also no spkr info are available so it's 
 * leaved empty.
 * 
 * @todo Finish this :P
 */
SpeechSet* CTMInputParser::loadFile(const string& name)
{
	string line;
	ifstream file;
	file.open(name.c_str(), ifstream::in);
    
	if (! file.is_open())
	{ 
		LOG_FATAL(logger, "Error opening CTM file " + name); 
		exit (E_LOAD); 
	}
	
	SpeechSet* res = new SpeechSet(name);
	Speech* speech = new Speech(res);
	char f_file[BUFFER_SIZE];
	char channel[BUFFER_SIZE];
	char start[BUFFER_SIZE];
	char dur[BUFFER_SIZE];
	char conf[BUFFER_SIZE];
	int start_ms, dur_ms;
	char text[BUFFER_SIZE];
	Token* prec_tok = NULL; // initialization to NULL to avoid warning compilation
	bool isFirstToken = true;
	unsigned long int line_nb = 1;
	bool use_confidence = true;
	
	// Alt temp variables
	Token* prec_alt_tok = NULL; // initialization to NULL to avoid warning compilation
	Segment* alt_seg = NULL; // initialization to NULL to avoid warning compilation
	Segment* prec_alt_seg = NULL; // initialization to NULL to avoid warning compilation
	bool isAltBegin = false;
	bool isInsideAlt = false;
	bool isAlt = false;
	bool isAltEnd = false;
	
	while (getline(file,line,'\n'))
	{
		if(line.find_first_of(";;") != 0)
        // Not a comment
		{
            int nbArgParsed = 0;
			nbArgParsed = sscanf(line.c_str(), "%s %s %s %s %s %s", (char*) &f_file, (char*) &channel, (char*) &start, (char*) &dur, (char*) &text, (char*) &conf);
            
			if (nbArgParsed < 5)
			{
                char temp_start[BUFFER_SIZE];
                char temp_dur[BUFFER_SIZE];
                nbArgParsed = sscanf(line.c_str(), "%s %s %s %s %s %s", (char*) &f_file, (char*) &channel, (char*) &temp_start, (char*) &temp_dur, (char*) &text, (char*) &conf);
                
                if (nbArgParsed < 5)
                {
                    char buffer[BUFFER_SIZE];
                    sprintf(buffer, "Error parsing the line %li in file %s", line_nb, name.c_str());
                    LOG_ERR(logger, buffer);
                }
            }
			
			start_ms = ParseString(string(start));
			dur_ms = ParseString(string(dur));
			
			//start_ms = (int)(atof(start)*1000.0);
			//dur_ms = (int)(atof(dur)*1000.0);
	
            if (nbArgParsed == 5)
            {
                use_confidence = false;
            }
            else if (nbArgParsed == 6)
            {
                use_confidence = true;
            }
      
			//cout << "load line: " << line << endl;
			//cout << "     file: " << f_file << endl;
			//cout << "     chan: " << channel << endl;
			//cout << "     beg : " << start << endl;  
			//cout << "     dur : " << dur << endl;
			//cout << "     text: " << text << endl;
			if (string(text).compare("<ALT_BEGIN>") == 0)
			{
				isAltBegin = true;
				isInsideAlt = true;
			} 
			else if (string(text).compare("<ALT_END>") == 0)
			{
				isAltEnd = true;
				isInsideAlt = false;
			}
			else if (string(text).compare("<ALT>") == 0)
			{
				isAlt = true;
			}
			else
			{
				if (isAltBegin) // After a ALT_BEGIN tag
				{
					Segment* new_seg = Segment::CreateWithDuration(start_ms, dur_ms, speech);
					new_seg->SetSource(string(f_file));
					new_seg->SetChannel(string(channel));
					
					Token* tok = BuildToken(start_ms, dur_ms, string(text), new_seg);
                    
					if (!isFirstToken)
                    {
						if (isAltEnd)
						{
                            alt_seg->AddLastToken(prec_alt_tok);
                            
							for (size_t i=0 ; i < alt_seg->GetNumberOfLastToken() ; ++i)
							{
								alt_seg->GetLastToken(i)->AddNextToken(tok);
                                tok->AddPrecToken(alt_seg->GetLastToken(i));
							}
                            
							isAltEnd = false;
						} 
						else
						{
							prec_tok->AddNextToken(tok);
							tok->AddPrecToken(prec_tok);
						}
					}
                    
					new_seg->AddFirstToken(tok);
					isFirstToken = false;
					prec_alt_tok = tok;
                    
					if (use_confidence)
                        tok->SetConfidence(atof(conf));
                    
					speech->AddSegment(new_seg);
					prec_alt_seg = alt_seg;
					alt_seg = new_seg;
					isAltBegin = false;
				}
				else if (isAlt) // After a ALT tag
				{
					Token* tok = BuildToken(start_ms, dur_ms, string(text), alt_seg);
                    
					if (start_ms+dur_ms > alt_seg->GetEndTime())
					{
						alt_seg->SetEndTime(start_ms+dur_ms);
					}
                    
					alt_seg->AddFirstToken(tok);
					
					if (prec_alt_seg == NULL)
					{
                        if (prec_tok != NULL)
                        {
                            prec_tok->AddNextToken(tok);
                            tok->AddPrecToken(prec_tok);
                        }
                    }
                    else
                    {
                        for (size_t i=0 ; i < prec_alt_seg->GetNumberOfLastToken() ; ++i)
                        {
                            prec_alt_seg->GetLastToken(i)->AddNextToken(tok);
                            tok->AddPrecToken(prec_alt_seg->GetLastToken(i));
                        }  
                    }
                    
                    alt_seg->AddLastToken(prec_alt_tok);
                    prec_alt_tok = tok;
                    
                    if (use_confidence)
                        tok->SetConfidence(atof(conf));
                    
                    isAlt = false;
                }
				else if (isInsideAlt && !isAltBegin && !isAlt) // Inside a ALT tag but not in first pos
				{
					Token* tok = BuildToken(start_ms, dur_ms, string(text), alt_seg);
                    
					if (start_ms+dur_ms > alt_seg->GetEndTime())
					{
						alt_seg->SetEndTime(start_ms+dur_ms);
					}
                    
					prec_alt_tok->AddNextToken(tok);
					tok->AddPrecToken(prec_alt_tok);
					prec_alt_tok = tok;
                    
                    if (use_confidence)
                        tok->SetConfidence(atof(conf));
                }
                else if (isAltEnd) // After a ALT_END tag
                {
                    Segment* seg = Segment::CreateWithDuration(start_ms, dur_ms, speech);
                    seg->SetSource(string(f_file));
                    seg->SetChannel(string(channel));
                    Token* tok = BuildToken(start_ms, dur_ms, string(text), alt_seg);
                    seg->AddFirstToken(tok);
                    seg->AddLastToken(tok);
                    alt_seg->AddLastToken(prec_alt_tok);
                    
                    for (size_t i=0 ; i < alt_seg->GetNumberOfLastToken() ; ++i)
                    {
                        alt_seg->GetLastToken(i)->AddNextToken(tok);
                        tok->AddPrecToken(alt_seg->GetLastToken(i));
                    }
					
					prec_tok = tok;
                    
                    if (use_confidence)
                        tok->SetConfidence(atof(conf));
                        
					speech->AddSegment(seg);
					prec_alt_tok = NULL;
					alt_seg = NULL;
					prec_alt_seg = NULL;
					isAltEnd = false;
					isInsideAlt = false;
				}
				else // Outside an ALT_BEGIN, ALT_END tag.
				{
					Segment* seg = Segment::CreateWithDuration(start_ms, dur_ms, speech);
					seg->SetSource(string(f_file));
					seg->SetChannel(string(channel));
					Token* tok = BuildToken(start_ms, dur_ms, string(text), seg);
					seg->AddFirstToken(tok);
					seg->AddLastToken(tok);
                    
					if (!isFirstToken)
                    {
						prec_tok->AddNextToken(tok);
						tok->AddPrecToken(prec_tok);
					}
                    
					isFirstToken = false;
					prec_tok = tok;
                    
                    if (use_confidence)
					   tok->SetConfidence(atof(conf));
                       
					speech->AddSegment(seg);
				}
			}
		}
        
		++line_nb;
	}
    
	LOG_INFO(logger, "loading of file '" + name + "' done");
	file.close();
	
	if(speech->NbOfSegments() == 0)
	{
		LOG_FATAL(logger, "CTM file '" + name + "' contains no data!");
		exit(E_MISSINFO);
	}
	
	res->AddSpeech(speech);
	    
	return res;
}
