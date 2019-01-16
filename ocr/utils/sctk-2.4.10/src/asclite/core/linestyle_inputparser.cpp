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
 * This class is a generic class for all line oriented parser.
 * It give methods to parse line easier.
 */

#include "linestyle_inputparser.h" // class's header file

Segment* LineStyleInputParser::ParseWords(const string& source, const string& channel, const string& spkr, const int& start, const int& end, Speech* speech, const string& tokens)
{
    Segment* seg = Segment::CreateWithEndTime(start, end, speech);

    seg->SetSource(source);
    seg->SetChannel(channel);
    seg->SetSpeakerId(spkr);
    m_bUseConfidence = false;
    m_bUseExtended = false;

    string tokens_filtered = FilterSpace(tokens);
	
    LineStyleInputParser::VirtualSegment* words = ParseWords(seg, tokens_filtered, false);

    for (size_t i=0 ; i < words->GetNbStartToken() ; ++i)
        seg->AddFirstToken(words->GetStartToken(i));
    
    for (size_t i=0 ; i < words->GetNbEndToken() ; ++i)
        seg->AddLastToken(words->GetEndToken(i));
	
	delete words;
	
    return seg;
}

Segment* LineStyleInputParser::ParseWordsEx(const string& source, const string& channel, const string& spkr, const int& start, const int& end, Speech* speech, const string& tokens, const bool& hasconf, const float& confscr, bool bOptionallyDeletable)
{
	Segment* seg = Segment::CreateWithEndTime(start, end, speech);
	
	seg->SetSource(source);
	seg->SetChannel(channel);
	seg->SetSpeakerId(spkr);
	m_bUseConfidence = hasconf;
	m_Confidence = confscr;
	m_bUseExtended = true;
	m_starttime = start;
	m_endtime = end;
	
	string tokens_prefiltered0 = FilterSpace(tokens);
	string tokens_prefiltered1 = ReplaceChar(tokens_prefiltered0, string("_"), string(" "));
	string tokens_prefiltered2 = ReplaceChar(tokens_prefiltered1, string("#"), string(" "));	
	
	LineStyleInputParser::VirtualSegment* words = ParseWords(seg, tokens_prefiltered2, bOptionallyDeletable);
	
	for (size_t i=0 ; i < words->GetNbStartToken() ; ++i)
		seg->AddFirstToken(words->GetStartToken(i));
	
	for (size_t i=0 ; i < words->GetNbEndToken() ; ++i)
		seg->AddLastToken(words->GetEndToken(i));
	
	delete words;
	
//	if(bOptionallyDeletable)
//		seg->SetTokensOptionallyDeletable();
	
	return seg;
}

SpeechSet *LineStyleInputParser::ExpandAlternationSpeechSet(SpeechSet *sset)
{
	// Loop through speeches
	for (size_t ss=0; ss < sset->GetNumberOfSpeech(); ++ss)
	{
		// Loop through segments
		for (size_t segI=0; segI<sset->GetSpeech(ss)->NbOfSegments(); ++segI)
		{
			Segment *seg = sset->GetSpeech(ss)->GetSegment(segI);
			vector<Token *> tokens = seg->ToTopologicalOrderedStruct();
			 
			for (size_t t=0; t<tokens.size(); ++t)
			{
				Token *token = tokens[t];
				
				// Check for the syntax 
				size_t beg = token->GetText().find_first_of("{_",0);
				size_t end = token->GetText().find_last_of("_}",string::npos);
				
				if(beg != string::npos && end != string::npos)
				{
					string tokens_prefiltered0 = FilterSpace(token->GetText());
					string tokens_prefiltered1 = ReplaceChar(tokens_prefiltered0, string("_"), string(" "));
					string tokens_prefiltered2 = ReplaceChar(tokens_prefiltered1, string("#"), string(" ")); 
					
					m_bUseExtended = true;
					m_starttime = token->GetStartTime();
					m_endtime = token->GetEndTime();
					
					LineStyleInputParser::VirtualSegment* words = ParseWords(seg, tokens_prefiltered2, token->IsOptional());
					
					seg->ReplaceTokenWith(token, words->GetStartTokenVector(), words->GetEndTokenVector());
					
					if (words)
						delete words;
				}
			}
		}
	}	

	return(sset);
} 

LineStyleInputParser::VirtualSegment* LineStyleInputParser::ParseWords(Segment* seg, const string& tokens, bool bOptionallyDeletable)
{
    LineStyleInputParser::VirtualSegment* out_struct = new LineStyleInputParser::VirtualSegment();
	vector<string> orStatements = SeparateBySlash(tokens);
  
    if (orStatements.size() != 1)
    {
        for (size_t i = 0; i < orStatements.size(); ++i)
        {
			LineStyleInputParser::VirtualSegment* por = ParseWords(seg, orStatements[i], bOptionallyDeletable);
			
            if (por->IsTraversable())
                out_struct->SetTraversable(true);
      
            for (size_t j=0 ; j < por->GetNbStartToken() ; ++j)
                out_struct->AddStartToken(por->GetStartToken(j));

            for (size_t j=0 ; j < por->GetNbEndToken() ; ++j)
                out_struct->AddEndToken(por->GetEndToken(j));
			
			if(por)
				delete por;
        }
	}
    else
    {
        vector<string> t_words = TokeniseWords(orStatements[0]);
        bool isFirstWord = true;
		LineStyleInputParser::VirtualSegment* prec_token = NULL;
		
		for (size_t i=0; i < t_words.size(); ++i)
        {
			bool skipToken = false;
            LineStyleInputParser::VirtualSegment* toks;
			
			if (t_words[i].find(string(" ")) != string::npos)
            {
                toks = ParseWords(seg, t_words[i], bOptionallyDeletable);
				
                if (toks->IsTraversable())
                {
                    //only set the containing structur as traversable if it contain one token
                    if (t_words.size() == 1)
                        out_struct->SetTraversable(true);
                }
            }
            else
            {
                toks = new LineStyleInputParser::VirtualSegment();
        
				if (t_words[i].compare("") != 0 && (t_words[i].compare("@") != 0))
                {
                    if(!m_bUseExtended)
                    {
						Token* tok = BuildToken(-1, -1, t_words[i], seg);
						
						if(bOptionallyDeletable)
							tok->BecomeOptionallyDeletable();
						
                        toks->AddStartToken(tok);
                        toks->AddEndToken(tok);
                    }
                    else
                    {
						int durationwordtime;
                        int startwordtime;
                        
                        if(t_words.size() == 1)
                        {
                            durationwordtime = m_endtime-m_starttime;
                            startwordtime = m_starttime;
                        }
                        else
                        {
                            int wordnumber = i;
                            int nbrword = t_words.size();
                            durationwordtime = (m_endtime-m_starttime)/nbrword;							
                            startwordtime = m_starttime + wordnumber*durationwordtime;
                        }
                        
                        Token* tok = BuildToken(startwordtime, durationwordtime, t_words[i], seg);
                        
                        if(bOptionallyDeletable)
							tok->BecomeOptionallyDeletable();
                         
                        if(m_bUseConfidence)
                            tok->SetConfidence(m_Confidence);

                        toks->AddStartToken(tok);
                        toks->AddEndToken(tok);
                    }
                }
                else
                {
                    //if the empty token is alone then the virtual segment is traversable
                    //else just skip it
                    if (t_words.size() == 1)
                        out_struct->SetTraversable(true);
                    else
                        skipToken = true;
                }
            }
      
            if (skipToken == false)
            {
                //Attach with precedent to prepare next
                if (prec_token != NULL)
                    Attach(prec_token, toks);
      
                if (isFirstWord) // this is a first token of the segment
                {
                    //cout << t_words[i] << " >>first" << endl;
                    for (size_t j=0 ; j < toks->GetNbStartToken() ; ++j)
                        out_struct->AddStartToken(toks->GetStartToken(j));
                }
                /*
                else
                {
                    cout << t_words[i] << " >>not first" << endl;
                }
                */
                //Define if the next word might be a first token also
                if (!toks->IsTraversable() && isFirstWord) 
                    isFirstWord = false;

                //If the current statement is not traversable, clean the last tokens.
                if (!toks->IsTraversable())
                    out_struct->ClearEndToken();

                //Set the the last tokens of the segment
                out_struct->AddEndTokens(toks);
				
				LineStyleInputParser::VirtualSegment* temptoks = Transition(prec_token, toks);
				
				if(temptoks == toks)
				{
					if(prec_token)
						delete prec_token;
					
					prec_token = temptoks;
				}
				else if(temptoks == prec_token)
				{
					/*if(prec_token)
						delete prec_token;*/
					
					prec_token = temptoks;
				}
					
                //prec_token = Transition(prec_token, toks);
            }
			
			if(toks && (toks != out_struct) && (toks != prec_token))
				delete toks;
        }
		
		if(prec_token && (prec_token != out_struct))
            delete prec_token;
		
		t_words.clear();
    }
	
	orStatements.clear();
	
    return out_struct;
}

vector<string> LineStyleInputParser::SeparateBySlash(const string& line)
{
    int currentParenthesisOpen = 0;
    vector<string> res;
    string currentStatement = "";
    char* word = strtok(const_cast<char*>(line.c_str()), " \t\r\n");
    
    while(word != NULL)
    {
        if (string(word).compare("{") == 0)
        {
            ++currentParenthesisOpen;
        }
        else if (string(word).compare("}") == 0)
        {
            --currentParenthesisOpen;
        }
     
        if (string(word).compare("/") == 0 && currentParenthesisOpen == 0)
        {
            res.push_back(currentStatement);
            currentStatement = "";
        }
        else
        {
            currentStatement.append(" ");
            currentStatement.append(word);
        }
    
        word = strtok(NULL, " \t\r\n");
    }
    
    res.push_back(currentStatement);

    return res;
}

vector<string> LineStyleInputParser::TokeniseWords(const string& line)
{
    int currentParenthesisOpen = 0;
    vector<string> res;
    string currentStatement = "";
    //if it's an empty one
    
    if (line.find_first_not_of(" ") == string::npos)
    {
        res.push_back("");
        return res;
    }
    
    char* word = strtok(const_cast<char*>(line.c_str()), " \t\r\n");
    
    while(word != NULL)
    {
        if (string(word).compare("{") == 0)
        {
            ++currentParenthesisOpen;
        }
        else if (string(word).compare("}") == 0)
        {
            --currentParenthesisOpen;
        }
    
        if (string(word).compare("{") == 0 && currentParenthesisOpen == 1)
        {
            //nothing
        }
        else if (string(word).compare("}") == 0 && currentParenthesisOpen == 0)
        {
            res.push_back(currentStatement);
            currentStatement = "";
        }
        else
        {
            if (currentStatement != "")
                currentStatement.append(" ");
      
            currentStatement.append(word);
      
            if (currentParenthesisOpen == 0)
            {
                res.push_back(currentStatement);
                currentStatement = "";
            }
        }
    
        word = strtok(NULL, " \t\r\n");
    }
  
    return res;
}

void LineStyleInputParser::Attach(LineStyleInputParser::VirtualSegment* tok1, LineStyleInputParser::VirtualSegment* tok2)
{
    for (size_t i=0 ; i < tok1->GetNbEndToken() ; ++i)
    {
        for (size_t j=0 ; j < tok2->GetNbStartToken() ; ++j)
        {
            tok1->GetEndToken(i)->AddNextToken(tok2->GetStartToken(j));
            tok2->GetStartToken(j)->AddPrecToken(tok1->GetEndToken(i));
        } 
    }
}

string LineStyleInputParser::FilterSpace(string line)
{
    const char* SPECIALS_TOKENS = "{}/@";

    // 64bits fix - Thanks David Huggins-Daines
    size_t end_line = line.size()-1;
    size_t current_tok_index = line.find_first_of(SPECIALS_TOKENS);

    while (current_tok_index != string::npos)
    {
        //cout << "line before is : " << line << endl;
        //cout << "examining(" << current_tok_index << ") : " << line[current_tok_index] << endl;
        //uint t_end_line = end_line ;
        if (current_tok_index != 0)
        {
            if(line[current_tok_index-1] != ' ')
            {
                line.insert(current_tok_index, " ");
                ++current_tok_index;
                ++end_line;
            }
        }
    
        if (current_tok_index != end_line)
        {
            if(line[current_tok_index+1] != ' ')
            {
                line.insert(current_tok_index+1, " ");
                ++end_line;
            }
        }
        
        //cout << "line after is : " << line << endl;
        current_tok_index = line.find_first_of(SPECIALS_TOKENS, current_tok_index+1);
    }
  
    return line;
}

string LineStyleInputParser::ReplaceChar(const string& line, const string& badstr, const string& goodstr)
{
    size_t badpos = line.find(badstr, 0);
    string outstring = line;    
    
    while(badpos != string::npos)
    {
        size_t oldbadpos = badpos;
        outstring.replace(badpos, badstr.length(), goodstr);
        badpos = outstring.find(badstr, oldbadpos - badstr.length() + goodstr.length());
    }
    
    return outstring;
}

LineStyleInputParser::VirtualSegment* LineStyleInputParser::Transition(LineStyleInputParser::VirtualSegment* prec_token, LineStyleInputParser::VirtualSegment* toks)
{
    //if the current statement is traversable we need to 
    //link to the end tokens of both current and prec
    //otherwise only current
    //cout << "In Transition: " << toks->IsTraversable();
    if (toks->IsTraversable()) 
    {
        if (prec_token != NULL) 
        {
            for(size_t i=0; i<toks->GetNbEndToken(); ++i)
                prec_token->AddEndToken(toks->GetEndToken(i));
            
            return prec_token;
        }
        else
        {
            return toks;
        }
    }
    else
    {
        return toks;
    }
}

void LineStyleInputParser::VirtualSegment::AddEndTokens(LineStyleInputParser::VirtualSegment* toks)
{
    for (size_t j=0; j<toks->GetNbEndToken(); ++j)
        a_endTokens.push_back(toks->GetEndToken(j));
}

LineStyleInputParser::VirtualSegment::~VirtualSegment()
{
	a_startTokens.clear();
	a_endTokens.clear();
}

