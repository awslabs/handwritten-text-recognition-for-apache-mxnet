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

#ifndef LINESTYLE_INPUTPARSER_H
#define LINESTYLE_INPUTPARSER_H

#include "inputparser.h" // inheriting class's header file
#include "token.h"
#include "logger.h"

/**
 * This class is a generic class for all line oriented parser.
 * It give methods to parse line easier.
 */
class LineStyleInputParser : public InputParser
{
	public:
		// class constructor
		LineStyleInputParser() {}
		// class destructor
		virtual ~LineStyleInputParser() {}
		
	protected:
    /**
     * Parse a string as a line of tokens
     * and return the corresponding Segment
     */	
    Segment* ParseWords(const string& source, const string& channel, const string& spkr, const int& start, const int& end, Speech* speech, const string& tokens);
    Segment* ParseWordsEx(const string& source, const string& channel, const string& spkr, const int& start, const int& end, Speech* speech, const string& tokens, const bool& hasconf, const float& confscr, bool bOptionallyDeletable); 
	SpeechSet* ExpandAlternationSpeechSet(SpeechSet *speechs);
	
    private:
        class VirtualSegment
        {
            public:
                VirtualSegment() { SetTraversable(false); }
                ~VirtualSegment();
				vector<Token *> GetStartTokenVector() { return a_startTokens; }
				vector<Token *> GetEndTokenVector() { return a_endTokens; }
                Token* GetStartToken(const size_t& index) { return a_startTokens[index]; }
                size_t GetNbStartToken() { return a_startTokens.size(); }
                Token* GetEndToken(const size_t& index) { return a_endTokens[index]; }
                size_t GetNbEndToken() { return a_endTokens.size(); }
                void AddStartToken(Token* tok) { a_startTokens.push_back(tok); }
                void AddEndToken(Token* tok) { a_endTokens.push_back(tok); }
                void AddEndTokens(LineStyleInputParser::VirtualSegment* toks);
                void ClearEndToken() { a_endTokens.clear(); }
                void SetTraversable(const bool& trav) { traversable = trav; }
                bool IsTraversable() { return traversable; }

            private:
                vector<Token*> a_startTokens;
                vector<Token*> a_endTokens;
                bool traversable;
        };
		
        VirtualSegment* ParseWords(Segment* seg, const string& tokens, bool bOptionallyDeletable);
		vector<string> SeparateBySlash(const string& line);
        vector<string> TokeniseWords(const string& line);
        void Attach(VirtualSegment* tok1, VirtualSegment* tok2);
        VirtualSegment* Transition(VirtualSegment* prec_token, VirtualSegment* toks);
        string FilterSpace(string line);
        string ReplaceChar(const string& line, const string& badstr, const string& goodstr);
        
        bool m_bUseConfidence;
        bool m_bUseExtended;
        float m_Confidence;
        int m_starttime;
        int m_endtime;
};

#endif // LINESTYLE_INPUTPARSER_H
