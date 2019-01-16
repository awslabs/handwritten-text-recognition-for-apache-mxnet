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
 
#ifndef TOKEN_H
#define TOKEN_H

#include "timedobject.h"
#include "properties.h"
#include "stdarg.h"
#include "logger.h"

class Segment;

/**
 * Internal representation of a token.
 * A token represent every informations needed on a word to align it.
 */
class Token : public TimedObject
{
	public:
		static const int NOT_FRAGMENT = 0;
		static const int BEGIN_FRAGMENT = 1;
		static const int END_FRAGMENT = 2;
		
		// class destructor
		~Token();
		
		void SetSourceText(const string& x); // sets the value of text
		/** Retrieves the raw text for this Token. */
		string GetSourceText() { return sourceText; }
		
		string GetText(); // returns the cleaned-up version of the text, used for alignment purposes.
		string GetTextInLowerCase(); // Return the text in lower case FIX-ME: REMOVE?
		void SetConfidence(const float& x); // sets the value of confidence
		float GetConfidence() { return confidence; } // returns the value of confidence
        Token* GetNextToken(const size_t& i) { return next[i]; } //retrieve the next token indexed
		void AddNextToken(Token* token) { next.push_back(token); } //add a "next" token indexed
		Token* GetPrecToken(const size_t& i) { return prec[i]; } //retrieve the next token indexed
		void AddPrecToken(Token* token) { prec.push_back(token); } //add a "prec" token indexed     
		
		static Token* CreateWithDuration(const int& _startTime, const int& _duration, Segment* parent);
		static Token* CreateWithDuration(const int& _startTime, const int& _duration, Segment* parent, const string& _text);
		static Token* CreateWithEndTime(const int& _startTime, const int& _endTime, Segment* parent);
		static Token* CreateWithEndTime(const int& _startTime, const int& _endTime, Segment* parent, const string& _text);
		static Token* CreateWithEndTime(const int& _startTime, const int& _endTime, Segment* parent, const string& _text, Token* first_prec_tokens, ...);
		
		void LinkTokens(Token *nextToken);    // Links the two tokens together
		
		/** Breaks the link between tokens **/
		void UnlinkTokens(Token *nextToken);

		/** Breaks the next token link between tokens **/
		void UnlinkNextToken(Token *nextToken);
		void UnlinkPrevToken(Token *prevToken);
		
		/**
		 * Return the number of next Tokens.
		 */
		size_t GetNbOfNextTokens() { return next.size(); }
		/**
		 * Return the number of prec Tokens.
		 */
		size_t GetNbOfPrecTokens() { return prec.size(); }
		/**
		 * Return true if the two token are equivalent in a 
		 * Speech recognition way.
		 */
		bool IsEquivalentTo(Token* token);
		/**
		 * Return if the Token is Optionnaly Deletable/Insertable.
		 */
		bool IsOptional();
		/**
         * Return true IF the confidence value was set
         */
		bool IsConfidenceSet() { return hasConfidence; }
        /**
         * Returns if the token is a Fragment:
		 * - Token::NOT_FRAGMENT if the token is not a fragment.
		 * - Token::BEGIN_FRAGMENT if the token is a beginning fragment => frag-
		 * - Token::END_FRAGMENT if the token is an ending fragment => -ment
         */
        int GetFragmentStatus();
   
		/**
		 * Return if two tokens are equals
		 */
		bool Equals(Token* token);
		
		int EditDistance(Token* token);
		
		/** Returns a string representation of this Token. */
		string ToString();
		
		/** Retrieves the Segment in which this Token is located */
		Segment* GetParentSegment() { return segment; }
		
		/** Output the string information for csv */
		string GetCSVInformation();
		
		void BecomeOptionallyDeletable() { SetSourceText(Token::BEGIN_OPTIONAL_MARKER + GetSourceText() + Token::END_OPTIONAL_MARKER); }
		
	protected:
		// class constructor
		Token();
		
	private:
		/**
		 * Raw text associated with this Token. We distinguish between source text as
		 * found in input source files, which might contain metadata, and cleaned-up
		 * text, which is the actual text of the Token (excluding metadata).
		 */
		string sourceText;
		/** Index position in sourceText of the cleaned-up text. */
		short start;
		/** Size (in characters) of the cleaned-up text. */
		short size;
		
		/** Updates cleaned-up text if needed. */
		void UpdateCleanedUpTextIfNeeded(const bool& force);
	
		/**
		 * The confidence score of the token.
		 * The confidence score is a number between 0 and 1 
		 * which represent the guessed accuracy of the token text.
		 */
		float confidence;
		/**
		 * True if the confidence was set
		 */
		bool hasConfidence;
        /**
		 * Store if the Token is Optionnaly Deletable/Insertable.
		 */
		bool optional;
		/**
		 * Store if the Token is fragment.
		 * - Token::NOT_FRAGMENT if the token is not a fragment
		 * - Token::BEGIN_FRAGMENT if the token is a beginning fragment => frie-
		 * - Token::END_FRAGMENT if the token is an ending fragment => -ing
		 */
		int fragment;
		/**
		 * Precedent Tokens on the graph
		 */
        vector<Token*> prec;
        /**
		 * Next Tokens on the graph
		 */
        vector<Token*> next;
        /**
         * Parent Segment.
         */
        Segment* segment;
        /**
         * log
         */
        static Logger* logger;
		
		static const char FRAGMENT_MARKER;
		static const char BEGIN_OPTIONAL_MARKER;
		static const char END_OPTIONAL_MARKER;
};

#endif // TOKEN_H
