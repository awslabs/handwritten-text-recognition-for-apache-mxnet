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

#ifndef SEGMENT_H
#define SEGMENT_H

#include "timedobject.h"
#include "token.h"
#include "logger.h"

class Speech;

/**
 * Internal representation of a segment.
 * A segment is a list of Token.
 */
class Segment : public TimedObject
{
	public:		
		// class destructor
		~Segment();
		/**
		 * Register a Token as a begining token of this segment
		 * A Segment can have multiple first token as the token is part of a graph.
		 */
		void AddFirstToken(Token* token);
		/**
		 * Try to retrieve the beginning token indexed number "index"
		 */
		Token* GetFirstToken(const size_t& index) { return f_token->GetNextToken(index); }
		/**
		 * Register a Token as an ending token of this segment
		 * A Segment can have multiple ending token as the token is part of a graph.
		 */
		void AddLastToken(Token* token);
		/**
		 * Try to retrieve the ending token indexed number "index"
		 */
		Token* GetLastToken(const size_t& index) { return l_token->GetNextToken(index); }
		/**
		 * Return the number of last token of this segments
		 */
		size_t GetNumberOfLastToken() { return l_token->GetNbOfNextTokens(); }
		/**
		 * Return the number of begin token of this segments
		 */
		size_t GetNumberOfFirstToken() { return f_token->GetNbOfNextTokens(); }
		/**
		 * Set the Channel name of this segment
		 */
		void SetChannel(const string& x) { channel = x; }
		/**
		 * Return the channel name of this segment
		 */
		string GetChannel() { return channel; }
		/**
		 * Set the Speaker Id of this segment
		 */
		void SetSpeakerId(const string& x) { speakerId = x; }
		/**
		 * Retrieve the Speaker Id of this segment
		 */
		string GetSpeakerId() { return speakerId; }
		/**
		 * Set the source name of this segment
		 */
		void SetSource(const string& x) { source = x; }
		/**
		 * Retrieve the source name of this segment
		 */
		string GetSource() { return source; }
		/**
		 * Change the end time of this segment
		 */
		void SetEndTime(const int& _newEndTime) { endTime = _newEndTime; }
		/**
		 * Set the ID of the segment
		 */
		void SetId(const string& _id) { id = _id; }
		/**
		 * Retrieve ID of the segments
		 */
		string GetId() { return id; }
		/**
         * Set the line number of this element in the source file.
         */
		void SetSourceLineNum(long int _ln) { sourceLineNum = _ln; }
		/**
         * Get the sequence number of this segment
         */
		long int GetSourceLineNum() { return sourceLineNum; }
		/**
         * Set the element sequence number with the source file.
         */
		void SetSourceElementNum(const long int& _ln) { sourceElementNum = _ln; }
		/**
         * Get the elements sequence number within the source file
         */
		long int GetSourceElementNum() { return sourceElementNum; }
		
		void SetLabel(const string& _label) { m_Label = _label; }
		string GetLabel() { return m_Label; }
		
		static Segment* CreateWithDuration(const int& _startTime, const int& _duration, Speech* parent);
		static Segment* CreateWithEndTime(const int& _startTime, const int& _endTime, Speech* parent);
		/**
		 * Merge all Segments in one segments
		 * @deprecated Shoudnt be use anymore. Try to work on the real segments instead
		 */
        static Segment* Merge(const vector<Segment*> & segments);
		/**
		 * Output a planar version of the segment.
		 */
		vector<Token*> ToTopologicalOrderedStruct();
		/** 
         * Return if the token is on the list of the First tokens.
         */
		bool isFirstToken(Token* token);
		/**
		 * Return if the token is on the list of the last tokens.
		 */
		bool isLastToken(Token* token);
		/**
		 * Return true if no token are into this segment
		 */
		bool isEmpty() { return (f_token->GetNbOfNextTokens() == 0 || l_token->GetNbOfNextTokens() == 0); }
		/**
		 * Say if this segment should be exclude from scoring
		 */
		bool isSegmentExcludeFromScoring() { return ignoreSegmentInScoring; }
		/**
         * Change the segment status as "not to score"
         */
        void SetAsSegmentExcludeFromScoring() { ignoreSegmentInScoring = true; }
		
		/** Retrieves the Speech in which this Segment is located. */
		Speech* GetParentSpeech();
		
		/** Returns a string representation of this Segment */
		string ToString();
		
		/** Returns a string representation of this Segment as a single line*/
		string ToStringAsLine();
		
		/** Determines if case is taken into account to align Tokens part of this Speech. */
		bool PerformCaseSensitiveAlignment();
		
		/** Determines if fragments are considered as correct when aligning Tokens part of this Speech. */
		bool AreFragmentsCorrect();
		
		/** Determines if optionally deletable Tokens need to be accounted for. */
		bool UseOptionallyDeletable();
		
		/** Replaces the token with the link list of tokes pointed to by the vectors containing pointers to the start and end tokens **/
		void ReplaceTokenWith(Token *token, const vector<Token*> & startTokens, const vector<Token*> & endTokens);
		
		int GetMinTokensTime();
		int GetMaxTokensTime();
		
		/** Set all the token optionaly deletable by adding () in the text */
		void SetTokensOptionallyDeletable();
		
	protected:	
        /** Checks that start time and duration are valid for this TimedObject. Extension point for subclasses. */
        virtual bool AreStartTimeAndDurationValid(const int& _startTime, const int& _duration) { return AreStartTimeAndEndTimeValid(_startTime, _startTime + _duration); }
		
        /** Checks that start time and end time are valid for this TimedObject. Extension point for subclasses. */
        virtual bool AreStartTimeAndEndTimeValid(const int& _startTime, const int& _endTime);
		
		// class constructor
		Segment();
			
	private:
		/**
		 * Access to the first token of the segment.
		 */
		Token* f_token;
		/**
		 * Access to the last token of the segment.
		 */
		Token* l_token;
		/**
		 * The speaker identificator of this speaker.
		 */
		string speakerId;
		/**
		 * The channel this segment is referenced to.
		 */
		string channel;
		/**
		 * Represent the source of this segment.
		 * This can be a meeting number/id or a the show name...
		 */
		string source;
		/**
		 * The id of the segment (if it has one)
		 */
		string id;
		/**
		 * The line number of the element within the originating source file
		 */
		long int sourceLineNum;
		/**
		 * The element number within the originating source file
		 */
		long int sourceElementNum;
		/**
		 * Define the scoring state of this segment
		 */
		bool ignoreSegmentInScoring;
		/**
		 * Return the parent speech object
		 */
        Speech* speech;

		/**
		 * Labels used for stm files 
		 */
		string m_Label;
		
		int GetLowestTokenStartTime();
		int GetHighestTokenEndTime();
		/**
         * Recurs methods to compute a topological order of a graph
         */
        void ToTopologicalOrderedStruct(Token* start, vector<Token*> *doneNode);
		
		static Logger* logger;
};

#endif // SEGMENT_H
