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

#ifndef SPEECH_H
#define SPEECH_H

#include "stdinc.h"
#include "segment.h"
#include "logger.h"

class SpeechSet;

/**
 * Internal representation of an hypothesis file or a reference file.
 */
class Speech
{
	public:
		// class constructor
		Speech(SpeechSet* parentSet);
		Speech(const vector<Segment *>& segments);
		
		// class destructor
		virtual ~Speech();
		/**
		 * Retrieve the segment indexed by index.
		 */
		Segment* GetSegment(const size_t& index) { return m_segments[index]; }
		/**
		 * Remove the segment from the list and link the previous tokens to the nexts.
		 */
		void RemoveSegment(Segment* currentSegment);
		/**
		 * Add this segment at the end of the segment list.
		 */
		virtual void AddSegment(Segment* segment) { m_segments.push_back(segment); }
		/**
		 * Return the number of Segments inside the Speech
		 */
		size_t NbOfSegments() { return m_segments.size(); }
		/**
		 * Return the next Segment starting at the specified time for
		 * the specified source and channel. If the time is in the middle of
		 * A segment return the segment itself.
		 */
		Segment* NextSegment(const int& time, const string& source, const string& channel);
		/**
		 * Return the segments of this speech by the given time
		 */
		vector<Segment*> GetSegmentsByTime(const int& start, const int& end, const string& source, const string& channel);
		/** Determines if case is taken into account to align Tokens part of this Speech. */
		bool PerformCaseSensitiveAlignment();
		/** Determines if fragments are considered as correct when aligning Tokens part of this Speech. */
		bool AreFragmentsCorrect();
		/** Determines if optionally deletable Tokens need to be accounted for. */
		bool UseOptionallyDeletable();
		/** Retrieves the SpeechSet in which this Speech is located */
		SpeechSet* GetParentSpeechSet();
		/** returns the Speech as a string */
		string ToString();
		
		int GetMinTokensTime();
		int GetMaxTokensTime();
	private:
		Speech();
		/**
		 * Represent all the segments of this speech.
		 * Note : By definition they cannot overlap with each other
		 */
		vector<Segment*> m_segments;		
		/**
		 * The parent Set that contain the speech
		 */
		SpeechSet* parentSet;
		
		static Logger* m_pLogger;
};

#endif // SPEECH_H
