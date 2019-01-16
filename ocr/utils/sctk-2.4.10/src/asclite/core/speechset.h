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

#ifndef SPEECHSET_H
#define SPEECHSET_H

#include "stdinc.h"
#include "speech.h"
#include "logger.h"

typedef struct CATEGORYLABEL
{
	string type, id, title, desc;
} stcCategoryLabel;

/**
 * A speech set represent all the data from one source (reference, participant)
 * it's a collection of Speech(Speaker) from one source.
 */
class SpeechSet
{
	public:
		// class constructors
		SpeechSet(const string& sourceFileName = "DEFAULT_FILE_NAME");
		// class destructor
		~SpeechSet();
		/**
		 * Return the nb of Speech contain in this set
		 */
		size_t GetNumberOfSpeech() { return speeches.size(); }
		/**
		 * Return the speech number i in the set
		 */
		Speech* GetSpeech(const size_t& index) {  return speeches[index]; }
		/**
		 * Add a speech into the set
		 */
		void AddSpeech(Speech* speech) { speeches.push_back(speech); }
		/**
		 * Return true id the set contain only references
		 */
		bool IsRef() { return ref; }
		/**
		 * Return true id the set contain only hypothesis
		 */
		bool IsHyp() { return hyp; }
		bool IsGen() { return gen; }
		/**
		 * Set the hyp/ref status of this set
		 */
		void SetOrigin(const string& status);
		/** Determines if case is taken into account to align Tokens part of this Speech. */
		bool PerformCaseSensitiveAlignment();
		/** Determines if fragments are considered as correct when aligning Tokens part of this Speech. */
		bool AreFragmentsCorrect();
		/** Determines if optionally deletable Tokens need to be accounted for. */
		bool UseOptionallyDeletable();
		
		/** Retrieves the name of the file from which this SpeechSet originated. */
		string GetSourceFileName() { return fileName; }
		
		bool HasInterSegmentGap();
		
		int GetMinTokensTime();
		int GetMaxTokensTime();
		
		void SetTitle(const string& title) { titleName = title; }
		string GetTitle() { return titleName; }
		
		void AddLabelCategory(const string& type, const string& id, const string& title, const string& desc);
		
		size_t GetNumberCategoryLabel() { return m_VectCategoryLabel.size(); }
		string GetCategoryLabelType(const size_t& ind) { return m_VectCategoryLabel[ind].type; }
		string GetCategoryLabelID(const size_t& ind) { return m_VectCategoryLabel[ind].id; }
		string GetCategoryLabelTitle(const size_t& ind) { return m_VectCategoryLabel[ind].title; }
		string GetCategoryLabelDesc(const size_t& ind) { return m_VectCategoryLabel[ind].desc; }
	private:
        /**
         * The internal speech collection
         */
        vector<Speech*> speeches;
        /**
         * Store if the set is a reference set
         */
        bool ref;
        /**
         * Store if the set is a hypothesis set
         */
        bool hyp;
        
        bool gen;
        /**
         * Reference to the logger
         */
        static Logger* logger;
		
		/** The name of the file from which this SpeechSet originated. */
		string fileName;
		string titleName;
		
		/** Category/Label information (only for stm) */
		vector<stcCategoryLabel> m_VectCategoryLabel;
		
		/** Caches the value of the "align.case_sensitive" property. */
		bool case_sensitive;
		/** Caches the value of the "align.fragment_are_correct" property. */
		bool fragments_are_correct;
		/** Caches the values of the "align.optionally" property. */
		bool optionally_deletable;		
		/** Updates the cached properties if needed. */
		void UpdatePropertiesIfNeeded(const bool& force);
};

#endif // SPEECHSET_H
