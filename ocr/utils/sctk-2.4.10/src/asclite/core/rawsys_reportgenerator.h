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

#ifndef RAWSYS_REPORTGENERATOR_H
#define RAWSYS_REPORTGENERATOR_H

#include "reportgenerator.h" // inheriting class's header file
#include "alignment.h"
#include "statistics.h"
#include "logger.h"

const double log2e = 1.442695041;

inline double F_ROUND(const double& _n, const double& _p)
{
    return( static_cast<double>( static_cast<int>( (_n) * pow(10.0, _p) + ( (_n>0.0)?0.5:-0.5) ) ) / pow(10.0, _p) );
}

class RAWSYS_Datas
{
	private:
		double m_SumConfidenceCorrect;
		double m_SumConfidenceIncorrect;
		uint m_NumberSegments;
		uint m_NumberRefWords;
		uint m_NumberCorrectWords;
		uint m_NumberSubstitutions;
		uint m_NumberSpeakerErrors;
		uint m_NumberDeletions;
		uint m_NumberInsertions;
		uint m_NumberSegmentsErrors;
		
	public:
		RAWSYS_Datas();
		~RAWSYS_Datas() {}
	
		double GetNCE();
		uint GetNumberSegments() { return(m_NumberSegments); }
		uint GetNumberRefWords() { return(m_NumberRefWords); }
		uint GetNumberCorrectWords() { return(m_NumberCorrectWords); }
		uint GetNumberSubstitutions() { return(m_NumberSubstitutions); }
		uint GetNumberSpeakerErrors() { return(m_NumberSpeakerErrors); }
		uint GetNumberDeletions() { return(m_NumberDeletions); }
		uint GetNumberInsertions() { return(m_NumberInsertions); }
		uint GetNumberErrors() { return(m_NumberSubstitutions+m_NumberInsertions+m_NumberDeletions+m_NumberSpeakerErrors); }
		uint GetNumberSegmentsErrors() { return(m_NumberSegmentsErrors); }
	
		double GetPercentCorrectWords() { return( 100.0*static_cast<double>(m_NumberCorrectWords)/static_cast<double>(m_NumberRefWords) ); }
		double GetPercentSubstitutions() { return( 100.0*static_cast<double>(m_NumberSubstitutions)/static_cast<double>(m_NumberRefWords) ); }
		double GetPercentSpeakerErrors() { return( 100.0*static_cast<double>(m_NumberSpeakerErrors)/static_cast<double>(m_NumberRefWords) ); }
		double GetPercentDeletions() { return( 100.0*static_cast<double>(m_NumberDeletions)/static_cast<double>(m_NumberRefWords) ); }
		double GetPercentInsertions() { return( 100.0*static_cast<double>(m_NumberInsertions)/static_cast<double>(m_NumberRefWords) ); }
		double GetPercentErrors() { return( 100.0*static_cast<double>(GetNumberErrors())/static_cast<double>(m_NumberRefWords) ); }
		double GetPercentSegmentsErrors() { return( 100.0*static_cast<double>(m_NumberSegmentsErrors)/static_cast<double>(m_NumberSegments) ); }
	
		double AddCorrectWord(const double& confidence);
		void AddCorrectWord();
		double AddSubstitutionWord(const double& confidence);
		double AddSpeakerErrorWord(const double& confidence);
		void AddDeletionWord();
		double AddInsertionWord(const double& confidence);
		void AddSegment(const bool& withinError);
};

/**
 * Generate a Raw or Sys like report based on
 * how sclite does it.
 */
class RAWSYSReportGenerator : public ReportGenerator
{
	private:
		/** Raw = 1
		  * Sys = 2
		  */
		int m_RawSys;
		/** Structure which store the needed infos sorted by name (usedID) */
		map<string, RAWSYS_Datas*> m_MapDatas;
	
		double m_SumTotConfidenceCorrect;
		double m_SumTotConfidenceIncorrect;
	
		/** Add spaces to the output */
		void AddChar(const int& numspc, const string& str, ostream& outpt);
		/** Adding a string, double, and integer to the display */
		void AddStringText(ostream& outpt, const string& value, const string& justify, const uint& totalspace, const string& addstr = "");
		void AddStringText(ostream& outpt, const double& value, const int& floating, const string& justify, const uint& totalspace, const string& addstr = "");
		void AddStringText(ostream& outpt, const int& value, const string& justify, const uint& totalspace, const string& addstr = "");
	
		void AddSeparator(ostream& outpt, const string& str, const uint& fullsize);
		
		static Logger* logger;
	
	public:
		/** class constructor with the type */
		RAWSYSReportGenerator(const int& _RawSys);
		/** class destructor */
		virtual ~RAWSYSReportGenerator();
		/** Generate the SYSRAW report */
        void Generate(Alignment* alignment, int where);
		/** Generate the SYSRAW report by system */
        void GenerateSystem(Alignment* alignment, const string& systm, ostream& output);
		/** Return the number of different speakers */
		size_t GetNumSpeakers() { return m_MapDatas.size(); }
	
		double GetTotalNCE(const double& numcorrects, const double& numinsertions, const double& numsubstitutions, const double& numspeakererrors);
};

#endif // RAWSYS_REPORTGENERATOR_H
