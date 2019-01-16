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
 
#ifndef UEMFILTER_H
#define UEMFILTER_H

#include "stdinc.h"
#include "checker.h"
#include "ctmstmrttm_segmentor.h"
#include "properties.h"
#include "logger.h"

class UEMElement
{		
    public:
        UEMElement() {}
		UEMElement(const string& _file, const string& _channel, const int& _startTime, const int& _endTime);
        ~UEMElement() {}
		
		string GetFile() { return m_File; }
		string GetChannel() { return m_Channel; }
		int    GetStartTime() { return m_StartTime; }
		int    GetEndTime() { return m_EndTime; }
			
    private:
		string m_File;
		string m_Channel;
		int    m_StartTime;
		int    m_EndTime;
		
        static Logger* m_pLogger;
};

class UEMFilter : public Checker
{
	public:
		// class constructor
		UEMFilter() { m_bUseFile = false; }
		// class destructor
		~UEMFilter();
		
		void    AddUEMElement(UEMElement* _pUEMElement) { m_VectUEMElements.push_back(_pUEMElement); }
		size_t 	GetNumberElement() { return m_VectUEMElements.size(); }
		void 	FindElement(const string& file, const string& channel, list<UEMElement*>* pList);
		
		bool isEmpty() { return m_VectUEMElements.empty(); }
		void LoadFile(const string& filename);
		
		unsigned long int ProcessSingleSpeech(Speech* speech);
		
		bool isProcessAllSpeechSet();
		unsigned long int ProcessSpeechSet(SpeechSet* ref, map<string, SpeechSet*> &hyp);
	
	private:
		vector<UEMElement*> m_VectUEMElements;
		bool m_bUseFile;
		
		int ParseString(const string& chaine) { return static_cast<int>(floor( ( atof(chaine.c_str()) * 1000 ) + 0.5)); }
		
        static Logger* m_pLogger;
};

#endif
