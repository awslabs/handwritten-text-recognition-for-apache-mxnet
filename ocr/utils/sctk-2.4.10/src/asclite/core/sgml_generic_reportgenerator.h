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

#ifndef SGML_GENERIC_REPORTGENERATOR_H
#define SGML_GENERIC_REPORTGENERATOR_H

#include "segment.h"
#include "segment.h"
#include "logger.h"
#include "dateutils.h"
#include "token.h"
#include "speechset.h"
#include "graphalignedtoken.h"
#include "graphalignedsegment.h"

class SGMLGenericReportGenerator
{
	private:
		vector<GraphAlignedSegment*> m_vGAS;
		vector<string> m_vTitle;
		vector<string> m_vFilename;
		static Logger* m_pLogger;
	public:
		/** class constructor */
		SGMLGenericReportGenerator() {}
		/** class destructor */
		~SGMLGenericReportGenerator();
		/** Generate the SGML report */
        void Generate(int where);

		void AddTitleAndFilename(const string& _filename, const string& _title) { m_vFilename.push_back(_filename); m_vTitle.push_back(_title); }
		void AddGraphAlignSegment(GraphAlignedSegment* gas) { m_vGAS.push_back(gas); }
	
		string GetTitle(const size_t& i) { return m_vTitle[i]; }
		string GetFilename(const size_t& i) { return m_vFilename[i]; }
		size_t GetNbOfSystems() { return m_vTitle.size(); }
};

#endif
