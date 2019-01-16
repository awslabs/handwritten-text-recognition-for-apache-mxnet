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

#ifndef STATISTICS_H
#define STATISTICS_H

#include "stdinc.h"

class Statistics
{
	private:
		vector<double> m_VecValues;
		double m_Mean;
		double m_SD;
		double m_Median;
		double m_Sum;
		int m_MaxSize;
		void Compute();
	
	public:
		Statistics(const vector<int> & _vecValues);
		Statistics(const vector<double> & _vecValues);
		~Statistics() { m_VecValues.clear(); }
	
		double GetMean(const bool& safe = false) { return(((m_Mean == 0.0) && safe) ? 1.0 : m_Mean); }
		double GetSD(const bool& safe = false) { return(((m_SD == 0.0) && safe) ? 1.0 : m_SD); }
		double GetMedian(const bool& safe = false) { return(((m_Median == 0.0) && safe) ? 1.0 : m_Median); }
		double GetSum(const bool& safe = false) { return(((m_Sum == 0.0) && safe) ? 1.0 : m_Sum); }
		int GetSize(const bool& safe = false) { return(((m_VecValues.size() == 0) && safe) ? 1 : m_VecValues.size()); }
		int GetMaxSizeString() { return m_MaxSize; }
};

#endif
