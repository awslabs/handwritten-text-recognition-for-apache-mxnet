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

#include "statistics.h"

Statistics::Statistics(const vector<int> & _vecValues)
{
	for(size_t i=0; i< _vecValues.size(); ++i)
		m_VecValues.push_back((double) _vecValues[i]);
	Compute(); 
	
}

Statistics::Statistics(const vector<double> & _vecValues) : m_VecValues(_vecValues)
{
	Compute();
}

void Statistics::Compute()	
{
	if(!m_VecValues.empty())
	{
		size_t i;
		double n = (double)m_VecValues.size();
		double sumSqr = 0.0;
		
		sort(m_VecValues.begin(), m_VecValues.end());
		m_Mean = m_SD = m_Sum = 0.0;
		m_MaxSize = 0;
		
		for(i=0; i<m_VecValues.size(); ++i)
		{
			m_Sum += m_VecValues[i];
			sumSqr += m_VecValues[i]*m_VecValues[i];
			m_MaxSize = max(m_MaxSize, (int)( ceil(log((double)m_VecValues[i])/log(10.0)) ) );
		}
		
		m_Mean = m_Sum/((double) m_VecValues.size());
		m_SD = sqrt((n * sumSqr - m_Sum*m_Sum) / ( n * (n-1)));
		
		if(m_VecValues.size()%2 == 0)
			m_Median = (m_VecValues[m_VecValues.size()/2]+m_VecValues[m_VecValues.size()/2-1])/2.0;
		else
			m_Median = m_VecValues[m_VecValues.size()/2];
		
		m_MaxSize = max(m_MaxSize, (int)( ceil(log(m_Sum)/log(10.0)) ) +2);
		m_MaxSize = max(m_MaxSize, (int)( ceil(log(m_Mean)/log(10.0)) ) +2);
		m_MaxSize = max(m_MaxSize, (int)( ceil(log(m_SD)/log(10.0)) ) +2);
		m_MaxSize = max(m_MaxSize, (int)( ceil(log(m_Median)/log(10.0)) ) +2);
	}
	else
	{
		m_Sum = m_Mean = m_SD = m_Median = 0.0;
		m_MaxSize = 1;
	}
}

