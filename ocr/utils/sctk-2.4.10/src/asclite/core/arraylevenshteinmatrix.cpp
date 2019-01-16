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
 
/**
 * Represent the Levenshtein Distance Matrix
 */
	
#include "arraylevenshteinmatrix.h"

Logger* ArrayLevenshteinMatrix::m_pLogger = Logger::getLogger(); 

ArrayLevenshteinMatrix::ArrayLevenshteinMatrix(const size_t& _NbrDimensions, size_t* _TabDimensionDeep)
	: m_NbrDimensions(_NbrDimensions)
{
//	m_NbrDimensions = _NbrDimensions;
	m_MultiplicatorDimension = new size_t[m_NbrDimensions];
	
	m_MultiplicatorDimension[0] = 1;
	m_MaxSize = _TabDimensionDeep[0] - 1;
		
	for(size_t i=1; i<m_NbrDimensions; ++i)
	{
		m_MultiplicatorDimension[i] = m_MultiplicatorDimension[i-1]*(_TabDimensionDeep[i-1] - 1);
		m_MaxSize = m_MaxSize * (_TabDimensionDeep[i] - 1);
	}
        
    char buffer[BUFFER_SIZE];
    sprintf(buffer, "ArrayLevenshteinMatrix: %lu cells", static_cast<ulint>(m_MaxSize) );
	LOG_DEBUG(m_pLogger, buffer);
    
    m_TabCost = new int[m_MaxSize];
    
    for(size_t i=0; i<m_MaxSize; ++i)
        m_TabCost[i] = C_UNCALCULATED;
		
	m_SizeOfArray = 0;
	
    LOG_DEBUG(m_pLogger, "Allocation done!");
}

ArrayLevenshteinMatrix::~ArrayLevenshteinMatrix()
{
	char buffer[BUFFER_SIZE];
	sprintf(buffer, "Array Levenshtein Matrix: Total Size: %lu, Calculated: %lu", static_cast<ulint>(m_MaxSize), static_cast<ulint>(m_SizeOfArray) );	   
	LOG_DEBUG(m_pLogger, buffer);
	
    delete [] m_TabCost;
	delete [] m_MultiplicatorDimension;
}

void ArrayLevenshteinMatrix::SetCostFor(size_t* coordinates, const int& cost)
{
	size_t coord = CoordinatesToSize(coordinates);
	
	if(m_TabCost[coord] == C_UNCALCULATED)
		++m_SizeOfArray;
	
	m_TabCost[coord] = cost;
}

string ArrayLevenshteinMatrix::ToString()
{
	std::ostringstream oss;
	
	oss << "----------------" << endl;
	
	for(size_t j=0; j<m_MaxSize; ++j)
		oss << j << " : " << m_TabCost[j] << endl;
	
	oss << "----------------" << endl;
	
	return oss.str();
}

size_t ArrayLevenshteinMatrix::CoordinatesToSize(size_t* coordinates) 
{
	size_t outSize = 0;
	
	for(size_t i=0; i<m_NbrDimensions; ++i)
		outSize += m_MultiplicatorDimension[i]*coordinates[i];
			
	if(outSize >= m_MaxSize)
	{
		char buffer[BUFFER_SIZE];		
		sprintf(buffer, "Try to access data too far %lu (size matrix %lu)", static_cast<ulint>(outSize), static_cast<ulint>(m_MaxSize) );
		LOG_FATAL(m_pLogger, buffer);
		exit(E_INVALID);
	}
	
	return outSize;
}
