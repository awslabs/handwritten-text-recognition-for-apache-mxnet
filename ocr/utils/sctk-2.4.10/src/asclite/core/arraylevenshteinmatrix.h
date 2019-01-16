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
	
#ifndef ARRAYLEVENSHTEINMATRIX_H
#define ARRAYLEVENSHTEINMATRIX_H

#include "levenshteinmatrix.h"

/**
 * Represent the Levenshtein Distance Matrix
 */
class ArrayLevenshteinMatrix : public LevenshteinMatrix
{
	private:
		int* m_TabCost;
		size_t m_SizeOfArray;
		size_t m_NbrDimensions;
		size_t m_MaxSize;
		size_t* m_MultiplicatorDimension;
		
		static Logger* m_pLogger;
	
        size_t CoordinatesToSize(size_t* coordinates);

	public:
		ArrayLevenshteinMatrix(const size_t& _NbrDimensions, size_t* _TabDimensionDeep);
		~ArrayLevenshteinMatrix();
	
		int GetCostFor(size_t* coordinates) { return m_TabCost[CoordinatesToSize(coordinates)]; }
		void SetCostFor(size_t* coordinates, const int& cost);
		bool IsCostCalculatedFor(size_t* coordinates) { return(GetCostFor(coordinates) != C_UNCALCULATED); }	
		size_t GetNumberOfCalculatedCosts() { return m_SizeOfArray; }
        size_t GetMaxSize() { return m_MaxSize; }
	
		string ToString();
};

#endif
