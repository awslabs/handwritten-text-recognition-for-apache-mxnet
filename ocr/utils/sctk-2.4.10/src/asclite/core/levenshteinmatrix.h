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
	
#ifndef LEVENSHTEINMATRIX_H
#define LEVENSHTEINMATRIX_H

#include "stdinc.h"
#include "logger.h"

const int C_UNCALCULATED = -1;

/**
 * Generic class for the Levenshtein Cost Matrix (LCM)
 */
class LevenshteinMatrix
{
	public:
		LevenshteinMatrix();
		virtual ~LevenshteinMatrix();
	
		virtual int GetCostFor(size_t* coordinates) = 0;
		virtual void SetCostFor(size_t* coordinates, const int& cost) = 0 ;
		virtual bool IsCostCalculatedFor(size_t* coordinates) = 0;	
		virtual size_t GetNumberOfCalculatedCosts() = 0;
	
		virtual string ToString() = 0;
};

#endif
