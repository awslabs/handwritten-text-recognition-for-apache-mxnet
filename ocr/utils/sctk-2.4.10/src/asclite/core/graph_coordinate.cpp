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
 * Gestion of the coordinates in the LCM
 */
	
#include "graph_coordinate.h"

void GraphCoordinateList::AddFront(size_t* coordinate)
{
	size_t* addcoord = new size_t[m_CoordinateDimension];
	
	for(size_t i=0; i<m_CoordinateDimension; ++i)
		addcoord[i] = coordinate[i];
	
	m_ListCoordinates.push_front(addcoord);
}

void GraphCoordinateList::AddBack(size_t* coordinate)
{
	size_t* addcoord = new size_t[m_CoordinateDimension];
	
	for(size_t i=0; i<m_CoordinateDimension; ++i)
		addcoord[i] = coordinate[i];
	
	m_ListCoordinates.push_back(addcoord);
}

size_t* GraphCoordinateList::GetAt(const GraphCoordinateListPosition& GCPosition)
{
	size_t* outcoord = new size_t[m_CoordinateDimension];
	
	for(size_t i=0; i<m_CoordinateDimension; ++i)
		outcoord[i] = (*GCPosition)[i];
	
	return outcoord;
}

void GraphCoordinateList::RemoveAll()
{
	if(!m_ListCoordinates.empty())
	{
		GraphCoordinateListPosition i = GetBeginPosition();
		GraphCoordinateListPosition ei = GetEndPosition();
		
		while(i != ei)
		{
			size_t* elt_ptr = *i;
			
			if(elt_ptr)
				delete [] elt_ptr;
			
			NextPosition(i);
		}
	}
	
	m_ListCoordinates.clear();
}
