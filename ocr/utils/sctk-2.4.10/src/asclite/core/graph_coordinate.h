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
 
#ifndef GRAPH_COORDINATE_H
#define GRAPH_COORDINATE_H

#include "stdinc.h"

typedef list<size_t*>::iterator GraphCoordinateListPosition;

class GraphCoordinateList
{
	private:
		list<size_t*> m_ListCoordinates;
		size_t m_CoordinateDimension;
	
	public:
		GraphCoordinateList(const size_t& _CoordinateDimension) { m_CoordinateDimension = _CoordinateDimension; }
		~GraphCoordinateList() { RemoveAll(); }
	
		void AddFront(size_t* coordinate);
		void AddBack(size_t* coordinate);
		GraphCoordinateListPosition GetBeginPosition() { return m_ListCoordinates.begin(); }
		GraphCoordinateListPosition GetEndPosition() { return m_ListCoordinates.end(); }
		size_t* GetAt(const GraphCoordinateListPosition& GCPosition);
		void RemoveAll();
		void NextPosition(GraphCoordinateListPosition& GCPosition) { ++GCPosition; }
		bool isEmpty() { return m_ListCoordinates.empty(); }
		size_t GetSize() {return m_ListCoordinates.size(); }
};

#endif
