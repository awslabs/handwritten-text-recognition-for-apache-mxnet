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
 
#ifndef GRAPHALIGNEDTOKEN_H
#define GRAPHALIGNEDTOKEN_H

#include "stdinc.h"
#include "token.h"

/** Class to aligned tokens returned by the graph */
class GraphAlignedToken
{
	private:
		/** Nombre of dimension */
		size_t m_Dimension;
		/** Array of pointers to tokens */
		Token** m_TabAlignedTokens;
		
	public:
		/** Constructor */
		GraphAlignedToken() {}
		/** Constructor with the number of dimension */
		GraphAlignedToken(const size_t& _dimension);
		/** Destructor */
		~GraphAlignedToken();
	
		/** Set the pointer of a token */
		void SetToken(const size_t& dim, Token* token);
		/** Returns the pointer of a token */
		Token* GetToken(const size_t& dim) { return m_TabAlignedTokens[dim]; }
		/** Returns the number of dimension */
		size_t GetDimension() { return m_Dimension; }
		/** Returns a string representation of this GraphAlignedToken */
		string ToString();
		/**
		 * Redefine the == operator to go through all the object for the comparison
		 */
		bool operator ==(const GraphAlignedToken & gat) const;
		bool operator !=(const GraphAlignedToken & gat) const;
};

#endif
