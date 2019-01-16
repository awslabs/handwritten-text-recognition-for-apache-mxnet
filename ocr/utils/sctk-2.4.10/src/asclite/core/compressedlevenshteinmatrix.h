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
	
#ifndef COMPRESSEDLEVENSHTEINMATRIX_H
#define COMPRESSEDLEVENSHTEINMATRIX_H

#include "levenshteinmatrix.h"
#include "properties.h"

/**
 * Represent the Levenshtein Distance Matrix w/ compression
 */
class CompressedLevenshteinMatrix : public LevenshteinMatrix
{
	private:
		ullint  m_SizeOfArray;
		size_t  m_NbrDimensions;
		ullint  m_MaxSize;
		ullint* m_MultiplicatorDimension;
		bool*   m_TabbIsCompressed;
		ulint*  m_TabHitsTimer;
		
		int** m_TabStartByte;
		int** m_TabStartByteCompressed;
		uint* m_TabSizes;
		size_t m_NbrCompressedTabs;
		size_t m_BaseLengthIn;
		size_t m_BaseLengthOut;
		
		size_t m_MaxMemoryKBProp;
		uint m_BlockSizeKB;
		size_t m_CurrentMemorySize;
			
		static Logger* m_pLogger;
		
		/* LZMA Compression options */
		int m_lzmaLevel;
		unsigned m_lzmaDictionarySize;
		int m_lzmaLc;
		int m_lzmaLp;
		int m_lzmaPb;
		int m_lzmaFb;
		int m_lzmaNumberThreads;
		size_t m_lzmaPropertiesSize;
	
        void CoordinatesToBlockOffset(size_t* coordinates, size_t& blockNum, size_t& blockOffset);
		
		void CreateBlock(const size_t& block_index);
		
		void CompressBlock(const size_t& block_index);
		bool DecompressBlock(const size_t& block_index);
		
		bool isBlockCreated(const size_t& block_index) { return m_TabIsCreated[block_index]; }
				
		void GarbageCollection();
		bool ForcedGarbageCollection();
		void TouchBlock(const size_t& block_index) { m_TabHitsTimer[block_index] = m_Accesses++; }
		
		ulint m_Decompressions;
		ulint m_Compressions;
		ulint m_NbrCompressedBlocks;
		ulint m_NbrDecompressedBlocks;
		
		bool*   m_TabIsCreated;
		size_t  m_NbrCreatedBlocks;
		
		double m_UsableMemoryKB;
		double m_PercentageMemoryTriggerStart;
		double m_PercentageMemoryTriggerStop;
		
		ulint m_Accesses;
		
		double MemoryUsedKB() { return( (static_cast<double>(m_CurrentMemorySize))/1024.0 ); }
		bool isCallGarbageCollector() { return( (MemoryUsedKB()+(static_cast<double>(m_BaseLengthIn))/1024.0) >= m_UsableMemoryKB*(1.0-m_PercentageMemoryTriggerStart) ); }
		bool isStopGarbageCollector() { return( MemoryUsedKB() <= m_UsableMemoryKB*(1.0-m_PercentageMemoryTriggerStop) ); }
		
		size_t* m_TabBlockDimensionDeep;
		size_t* m_TabBlockDivider;
		size_t* m_TabDimensionDeep;
		size_t* m_MultiplicatorDivider;
		
		void BlockComputation(const size_t& levelopt);
		
		size_t* m_MultiplicatorBlockDimension;
		size_t m_BlockSizeElts;

	public:
		CompressedLevenshteinMatrix(const size_t& _NbrDimensions, size_t* _TabDimensionDeep);
		~CompressedLevenshteinMatrix();
	
		int GetCostFor(size_t* coordinates);
		void SetCostFor(size_t* coordinates, const int& cost);
		bool IsCostCalculatedFor(size_t* coordinates) { return(GetCostFor(coordinates) != C_UNCALCULATED); }
		size_t GetNumberOfCalculatedCosts() { return m_SizeOfArray; }
        size_t GetMaxSize() { return m_MaxSize; }
		
		string ToString();
};

#endif
