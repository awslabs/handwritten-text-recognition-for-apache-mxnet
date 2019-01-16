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
 * Represent the Levenshtein Distance Matrix with compression
 */
	
#include "compressedlevenshteinmatrix.h"

Logger* CompressedLevenshteinMatrix::m_pLogger = Logger::getLogger(); 

CompressedLevenshteinMatrix::CompressedLevenshteinMatrix(const size_t& _NbrDimensions, size_t* _TabDimensionDeep)
{
	m_MaxMemoryKBProp = static_cast<size_t>(ceil(1024*1024*atof(Properties::GetProperty("recording.maxnbofgb").c_str())));
	m_BlockSizeKB = static_cast<uint>(atoi(Properties::GetProperty("align.memorycompressionblock").c_str()));
	
	if(m_BlockSizeKB > 1048576)
		m_BlockSizeKB = 1048575;
	
	/* LZMA Properties ; see lzma/LzmaLib.h */
	m_lzmaLevel = 4;
	m_lzmaDictionarySize = 1 << 24;
	m_lzmaLc = 3;
	m_lzmaLp = 0;
	m_lzmaPb = 2;
	m_lzmaFb = 32;
	m_lzmaNumberThreads = 2;
	m_lzmaPropertiesSize = LZMA_PROPS_SIZE;
	
	m_NbrDimensions = _NbrDimensions;
	m_TabDimensionDeep = new size_t[m_NbrDimensions];
	m_MultiplicatorDimension = new ullint[m_NbrDimensions];
	m_TabBlockDivider = new size_t[m_NbrDimensions];
	m_TabBlockDimensionDeep = new size_t[m_NbrDimensions];
	
	m_MultiplicatorDimension[0] = 1;
	m_TabDimensionDeep[0] = _TabDimensionDeep[0] - 1;
	m_MaxSize = m_TabDimensionDeep[0];
		
	for(size_t i=1; i<m_NbrDimensions; ++i)
	{
		m_TabDimensionDeep[i] = _TabDimensionDeep[i] - 1;
		m_MultiplicatorDimension[i] = m_MultiplicatorDimension[i-1]*m_TabDimensionDeep[i-1];
		m_MaxSize = m_MaxSize * m_TabDimensionDeep[i];
	}
	
	BlockComputation(0);
	
	if(m_BaseLengthIn < 0.2*m_BlockSizeKB*1024)
		BlockComputation(1);
	
	/* To guarantee that the compressed data will fit in its buffer, allocate 
	   an output buffer of size 2% larger than the uncompressed data, plus extra
	   size for the compression properties. */
	m_BaseLengthOut = m_BaseLengthIn + m_BaseLengthIn / 50 + m_lzmaPropertiesSize;
	
	m_MultiplicatorBlockDimension = new size_t[m_NbrDimensions];
	m_MultiplicatorDivider = new size_t[m_NbrDimensions];
		
	m_MultiplicatorBlockDimension[0] = 1;
	m_MultiplicatorDivider[0] = 1;
		
	for(size_t i=1; i<m_NbrDimensions; ++i)
	{
		m_MultiplicatorBlockDimension[i] = m_MultiplicatorBlockDimension[i-1]*m_TabBlockDimensionDeep[i-1];
		m_MultiplicatorDivider[i] = m_MultiplicatorDivider[i-1]*m_TabBlockDivider[i-1];
	}	
		
	m_TabStartByte = new int * [m_NbrCompressedTabs];
	m_TabStartByteCompressed = new int * [m_NbrCompressedTabs];
	m_TabSizes = new uint[m_NbrCompressedTabs];
	m_TabbIsCompressed = new bool[m_NbrCompressedTabs];
	m_TabHitsTimer = new ulint[m_NbrCompressedTabs];
	m_TabIsCreated = new bool[m_NbrCompressedTabs];
	
	m_CurrentMemorySize = 0;
	m_Decompressions = 0;
	m_Compressions = 0;
	m_NbrCompressedBlocks = 0;
	m_NbrDecompressedBlocks = 0;
	m_Accesses = 0;
	
	for(size_t i=0; i<m_NbrCompressedTabs; ++i)
	{
		m_TabIsCreated[i] = false;
		m_TabSizes[i] = 0;
		m_TabStartByte[i] = NULL;
		m_TabStartByteCompressed[i] = NULL;
		m_CurrentMemorySize += 0;
	}
	
	m_SizeOfArray = 0;
	m_NbrCreatedBlocks = 0;
	
	m_UsableMemoryKB = 0.98*((double) m_MaxMemoryKBProp);
	m_PercentageMemoryTriggerStart = 0.01;
	m_PercentageMemoryTriggerStop = 0.2;
	
	LOG_DEBUG(m_pLogger, "Allocation done!");
	
	char buffer[BUFFER_SIZE];
	sprintf(buffer, "Compressed Levenshtein Matrix: %lu blocks of %.1fKB, Usable: %.0fKB, StartGC: %.0fKB, StopGC: %.0fKB", 
			(ulint) m_NbrCompressedTabs, ((double)(m_BaseLengthIn))/1024.0, m_UsableMemoryKB, m_UsableMemoryKB*(1.0-m_PercentageMemoryTriggerStart), m_UsableMemoryKB*(1.0-m_PercentageMemoryTriggerStop));	   
	LOG_DEBUG(m_pLogger, buffer);
}

CompressedLevenshteinMatrix::~CompressedLevenshteinMatrix()
{
	char buffer[BUFFER_SIZE];
	sprintf(buffer, "Compressed Levenshtein Matrix: TotalNbrCells: %llu, CalculatedCells: %llu, RatioCells: %.1f%%, TheoryBlocks: %lu, CreatedBlocks: %lu, RatioBlocks: %.1f%%, ActualSize: %.1fKB, ExpendedSize: %.1fKB", (ullint) m_MaxSize, (ullint) m_SizeOfArray, 100.0*((double)m_SizeOfArray)/((double)m_MaxSize), (ulint) m_NbrCompressedTabs, (ulint) m_NbrCreatedBlocks, 100.0*((double)m_NbrCreatedBlocks)/((double)m_NbrCompressedTabs), ((double) m_CurrentMemorySize)/1024.0, ((double) m_NbrCreatedBlocks)*((double)(m_BaseLengthIn))/1024.0);	   
	LOG_DEBUG(m_pLogger, buffer);
	
	for(size_t i=0; i<m_NbrCompressedTabs; ++i)
	{
		if(isBlockCreated(i))
		{
			if(m_TabbIsCompressed[i])
			{
				if(m_TabStartByteCompressed[i])
					free(m_TabStartByteCompressed[i]);
			}
			else
			{
				if(m_TabStartByte[i])
					free(m_TabStartByte[i]);
			}
		}
	}
	
	delete [] m_TabStartByte;
	delete [] m_TabStartByteCompressed;
	delete [] m_TabSizes;
	delete [] m_TabbIsCompressed;
	delete [] m_TabHitsTimer;
	delete [] m_TabIsCreated;
	delete [] m_TabBlockDimensionDeep;
	delete [] m_MultiplicatorBlockDimension;
	delete [] m_TabBlockDivider;
	delete [] m_TabDimensionDeep;
	delete [] m_MultiplicatorDivider;
	delete [] m_MultiplicatorDimension;
}

void CompressedLevenshteinMatrix::CreateBlock(const size_t& block_index)
{
	if(! isBlockCreated(block_index))
	{
		uint decomp_lengh = m_BaseLengthIn;
		m_TabStartByte[block_index] = (int*) malloc(m_BaseLengthIn);
		memset(m_TabStartByte[block_index], C_UNCALCULATED, decomp_lengh);
		
		m_TabSizes[block_index] = decomp_lengh;
		m_CurrentMemorySize += decomp_lengh;
		m_TabbIsCompressed[block_index] = false;
		++m_NbrDecompressedBlocks;
		m_TabIsCreated[block_index] = true;
		TouchBlock(block_index);
		++m_NbrCreatedBlocks;
		GarbageCollection();
	}
}

void CompressedLevenshteinMatrix::CompressBlock(const size_t& block_index)
{
	CreateBlock(block_index);
	
	if(!m_TabbIsCompressed[block_index])
	{		
		// Block is not compressed, then compress it;
		size_t decomp_lengh = m_TabSizes[block_index];
		size_t comp_lengh = m_BaseLengthOut;
		m_TabStartByteCompressed[block_index] = (int*) malloc(m_BaseLengthOut);
		
		size_t outPropsize = m_lzmaPropertiesSize;
				
		if( LzmaCompress( (unsigned char*) m_TabStartByteCompressed[block_index] + m_lzmaPropertiesSize, &comp_lengh,
						  (unsigned char*) m_TabStartByte[block_index], decomp_lengh,
						  (unsigned char*) m_TabStartByteCompressed[block_index], &outPropsize,
						  m_lzmaLevel,
						  m_lzmaDictionarySize,
						  m_lzmaLc,
						  m_lzmaLp,
						  m_lzmaPb,
						  m_lzmaFb,
						  m_lzmaNumberThreads ) != SZ_OK)
		{
			LOG_FATAL(m_pLogger, "Compression: 'LzmaCompress()' failed!");
			exit(EXIT_FAILURE);
		}
		
		if( (comp_lengh + m_lzmaPropertiesSize >= decomp_lengh) || (outPropsize > m_lzmaPropertiesSize) )
		{
			//Incompressible data
			LOG_DEBUG(m_pLogger, "Compression: Incompressible block ignoring compression!");
			free(m_TabStartByteCompressed[block_index]);
			m_TabStartByteCompressed[block_index] = NULL;
		}
		else
		{
			free(m_TabStartByte[block_index]);
			m_TabStartByte[block_index] = NULL;
			
			m_TabSizes[block_index] = comp_lengh + m_lzmaPropertiesSize;
			m_TabbIsCompressed[block_index] = true;
			m_CurrentMemorySize += comp_lengh + m_lzmaPropertiesSize - decomp_lengh;
			++m_Compressions;
			++m_NbrCompressedBlocks;
			--m_NbrDecompressedBlocks;
		}
	}
}

bool CompressedLevenshteinMatrix::DecompressBlock(const size_t& block_index)
{
	CreateBlock(block_index);
	
	bool decomp = false;
	
	if(decomp = m_TabbIsCompressed[block_index])
	{
		// Block is compressed, then decompress it;
		size_t comp_lengh = m_TabSizes[block_index] - m_lzmaPropertiesSize;
		size_t decomp_lengh = m_BaseLengthIn;
		m_TabStartByte[block_index] = (int*) malloc(m_BaseLengthIn);
	
		if(LzmaUncompress( (unsigned char*) m_TabStartByte[block_index], &decomp_lengh,
						   (unsigned char*) m_TabStartByteCompressed[block_index] + m_lzmaPropertiesSize, &comp_lengh, 
						   (unsigned char*) m_TabStartByteCompressed[block_index], m_lzmaPropertiesSize) != SZ_OK)
		{
			LOG_FATAL(m_pLogger, "Decompression: 'LzmaUncompress()' failed!");
			exit(EXIT_FAILURE);
		}
		
		free(m_TabStartByteCompressed[block_index]);
		m_TabStartByteCompressed[block_index] = NULL;
		
		m_TabSizes[block_index] = decomp_lengh;
		m_TabbIsCompressed[block_index] = false;
		m_CurrentMemorySize += decomp_lengh - comp_lengh;
		++m_Decompressions;
		--m_NbrCompressedBlocks;
		++m_NbrDecompressedBlocks;
	}
	
	TouchBlock(block_index);
	return decomp;
}

void CompressedLevenshteinMatrix::GarbageCollection()
{
	char buffer[BUFFER_SIZE];
	sprintf(buffer, "Compressed Levenshtein Matrix: Current: %lu KB, Limit: %lu KB, CompressedBlocks: %lu, UncompressedBlocks: %lu", m_CurrentMemorySize/1024, 
	                                                                                                                                 static_cast<size_t>(m_UsableMemoryKB), 
	                                                                                                                                 m_NbrCompressedBlocks, 
	                                                                                                                                 m_NbrDecompressedBlocks);	   
	LOG_DEBUG(m_pLogger, buffer);

	if(isCallGarbageCollector())
	{
		bool found = false;
		ulint count = 0;
		
		do
		{
			if(found = ForcedGarbageCollection())
				++count;
		}
		while(found && !isStopGarbageCollector());

		char buffer[BUFFER_SIZE];
		sprintf(buffer, "Garbage collection: %lu blocks compressed", count);	   
		LOG_DEBUG(m_pLogger, buffer);
	}
}

bool CompressedLevenshteinMatrix::ForcedGarbageCollection()
{
	ulint mintouch = ULONG_MAX;
	size_t min_index = 0;

	// Do the ugly Java GC
	bool found = false;
	
	for(size_t i=0; i<m_NbrCompressedTabs; ++i)
	{
		if(isBlockCreated(i))
		{
			if(!m_TabbIsCompressed[i])
			{
				// not compressed
				if(m_TabHitsTimer[i] < mintouch)
				{				
					mintouch = m_TabHitsTimer[i];
					min_index = i;
					found = true;
				}
			}
		}
	}
	
	if(found)
		CompressBlock(min_index);
	
	return found;
}

string CompressedLevenshteinMatrix::ToString()
{
	return string("");
}

void CompressedLevenshteinMatrix::CoordinatesToBlockOffset(size_t* coordinates, size_t& blockNum, size_t& blockOffset)
{
	blockNum = 0;
	blockOffset = 0;
	
	for(size_t i=0; i<m_NbrDimensions; ++i)
	{
		blockNum += (coordinates[i]/m_TabBlockDimensionDeep[i])*m_MultiplicatorDivider[i];
		blockOffset += (coordinates[i]%m_TabBlockDimensionDeep[i])*m_MultiplicatorBlockDimension[i];
	}
}

int CompressedLevenshteinMatrix::GetCostFor(size_t* coordinates)
{
	size_t coord_x;
	size_t coord_y;
	CoordinatesToBlockOffset(coordinates, coord_x, coord_y);

	bool decomp = DecompressBlock(coord_x);
	int out = m_TabStartByte[coord_x][coord_y];
	
	if(decomp)
		GarbageCollection();
		
	return (out);
}

void CompressedLevenshteinMatrix::SetCostFor(size_t* coordinates, const int& cost)
{
	size_t coord_x;
	size_t coord_y;
	CoordinatesToBlockOffset(coordinates, coord_x, coord_y);
	
	bool decomp = DecompressBlock(coord_x);
	
	if(m_TabStartByte[coord_x][coord_y] == C_UNCALCULATED)
		++m_SizeOfArray;
	
	m_TabStartByte[coord_x][coord_y] = cost;
		
	if(decomp)
		GarbageCollection();
}

void CompressedLevenshteinMatrix::BlockComputation(const size_t& levelopt)
{
	// Declaration Vars
	size_t* Cursor = new size_t[m_NbrDimensions];
	vector <size_t>* PrimeDiv = new vector <size_t>[m_NbrDimensions];
	size_t* tmpDivider = new size_t[m_NbrDimensions];
	size_t* tmpBlockDimensions = new size_t[m_NbrDimensions];
	size_t blocksize = m_BlockSizeKB*256;
	
	// Computation
	
	// Initialization
	for(size_t i=0; i<m_NbrDimensions; ++i)
	{
		if(m_TabDimensionDeep[i] == 1)
			PrimeDiv[i].push_back(1);

		for(size_t j=2; j<=m_TabDimensionDeep[i]; ++j)
			if( (m_TabDimensionDeep[i] % j == 0) || 
				( (levelopt >= 1) && ((m_TabDimensionDeep[i]+1) % 2 == 0) && ((m_TabDimensionDeep[i]+1) % j == 0) ) ||
				( (levelopt >= 2) && ((m_TabDimensionDeep[i]+levelopt) % j == 0) )
			  )
				PrimeDiv[i].push_back(j);
			
		Cursor[i] = 0;
	}
	// End Initialization
	
	// Main research
	bool finished = false;
	size_t closestsize = ULONG_MAX;
	
	do
	{
		if(Cursor[0] == PrimeDiv[0].size())
		{
			finished = true;
		}
		else
		{
			size_t size = 1;
			
			for(size_t i=0; i<m_NbrDimensions; ++i)
			{
				tmpDivider[i] = PrimeDiv[i][Cursor[i]];
				tmpBlockDimensions[i] = m_TabDimensionDeep[i]/tmpDivider[i];
				
				if(m_TabDimensionDeep[i] % tmpDivider[i] != 0)
					++(tmpBlockDimensions[i]);
					
				size *= tmpBlockDimensions[i];
			}
			
			size_t closer = labs(blocksize - size);
			
			if(closer < closestsize)
			{
				closestsize = closer;
				
				for(size_t i=0; i<m_NbrDimensions; ++i)
				{
					m_TabBlockDivider[i] = tmpDivider[i];
					m_TabBlockDimensionDeep[i] = tmpBlockDimensions[i];
				}
			}
			
			// Next
			size_t currdim = m_NbrDimensions - 1;
			++(Cursor[currdim]);
			
			while( (currdim > 0) && (Cursor[currdim] == PrimeDiv[currdim].size()) )
			{
				Cursor[currdim] = 0;
				--currdim;
				++(Cursor[currdim]);
			}
		}
	}
	while(!finished);
	// Main research
	
	m_BlockSizeElts = 1;
	m_NbrCompressedTabs = 1;

	for(size_t i=0; i<m_NbrDimensions; ++i)
	{
		m_BlockSizeElts *= m_TabBlockDimensionDeep[i];
		m_NbrCompressedTabs *= m_TabBlockDivider[i];
	}
	
	if(m_BlockSizeElts*sizeof(int) < 16)
		m_BlockSizeElts = 16/sizeof(int);
	
	m_BaseLengthIn = m_BlockSizeElts * sizeof(int);
	// End Computation
	
	// Destruction Vars
	delete [] Cursor;
	
	for(size_t i=0; i<m_NbrDimensions; ++i)
		PrimeDiv[i].clear();

	delete [] PrimeDiv;
	delete [] tmpBlockDimensions;
	delete [] tmpDivider;
	
	//return isIncreasable;
}
