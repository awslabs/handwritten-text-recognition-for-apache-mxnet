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

#ifndef REPORTGENERATOR_H
#define REPORTGENERATOR_H

#include "alignment.h"

/**
 * Abstract class to generate report.
 */
class ReportGenerator
{
	public:
		/** class constructor */
		ReportGenerator() {}
		/** class destructor */
		virtual ~ReportGenerator() {}
		/**
		 * Generate the report based on the alignement.
		 * This report is writen to the disc or ouptut select by where
		 * where = 0 -> cout
		 * where = 1 -> file(s)	
		 */
		virtual void Generate(Alignment* alignment, int where)=0;
	
		/** Find the filename into a path */
		string GetFileNameFromPath(const string& path);
};

#endif // REPORTGENERATOR_H
