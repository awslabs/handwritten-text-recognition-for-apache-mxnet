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
 * Implementation of an Aligner with the Levenshtein algorithm
 */
 
#include "levenshtein.h" // class's header file

Logger* Levenshtein::logger = Logger::getLogger();

// class constructor
Levenshtein::Levenshtein()
{
   graph = NULL;
}

// class destructor
Levenshtein::~Levenshtein()
{
	// insert your code here
	if(graph)
		delete graph;
}

/**
 * Initialise the Aligner to work with this set of segments
 */
void Levenshtein::SetSegments(SegmentsGroup* segmentsGroup, SpeakerMatch* pSpeakerMatch, const bool& useCompArray)
{
    if (graph != NULL)
        delete graph;

    string opt_case = Properties::GetProperty("align.optionally");
    bool opt_ref = false;
    bool opt_hyp = false;

    if (string("both").compare(opt_case) == 0 || string("ref").compare(opt_case) == 0)
    {
        opt_ref = true;
    }
    else if (string("both").compare(opt_case) == 0 || string("hyp").compare(opt_case) == 0)
    {
        opt_hyp = true;
    }
    else if (string("none").compare(opt_case) == 0)
    {
        opt_ref = false;
        opt_hyp = false;
    }
    else
    {
        LOG_WARN(logger, "The <align.optionally> property has an unrecognized value: '"+opt_case+"'");
    }

	int typecost = atoi(Properties::GetProperty("align.typecost").c_str());
	
	if(typecost == 2)
	{
		LOG_DEBUG(logger, "Using Time base cost model");
    	//graph = new Graph(segmentsGroup, pSpeakerMatch, typecost, 400, 300, 200, 100, 50, opt_ref, opt_hyp, useCompArray);
    }
    else
    {
    	LOG_DEBUG(logger, "Using Word cost model");
    	//graph = new Graph(segmentsGroup, pSpeakerMatch, typecost, 400, 300, 200, 100, 50, opt_ref, opt_hyp, useCompArray);
    }

    graph = new Graph(segmentsGroup, pSpeakerMatch, typecost, 400, 300, 200, 100, 50, opt_ref, opt_hyp, useCompArray);
}
