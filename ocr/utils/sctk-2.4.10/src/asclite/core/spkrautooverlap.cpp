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
 * Check and Validate the overlaping of a speaker with himself
 */
 
#include "spkrautooverlap.h" // class's header file

Logger* SpkrAutoOverlap::logger = Logger::getLogger();

/**
 * Launch the check process on the specified speech.
 */
unsigned long int SpkrAutoOverlap::ProcessSingleSpeech(Speech* speech)
{
    Segment* segPrec = NULL;
    Segment* segCurr = NULL;
    string channelPrec;
    string sourcePrec;
    unsigned long int nbErr = 0;

    for (size_t i=0 ; i < speech->NbOfSegments() ; ++i)
    {
        segCurr = speech->GetSegment(i);
            
        if (segPrec != NULL)
        {
            if ( ( segCurr->GetStartTime() < segPrec->GetEndTime() ) && 
			     (sourcePrec.compare(segCurr->GetSource()) == 0) && 
				 (channelPrec.compare(segCurr->GetChannel()) == 0) )
            {
                ++nbErr;
                char buffer[2*BUFFER_SIZE];
                sprintf(buffer, "Error: The speaker %s for source: %s, channel: %s Overlap himself between: (%d, %d) ==> (%d, %d)", 
                segCurr->GetSpeakerId().c_str(), segCurr->GetSource().c_str(), segCurr->GetChannel().c_str(), segPrec->GetStartTime(), segPrec->GetEndTime(), segCurr->GetStartTime(), segCurr->GetEndTime());
                LOG_ERR(logger, buffer);
            }
        }
        
        segPrec = segCurr;
        channelPrec = segCurr->GetChannel();
        sourcePrec = segCurr->GetSource();
    }
    
    return nbErr;
}
