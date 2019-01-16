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
 * Segmentor for reference trn and hypothesis trn file.
 */
 
#include "trntrn_segmentor.h" // class's header file

Logger* TRNTRNSegmentor::logger = Logger::getLogger();

void TRNTRNSegmentor::Reset(SpeechSet* references, SpeechSet* hypothesis)
{
    if (references != refs)
    {
        refs = references;
        //init the uterance id list
        uteranceList.clear();
    
        for (size_t i=0 ; i < references->GetNumberOfSpeech() ; ++i)
        {
            for (size_t j=0 ; j < references->GetSpeech(i)->NbOfSegments() ; ++j)
            {
                Segment* seg = references->GetSpeech(i)->GetSegment(j);
                uteranceList.insert(seg->GetId());
            }
        }
    }
  
    hyps = hypothesis;
    //init the iterator
    currentUterance = *(uteranceList.begin());
    uteranceList.erase(currentUterance);
}

SegmentsGroup* TRNTRNSegmentor::Next()
{
    SegmentsGroup* segGroup = new SegmentsGroup();
    bool foundIt = false;
  
    for (size_t j=0 ; refs->GetNumberOfSpeech() ; ++j)
    {
        for (size_t i=0 ; i < refs->GetSpeech(j)->NbOfSegments() ; ++i)
        {
            Segment* v_seg = refs->GetSpeech(j)->GetSegment(i);
            //LOG_DEBUG(logger, "current uterance("+currentUterance+") compare too '"+v_seg->GetId()+"'");
            if (v_seg->GetId().compare(currentUterance) == 0)
            {
                //LOG_DEBUG(logger, "got it");
                segGroup->AddReference(v_seg);
                foundIt = true;
                break;
            }
        }
    
        if (foundIt)
        {
            foundIt = false;
            break;
        }
    }
    
    for (size_t j=0 ; hyps->GetNumberOfSpeech(); ++j)
    {
        for (size_t i=0 ; i < hyps->GetSpeech(j)->NbOfSegments() ; ++i)
        {
            Segment* v_seg = hyps->GetSpeech(j)->GetSegment(i);
            
            if (v_seg->GetId().compare(currentUterance) == 0)
            {
                segGroup->AddHypothesis(v_seg);
                foundIt = true;
                break;
            }
        }
    
        if (foundIt)
        {
            foundIt = false;
            break;
        }
    }
  
    if (uteranceList.size() == 0)
    {
        currentUterance = "";
    }
    else
    {
        currentUterance = *(uteranceList.begin());
        uteranceList.erase(currentUterance);
    }
  
    return segGroup;
}
