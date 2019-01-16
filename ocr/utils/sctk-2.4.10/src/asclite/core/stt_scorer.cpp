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
 * STT implementation of the scoring.
 */

#include "stt_scorer.h"
#include "alignedspeechiterator.h"
#include "alignedsegmentiterator.h"
#include "alignedspeech.h"
#include "alignedsegment.h"
#include "tokenalignment.h"
#include "speakermatch.h"

Logger* STTScorer::logger = Logger::getLogger();

/**
 * Launch the scoring on the Alignement object.
 */
void STTScorer::Score(Alignment* alignment, SpeakerMatch* speakerMatch)
{
	AlignedSpeechIterator* allSpeeches = alignment->AlignedSpeeches();
	AlignedSegmentIterator* allSegments = NULL;
	AlignedSpeech* currentSpeech;
	AlignedSegment* currentSegment;
	TokenAlignment* tokenAlign;
	
	string opt_case = Properties::GetProperty("align.optionally");
    bool opt_ref = false;
    bool opt_hyp = false;
  
    if ("both" == opt_case  || "ref" == opt_case)
    {
        opt_ref = true;
    }
    else if ("both" == opt_case || "hyp" == opt_case)
    {
        opt_hyp = true;
    }
    else if ("none" == opt_case)
    {
        // do nothing
    }
    else
    {
        LOG_WARN(logger, "The <align.optionally> property has an unrecognized value: '"+opt_case+"'");
    }
	
	while (allSpeeches->Current(&currentSpeech))
	{
		allSegments = currentSpeech->AlignedSegments();
		
		while (allSegments->Current(&currentSegment))
		{
			for (size_t k=0 ; k < currentSegment->GetTokenAlignmentCount() ; ++k)
			{
				tokenAlign = currentSegment->GetTokenAlignmentAt(k);
				Token* ref = tokenAlign->GetReferenceToken();
                
				for (size_t i=0 ; i < alignment->GetNbOfSystems() ; ++i)
				{
					string hyp_key = alignment->GetSystem(i);
					TokenAlignment::AlignmentEvaluation* ae = tokenAlign->GetAlignmentFor(hyp_key);
					Token* hyp = ae->GetToken();
					
					if (ref == NULL)
					{
						if (hyp->IsOptional() && opt_hyp)
						{
							ae->SetResult(TokenAlignment::CORRECT);
						}
						else
						{
							ae->SetResult(TokenAlignment::INSERTION);
						}
					}
					else if (hyp == NULL)
					{
						if (ref->IsOptional() && opt_ref)
						{
							ae->SetResult(TokenAlignment::CORRECT);
						}
						else
						{
							ae->SetResult(TokenAlignment::DELETION);
						}            
					}
					else 
					{
						if (ref->IsEquivalentTo(hyp))
						{
							if(string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0)
							{
								if (speakerMatch->GetRef(hyp->GetParentSegment()->GetSource(), hyp->GetParentSegment()->GetChannel(),
														 hyp->GetParentSegment()->GetSpeakerId()).compare(ref->GetParentSegment()->GetSpeakerId()) == 0)
								{
									ae->SetResult(TokenAlignment::CORRECT);
								} else 
								{
									ae->SetResult(TokenAlignment::SPEAKERSUB);
								}
							} else {
								ae->SetResult(TokenAlignment::CORRECT);
							}
						}
						else
						{
							ae->SetResult(TokenAlignment::SUBSTITUTION);
						}
					}
					
					if(logger->isAlignLogON())
					{
						string csvEval = ae->GetResult().GetShortName();
						string csvHyp = ",,,,,,,,,,";
						string csvRef = ",,,,,,,,,,";
						
						char buffer1[BUFFER_SIZE];
						sprintf(buffer1, "%li", currentSegment->GetSegGrpID());
						string seggrpsid = string(buffer1);
						
						string File;
						string Channel;
						
						if(hyp)
						{
							csvHyp = hyp->GetCSVInformation();
							File = hyp->GetParentSegment()->GetSource();
							Channel = hyp->GetParentSegment()->GetChannel();
						}
						
						if(ref)
						{
							csvRef = ref->GetCSVInformation();
							File = ref->GetParentSegment()->GetSource();
							Channel = ref->GetParentSegment()->GetChannel();
						}
						
						LOG_ALIGN(logger, "YES," + seggrpsid + "," + File + "," + Channel + "," + csvEval + "," + csvRef + "," + csvHyp);
					}
				}
			}
		}
		
		if(allSegments)
			delete allSegments;
	}
	
	if(allSpeeches)
		delete allSpeeches;
}
