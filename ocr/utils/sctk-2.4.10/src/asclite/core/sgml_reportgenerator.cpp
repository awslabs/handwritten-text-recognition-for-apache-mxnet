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

#include "sgml_reportgenerator.h" // class's header file

#include "alignedspeechiterator.h"
#include "alignedsegmentiterator.h"
#include "dateutils.h"
#include "speechset.h"
#include "tokenalignment.h"

Logger* SGMLReportGenerator::m_pLogger = Logger::getLogger();

/** Generate the SGML report */
void SGMLReportGenerator::Generate(Alignment* alignment, int where)
{	
	for(size_t i=0; i<alignment->GetNbOfSystems(); ++i)
	{
		ofstream file;
				
		if(where == 1)
		{
			string filename;
			
			if(Properties::GetProperty("report.outputdir") == string(""))
				filename = alignment->GetSystemFilename(i) + ".sgml";
			else
				filename = Properties::GetProperty("report.outputdir") + "/" + GetFileNameFromPath(alignment->GetSystemFilename(i)) + ".sgml";
			
			file.open(filename.c_str());
		
			if(! file.is_open())
			{
				LOG_ERR(m_pLogger, "Could not open file '" + filename + "' for SGML report, the output will be redirected in the stdout to avoid any lost.");
				where = 1;
			}
			else
			{
				LOG_INFO(m_pLogger, "Generating SGML report file '" + filename + "'.");
			}
		}
		else
		{
			LOG_INFO(m_pLogger, "Generating SGML report in the stdout.");
		}
		
		ostream output(where == 1 ? file.rdbuf() : cout.rdbuf());
		GenerateSystem(alignment, alignment->GetSystem(i), output);
		
		if(where == 1)
			file.close();
	}
}

/** Generate the SGML report by system */
void SGMLReportGenerator::GenerateSystem(Alignment* alignment, const string& systm, ostream& output)
{
	AlignedSpeechIterator* aAlignedSpeechs;
	AlignedSpeech* aAlignedSpeechCurrent;
	AlignedSegmentIterator* aAlignedSegments;
	AlignedSegment* aAlignedSegmentCurrent;
	typedef vector< Speaker* > Speakers;
	Speakers speakers;
	Speaker* speaker = NULL;
	string speakerId = "";
	Speakers::iterator potentialSpeaker;
	SpeakerNamePredicate predicate;
	SpeechSet* parentSpeechSet = NULL;
	
	string ref_fname, hyp_fname;
	ref_fname = "";
	hyp_fname = "";
	bool fileNamesSet = false;

	// Prepare the speakers for output
	aAlignedSpeechs = alignment->AlignedSpeeches();
	while(aAlignedSpeechs->Current(&aAlignedSpeechCurrent))
	{
		aAlignedSegments = aAlignedSpeechCurrent->AlignedSegments();
		
		if(!parentSpeechSet)
			parentSpeechSet = aAlignedSpeechCurrent->GetReferenceSpeech()->GetParentSpeechSet();
		
        if(!fileNamesSet)
			ref_fname = aAlignedSpeechCurrent->GetReferenceSpeech()->GetParentSpeechSet()->GetSourceFileName();
		
		while(aAlignedSegments->Current(&aAlignedSegmentCurrent))
		{
			if(!fileNamesSet)
            {
                if (aAlignedSegmentCurrent->GetTokenAlignmentCount() != 0)
                {
                    Token* token = aAlignedSegmentCurrent->GetTokenAlignmentAt(0)->GetAlignmentFor(systm)->GetToken();
				  
                    if(token != NULL) 
                    {
                        hyp_fname = token->GetParentSegment()->GetParentSpeech()->GetParentSpeechSet()->GetSourceFileName();
                        fileNamesSet = true;
                    }
                }
			}
			
			// First try to get the speaker associated with the current reference segment
			speakerId = aAlignedSegmentCurrent->GetReferenceSegment()->GetSpeakerId();
			predicate.SetWantedName(speakerId);
			potentialSpeaker = find_if(speakers.begin(), speakers.end(), predicate);
			
			// If we haven't created the speaker already, create it
			if(potentialSpeaker == speakers.end())
            {
				speaker = new Speaker(speakerId, aAlignedSegmentCurrent->GetReferenceSegment()->GetSourceElementNum());
				speakers.push_back(speaker);
			} 
            else 
            {
				speaker = *potentialSpeaker;
			}
			
			// Add the current AlignedSegment to the current speaker
			speaker->AddSegment(aAlignedSegmentCurrent);
		}
		
		// Sort by sequence order the AlignedSegments for this speaker
		speaker->SortSegments();
	}
	
	// Re-order the speakers by sequence order
	sort(speakers.begin(), speakers.end(), SpeakerSequenceComparator());
	
	string creation_date, format, frag_corr, opt_del, weight_ali, weight_filename;
	
	creation_date = DateUtils::GetDateString();
	format = "2.4";
	frag_corr = (Properties::GetProperty("align.fragment_are_correct") == "true")
          ? "TRUE" : "FALSE";
	opt_del = (Properties::GetProperty("align.optionally") == "both")
          ? "TRUE" : "FALSE";
	
	m_bCaseSensitive = (Properties::GetProperty("align.case_sensitive") == "true") ? true : false;
	weight_ali = "FALSE";
	weight_filename = "";
	
	output << "<SYSTEM";
	output << " title=" << "\"" << GetFileNameFromPath(systm) << "\"";
	output << " ref_fname=" << "\"" << ref_fname << "\"";
	output << " hyp_fname=" << "\"" << hyp_fname << "\"";
	output << " creation_date=" << "\"" << creation_date << "\"";
	output << " format=" << "\"" << format << "\"";
	output << " frag_corr=" << "\"" << frag_corr << "\"";
	output << " opt_del=" << "\"" << opt_del << "\"";
	output << " weight_ali=" << "\"" << weight_ali << "\"";
	output << " weight_filename=" << "\"" << weight_filename << "\"";
	output << ">" << endl;
	
	GenerateCategoryLabel(parentSpeechSet, output);
	
	m_bRefHasTimes = m_bRefHasConf = m_bHypHasConf = m_bHypHasTimes = false;
	Speakers::iterator current = speakers.begin();
	Speakers::iterator end = speakers.end();
	
	while (current != end)
	{
		PreProcessWordAux(*current, systm);
		++current;
	}
	
	// Output speakers
	current = speakers.begin();
	
	while (current != end)
	{
		GenerateSpeaker(*current, systm, output);
		++current;
	}
	
	output << "</SYSTEM>" << endl;
}

void SGMLReportGenerator::GenerateCategoryLabel(SpeechSet* speechSet, ostream& output)
{
	for(size_t i=0; i<speechSet->GetNumberCategoryLabel(); ++i)
	{
		output << "<" << speechSet->GetCategoryLabelType(i);
		output << " id=\"" << speechSet->GetCategoryLabelID(i) << "\"";
		output << " title=\"" << speechSet->GetCategoryLabelTitle(i) << "\"";
		output << " desc=\"" << speechSet->GetCategoryLabelDesc(i) << "\"";
		output << ">" << endl;
		output << "</" << speechSet->GetCategoryLabelType(i) << ">" << endl;
	}
}

void SGMLReportGenerator::GenerateSpeaker(Speaker* speaker, const string& systm, ostream& output)
{
	vector<AlignedSegment*> segments = speaker->GetSegments();
	vector<AlignedSegment*>::iterator i = segments.begin();
	vector<AlignedSegment*>::iterator ei = segments.end();
	
	string speakerName = speaker->GetName();
	output << "<SPEAKER";
	output << " id=" << "\"" << speakerName << "\"";
	output << ">" << endl;
	
	while(i != ei)
	{
		GeneratePath(*i,  systm, output);
		++i;
	}
	
	output << "</SPEAKER>" << endl;
}

void SGMLReportGenerator::GeneratePath(AlignedSegment* alignedSegment, const string& systm, ostream& output)
{
	//PreProcessPath(alignedSegment, systm, &refHasTimes, &refHasConf, &hypHasConf, &hypHasTimes);

	Segment* refSegment = alignedSegment->GetReferenceSegment();
	output << "<PATH";
	output << " id=\"" << refSegment->GetId() << "\"";
	output << " word_cnt=\"" << alignedSegment->GetTokenAlignmentCount() << "\"";
	
	if(refSegment->GetLabel() != "")
		output << " labels=\"" << refSegment->GetLabel() << "\"";
	
	if (refSegment->GetSource().compare("") != 0)
		output << " file=\"" << refSegment->GetSource() << "\"";
	
	if (refSegment->GetChannel().compare("") != 0)
		output << " channel=\"" << refSegment->GetChannel() << "\"";
	
	output << " sequence=\"" << refSegment->GetSourceElementNum() << "\"";
	
	if (refSegment->IsTimeReal()){
		output.setf(ios::fixed);
		output << setprecision(3);
		output << " R_T1=\"" << ((double)(refSegment->GetStartTime()))/1000.0 << "\"";
		output << " R_T2=\"" << ((double)(refSegment->GetEndTime()))/1000.0 << "\"";
	}
	
	HandleWordAux(output);
	
	if(m_bCaseSensitive)
		output << " case_sense=\"1\"";
	
	output << ">" << endl;
	
	size_t tokenNumber = alignedSegment->GetTokenAlignmentCount();
	
	if (tokenNumber > 0)
	{
		GenerateTokenAlignment(alignedSegment->GetTokenAlignmentAt(0), systm, output);
		
		for(size_t i = 1; i < tokenNumber; i++)
		{
			output << ":";
			GenerateTokenAlignment(alignedSegment->GetTokenAlignmentAt(i), systm, output);
		}
	}
	
	output << endl << "</PATH>" << endl;
}

void SGMLReportGenerator::PreProcessWordAux(Speaker* speaker, const string& systm)
{
	vector<AlignedSegment*> segments = speaker->GetSegments();
	vector<AlignedSegment*>::iterator j = segments.begin();
	vector<AlignedSegment*>::iterator ej = segments.end();
	
	while(j != ej)
	{
		uint tokenNumber = (*j)->GetTokenAlignmentCount();
		Token* ref;
		Token* hyp;
		TokenAlignment* ta;
		
		for(size_t i = 0; i < tokenNumber; i++)
		{
			ta = (*j)->GetTokenAlignmentAt(i);
			hyp = ta->GetTokenFor(systm);
			ref = ta->GetReferenceToken();
			
			if(ref)
			{
				if(!m_bRefHasConf)
					m_bRefHasConf = ref->IsConfidenceSet();
				
				if(!m_bRefHasTimes)
					m_bRefHasTimes = ref->IsTimeReal();
			}
			
			if(hyp)
			{
				if (!m_bHypHasConf)
					m_bHypHasConf = hyp->IsConfidenceSet();
				
				if(!m_bHypHasTimes)
					m_bHypHasTimes = hyp->IsTimeReal();
			}		
		}	
			
		++j;
	}
}

void SGMLReportGenerator::HandleWordAux(ostream& output)
{
	// Ignores ref and hyp weight as Jon instructed...
	if(m_bRefHasTimes || m_bHypHasTimes || m_bRefHasConf || m_bHypHasConf || 
	   (string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0))
	{
		output << " word_aux=\"";
		uint more = 0;
		
		if(m_bRefHasTimes)
		{
			output << "r_t1+t2";
			more++;
		}
		
		if(m_bHypHasTimes)
		{
			if (more++ != 0)
				output << ",";
		
			output << "h_t1+t2";
		}
		
		if(m_bRefHasConf)
		{
			if (more++ != 0)
				output << ",";
			
			output << "r_conf";
		}
		
		if(m_bHypHasConf)
		{
			if (more++ != 0)
				output << ",";
			
			output << "h_conf";
		}
			
		if (string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0)
		{
			if (more++ != 0)
				output << ",";
			
			output << "h_spkr,h_isSpkrSub";
		}

		output << "\"";
	}
}

void SGMLReportGenerator::GenerateTokenAlignment(TokenAlignment* tokenAlign, const string& systm, ostream& output)
{	
	TokenAlignment::AlignmentEvaluation* evaluation = tokenAlign->GetAlignmentFor(systm);
	TokenAlignment::AlignmentResult aAlignmentResult = evaluation->GetResult();
	
	if(aAlignmentResult == TokenAlignment::UNAVAILABLE)
	{
		cerr << "Scoring not set." << endl << "Report cannot be done!" << endl;
		return;
	}
	
	Token* hyp = evaluation->GetToken();
	Token* ref = tokenAlign->GetReferenceToken();

	if(aAlignmentResult == TokenAlignment::SPEAKERSUB)
	{
		output << ((TokenAlignment::AlignmentResult)(TokenAlignment::SUBSTITUTION)).GetShortName() << ",";
	} else {
		output << aAlignmentResult.GetShortName() << ",";
	}
	
	if(ref)
	{
		OutputTextFor(ref, output);
	}
	else if(aAlignmentResult == TokenAlignment::CORRECT)
	{
		output << "\"\"";
	}
	
	output << ",";
	
	if(hyp)
	{
		OutputTextFor(hyp, output);
    }
	else if(aAlignmentResult == TokenAlignment::CORRECT)
	{
		output << "\"\"";
	}
	
	if(m_bRefHasTimes)
	{
		output << ",";
		
		if(ref)
		{
			if(ref->IsTimeReal())
			{
				output.setf(ios::fixed);
				output << setprecision(3);
				output << ((double)(ref->GetStartTime()))/1000.0 << "+" << ((double)(ref->GetEndTime()))/1000.0;
			}
		}
	}
	
	if(m_bHypHasTimes)
	{
		output << ",";
		
		if(hyp)
		{
			if(hyp->IsTimeReal())
			{
				output.setf(ios::fixed);
				output << setprecision(3);
				output << ((double)(hyp->GetStartTime()))/1000.0 << "+" << ((double)(hyp->GetEndTime()))/1000.0;
			}
		}
	}
	
	if(m_bRefHasConf)
	{
		output << ",";
		
		if(ref)
		{
			if(ref->IsConfidenceSet())
			{
				output.setf(ios::fixed);
				output << setprecision(6);
				output << ref->GetConfidence();
			}
		}
	}
	
	if(m_bHypHasConf)
	{
		output << ",";
		
		if(hyp)
		{
			if(hyp->IsConfidenceSet())
			{
				output.setf(ios::fixed);
				output << setprecision(6);
				output << hyp->GetConfidence();
			}
		}
	}

	if (string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0)
	{
		output << ",";
		if(hyp)
			output << hyp->GetParentSegment()->GetSpeakerId();
		output << ",";
		if(hyp)
			if (aAlignmentResult == TokenAlignment::SPEAKERSUB)
				output << "true";
			else
				output << "false";
	}
}

void SGMLReportGenerator::OutputTextFor(Token* token, ostream& output)
{
	int fragStatus = token->GetFragmentStatus();
	bool optional = token->IsOptional();
	
	output << "\"";

	output << (optional ? "(" : ""); 

	if(fragStatus == Token::END_FRAGMENT)
		output << "-";
	
	
	if (m_bCaseSensitive)
		output << token->GetText();
	else
		output << token->GetTextInLowerCase();
	
	if(fragStatus == Token::BEGIN_FRAGMENT)
		output << "-";
	
	output << (optional ? ")" : "") << "\"";
}
