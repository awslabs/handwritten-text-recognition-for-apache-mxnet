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
 * Generate a Raw or Sys like report based on
 * how sclite does it.
 */
 
#include "rawsys_reportgenerator.h" // class's header file

#include "alignedspeechiterator.h"
#include "alignedsegmentiterator.h"

Logger* RAWSYSReportGenerator::logger = Logger::getLogger();

RAWSYS_Datas::RAWSYS_Datas()
{
	m_SumConfidenceCorrect = m_SumConfidenceIncorrect = 0.0;
	m_NumberSegments = m_NumberRefWords = m_NumberCorrectWords = m_NumberSubstitutions = m_NumberSpeakerErrors = m_NumberDeletions = m_NumberInsertions = m_NumberSegmentsErrors = 0;
}

void RAWSYS_Datas::AddCorrectWord()
{
	++m_NumberRefWords;
	++m_NumberCorrectWords;
}

double RAWSYS_Datas::AddCorrectWord(const double& confidence)
{
    double outSum = log(min(max(0.0000001, confidence), 0.9999999))*log2e;
	++m_NumberRefWords;
	++m_NumberCorrectWords;
	m_SumConfidenceCorrect += outSum;
	return outSum;
}

double RAWSYS_Datas::AddSubstitutionWord(const double& confidence)
{
	double outSum = log(min(max(0.0000001, confidence), 0.9999999))*log2e;
	++m_NumberRefWords;
	++m_NumberSubstitutions;
	m_SumConfidenceIncorrect += outSum;
	return outSum;
}

double RAWSYS_Datas::AddSpeakerErrorWord(const double& confidence)
{
	double outSum = log(min(max(0.0000001, confidence), 0.9999999))*log2e;
	++m_NumberRefWords;
	++m_NumberSpeakerErrors;
	m_SumConfidenceIncorrect += outSum;
	return outSum;
}

void RAWSYS_Datas::AddDeletionWord()
{
	++m_NumberRefWords;
	++m_NumberDeletions;
}

double RAWSYS_Datas::AddInsertionWord(const double& confidence)
{
	double outSum = log(min(max(0.0000001, confidence), 0.9999999))*log2e;
	++m_NumberInsertions;
	m_SumConfidenceIncorrect += outSum;
	return outSum;
}

void RAWSYS_Datas::AddSegment(const bool& withinError)
{
	++m_NumberSegments;
	
	if(withinError)
		++m_NumberSegmentsErrors;
}

double RAWSYS_Datas::GetNCE()
{
	if((m_NumberCorrectWords+m_NumberInsertions+m_NumberSubstitutions+m_NumberSpeakerErrors) == 0)
		return -2000.0;
	
	double pc = min(max(0.0000001, static_cast<double>(m_NumberCorrectWords)/static_cast<double>(m_NumberCorrectWords+m_NumberInsertions+m_NumberSubstitutions+m_NumberSpeakerErrors)), 0.9999999);
	
	double Hmax = -static_cast<double>(m_NumberCorrectWords)*log(pc)*log2e-static_cast<double>(m_NumberInsertions+m_NumberSubstitutions+m_NumberSpeakerErrors)*log(1.0-pc)*log2e;
	
	if(Hmax == 0.0)
		return -2000.0;
	else
		return((Hmax+m_SumConfidenceCorrect+m_SumConfidenceIncorrect)/Hmax);
}

/** class constructor with the type */
RAWSYSReportGenerator::RAWSYSReportGenerator(const int& _RawSys) : m_RawSys(_RawSys)
{
	m_SumTotConfidenceCorrect = m_SumTotConfidenceIncorrect = 0.0;
}

/** class destructor */
RAWSYSReportGenerator::~RAWSYSReportGenerator()
{
	map<string, RAWSYS_Datas*>::iterator i, ei;
	
	i = m_MapDatas.begin();
	ei = m_MapDatas.end();
	
	while(i != ei)
	{
		RAWSYS_Datas* ptr_elt = i->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++i;
	}
	
	m_MapDatas.clear();
}

/** Add spaces to the output */
void RAWSYSReportGenerator::AddChar(const int& numspc, const string& str, ostream& outpt)
{
	for(int i=0; i<numspc; ++i)
		outpt << str;
}

void RAWSYSReportGenerator::AddStringText(ostream& outpt, const string& value, const string& justify, const uint& totalspace, const string& addstr)
{
	string truevalue;
	
	if(addstr == "")
		truevalue = value;
	else
		truevalue = value + addstr;
	
	if (truevalue.length() > totalspace)
	{
        outpt << "...";
        outpt << truevalue.substr(truevalue.size()-totalspace+3);
    }
	else if(justify == "left")
	{
		outpt << truevalue;
		AddChar(totalspace - truevalue.length(), " ", outpt);
	}
	else if(justify == "middle")
	{
		int usedleft = (totalspace - truevalue.length())/2;
		AddChar(usedleft, " ", outpt);
		outpt << truevalue;
		AddChar(totalspace - usedleft - truevalue.length(), " ", outpt);
	}
	else if(justify == "right")
	{
		AddChar(totalspace - truevalue.length(), " ", outpt);
		outpt << truevalue;
	}
}

void RAWSYSReportGenerator::AddStringText(ostream& outpt, const double& value, const int& floating, const string& justify, const uint& totalspace, const string& addstr)
{
	char tempstr[BUFFER_SIZE];
	
	if(floating == 3)
        sprintf(tempstr, "%.3f" , F_ROUND(value, 3));
    else
        sprintf(tempstr, "%.1f" , F_ROUND(value, 1));
	
	string str(tempstr);
	AddStringText(outpt, str, justify, totalspace, addstr);
}

void RAWSYSReportGenerator::AddStringText(ostream& outpt, const int& value, const string& justify, const uint& totalspace, const string& addstr)
{
	char tempstr[BUFFER_SIZE];
	
	sprintf(tempstr, "%i" , value);
	string str(tempstr);
	AddStringText(outpt, str, justify, totalspace, addstr);
}

void RAWSYSReportGenerator::AddSeparator(ostream& outpt, const string& str, const uint& fullsize)
{
	outpt << "|";
	AddChar(fullsize-2, str, outpt);
	outpt << "|" << endl;
}

/** Generate the SYSRAW report by system */
void RAWSYSReportGenerator::GenerateSystem(Alignment* alignment, const string& systm, ostream& output)
{
	int sizeofSpeaker = 7;
	int sizeofSegments = 5;
	int sizeofRefWords = 6;
	int sizeofCorrectWords = 6;
	int sizeofSubstitutions = 6;
	int sizeofSpeakerErrors = 6;
	int sizeofDeletions = 6;
	int sizeofInsertions = 6;
	int sizeofErrors = 6;
	int sizeofSegmentsErrors = 6;
	int sizeofNCE = 9;
	int sizeoffullline;
	bool isStarDiezePlus = false;
	string stradd;
	map<string, RAWSYS_Datas*>::iterator i, ei;
	size_t j;
	/* Boolean to decide to include the speakerError column */
	bool incSpErr = (string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0);
	
	vector<int> VecSegments, VecRefWords, VecRefWordsNotNull, VecCorrectWords, VecCorrectWordsNotNull, VecSubstitutions, VecSubstitutionsNotNull, VecSpeakerErrors, VecSpeakerErrorsNotNull, VecDeletions, VecDeletionsNotNull, VecInsertions, VecInsertionsNotNull, VecErrors, VecErrorsNotNull, VecSegmentsErrors;
	vector<double> VecPctCorrectWordsNotNull, VecPctSubstitutionsNotNull, VecPctSpeakerErrorsNotNull, VecPctDeletionsNotNull, VecPctInsertionsNotNull, VecPctErrorsNotNull, VecPctSegmentsErrors;
	vector<double> VecNCENotNull;
	
	/* TODO catch datas */
	AlignedSpeechIterator* aAlignedSpeechs;
	AlignedSpeech* aAlignedSpeechCurrent;
	AlignedSegmentIterator* aAlignedSegments;
	AlignedSegment* aAlignedSegmentCurrent;
	Segment* aSegment;
	TokenAlignment* aTokenAlignmentCurrent;
	TokenAlignment::AlignmentEvaluation* aAlignmentEvaluation;
	TokenAlignment::AlignmentResult aAlignmentResult;
	Token* aToken;
	bool errorinsegment;
	string speakerId;
	
	aAlignedSpeechs = alignment->AlignedSpeeches();
	
	while(aAlignedSpeechs->Current(&aAlignedSpeechCurrent))
	{
		aAlignedSegments = aAlignedSpeechCurrent->AlignedSegments();
		
		while(aAlignedSegments->Current(&aAlignedSegmentCurrent))
		{
			aSegment = aAlignedSegmentCurrent->GetReferenceSegment();
			errorinsegment = false;
			speakerId = aSegment->GetSpeakerId();
			
			if(!m_MapDatas[speakerId])
				m_MapDatas[speakerId] = new RAWSYS_Datas;
			
			for(j=0; j<aAlignedSegmentCurrent->GetTokenAlignmentCount(); ++j)
			{
				aTokenAlignmentCurrent = aAlignedSegmentCurrent->GetTokenAlignmentAt(j);
				aAlignmentEvaluation = aTokenAlignmentCurrent->GetAlignmentFor(systm);
				aAlignmentResult = aAlignmentEvaluation->GetResult();
				aToken = aAlignmentEvaluation->GetToken();
				
				if(aAlignmentResult == TokenAlignment::UNAVAILABLE)
				{
					LOG_ERR(logger, "Scoring not set. Report cannot be done!");
					return;
				}
				
				if(aAlignmentResult == TokenAlignment::CORRECT)
				{
                    if (aToken)
                    {
                        m_SumTotConfidenceCorrect += m_MapDatas[speakerId]->AddCorrectWord(aToken->GetConfidence());
                    }
                    else
                    {
                        m_MapDatas[speakerId]->AddCorrectWord();
                    }
				}
				
				if(aAlignmentResult == TokenAlignment::SUBSTITUTION)
				{
					errorinsegment = true;
					m_SumTotConfidenceIncorrect += m_MapDatas[speakerId]->AddSubstitutionWord(aToken->GetConfidence());
				}
								
				if(aAlignmentResult == TokenAlignment::SPEAKERSUB)
				{
					errorinsegment = true;
					m_SumTotConfidenceIncorrect += m_MapDatas[speakerId]->AddSpeakerErrorWord(aToken->GetConfidence());
				}
				
				if(aAlignmentResult == TokenAlignment::DELETION)
				{
					errorinsegment = true;
					m_MapDatas[speakerId]->AddDeletionWord();
				}
				
				if(aAlignmentResult == TokenAlignment::INSERTION)
				{
					errorinsegment = true;
					m_SumTotConfidenceIncorrect += m_MapDatas[speakerId]->AddInsertionWord(aToken->GetConfidence());
				}
			}
			
			m_MapDatas[speakerId]->AddSegment(errorinsegment);
		}
		
		if(aAlignedSegments)
			delete aAlignedSegments;
	}
	
	if(aAlignedSpeechs)
		delete aAlignedSpeechs;
	
	i = m_MapDatas.begin();
	ei = m_MapDatas.end();
		
	while(i != ei)
	{
		sizeofSpeaker = max(sizeofSpeaker, static_cast<int>(i->first.length()) );
		
		VecSegments.push_back(i->second->GetNumberSegments());
		VecRefWords.push_back(i->second->GetNumberRefWords());
		VecCorrectWords.push_back(i->second->GetNumberCorrectWords());
		VecSubstitutions.push_back(i->second->GetNumberSubstitutions());
		VecSpeakerErrors.push_back(i->second->GetNumberSpeakerErrors());
		VecDeletions.push_back(i->second->GetNumberDeletions());
		VecInsertions.push_back(i->second->GetNumberInsertions());
		VecErrors.push_back(i->second->GetNumberErrors());
		VecSegmentsErrors.push_back(i->second->GetNumberSegmentsErrors());
		VecPctSegmentsErrors.push_back(100.0*i->second->GetNumberSegmentsErrors()/i->second->GetNumberSegments());
		
		//cout << "NCE: " << i->second->GetNCE() << endl;
		
		if( (i->second->GetNumberRefWords() != 0) && (fabs(i->second->GetNCE()) <= 1000.0) )
			VecNCENotNull.push_back(i->second->GetNCE());
				
		if(i->second->GetNumberRefWords() != 0)
		{
			VecRefWordsNotNull.push_back(i->second->GetNumberRefWords());
			VecCorrectWordsNotNull.push_back(i->second->GetNumberCorrectWords());
			VecSubstitutionsNotNull.push_back(i->second->GetNumberSubstitutions());
			VecSpeakerErrorsNotNull.push_back(i->second->GetNumberSpeakerErrors());
			VecDeletionsNotNull.push_back(i->second->GetNumberDeletions());
			VecInsertionsNotNull.push_back(i->second->GetNumberInsertions());
			VecErrorsNotNull.push_back(i->second->GetNumberErrors());
		
			VecPctCorrectWordsNotNull.push_back(100.0*i->second->GetNumberCorrectWords()/i->second->GetNumberRefWords());
			VecPctSubstitutionsNotNull.push_back(100.0*i->second->GetNumberSubstitutions()/i->second->GetNumberRefWords());
			VecPctSpeakerErrorsNotNull.push_back(100.0*i->second->GetNumberSpeakerErrors()/i->second->GetNumberRefWords());
			VecPctDeletionsNotNull.push_back(100.0*i->second->GetNumberDeletions()/i->second->GetNumberRefWords());
			VecPctInsertionsNotNull.push_back(100.0*i->second->GetNumberInsertions()/i->second->GetNumberRefWords());
			VecPctErrorsNotNull.push_back(100.0*i->second->GetNumberErrors()/i->second->GetNumberRefWords());			
		}
		else
			isStarDiezePlus = true;
		
		++i;
	}
	
	Statistics StatisticsSegments(VecSegments);
	Statistics StatisticsRefWords(VecRefWords);
	Statistics StatisticsRefWordsNotNull(VecRefWordsNotNull);
	Statistics StatisticsCorrectWords(VecCorrectWords);
	Statistics StatisticsCorrectWordsNotNull(VecCorrectWordsNotNull);
	Statistics StatisticsPctCorrectWordsNotNull(VecPctCorrectWordsNotNull);
	Statistics StatisticsSubstitutions(VecSubstitutions);
	Statistics StatisticsSubstitutionsNotNull(VecSubstitutionsNotNull);
	Statistics StatisticsPctSubstitutionsNotNull(VecPctSubstitutionsNotNull);
	Statistics StatisticsSpeakerErrors(VecSpeakerErrors);
	Statistics StatisticsSpeakerErrorsNotNull(VecSpeakerErrorsNotNull);
	Statistics StatisticsPctSpeakerErrorsNotNull(VecPctSpeakerErrorsNotNull);
	Statistics StatisticsDeletions(VecDeletions);
	Statistics StatisticsDeletionsNotNull(VecDeletionsNotNull);
	Statistics StatisticsPctDeletionsNotNull(VecPctDeletionsNotNull);
	Statistics StatisticsInsertions(VecInsertions);
	Statistics StatisticsInsertionsNotNull(VecInsertionsNotNull);
	Statistics StatisticsPctInsertionsNotNull(VecPctInsertionsNotNull);
	Statistics StatisticsErrors(VecErrors);
	Statistics StatisticsErrorsNotNull(VecErrorsNotNull);
	Statistics StatisticsPctErrorsNotNull(VecPctErrorsNotNull);
	Statistics StatisticsSegmentsErrors(VecSegmentsErrors);
	Statistics StatisticsPctSegmentsErrors(VecPctSegmentsErrors);
	Statistics StatisticsNCENotNull(VecNCENotNull);
	
	sizeofSegments = max(sizeofSegments, StatisticsSegments.GetMaxSizeString());
	sizeofRefWords = max(sizeofRefWords, StatisticsRefWords.GetMaxSizeString());
	sizeofCorrectWords = max(sizeofCorrectWords, StatisticsCorrectWords.GetMaxSizeString());
	sizeofCorrectWords = max(sizeofCorrectWords, StatisticsCorrectWordsNotNull.GetMaxSizeString());
	sizeofSubstitutions = max(sizeofSubstitutions, StatisticsSubstitutions.GetMaxSizeString());
	sizeofSubstitutions = max(sizeofSubstitutions, StatisticsSubstitutionsNotNull.GetMaxSizeString());
	sizeofSpeakerErrors = max(sizeofSpeakerErrors, StatisticsSpeakerErrors.GetMaxSizeString());
	sizeofSpeakerErrors = max(sizeofSpeakerErrors, StatisticsSpeakerErrorsNotNull.GetMaxSizeString());
	sizeofDeletions = max(sizeofDeletions, StatisticsDeletions.GetMaxSizeString());
	sizeofDeletions = max(sizeofDeletions, StatisticsDeletionsNotNull.GetMaxSizeString());
	sizeofInsertions = max(sizeofInsertions, StatisticsInsertions.GetMaxSizeString());
	sizeofInsertions = max(sizeofInsertions, StatisticsInsertionsNotNull.GetMaxSizeString());
	sizeofErrors = max(sizeofErrors, StatisticsErrors.GetMaxSizeString());
	sizeofErrors = max(sizeofErrors, StatisticsErrorsNotNull.GetMaxSizeString());
	sizeofSegmentsErrors = max(sizeofSegmentsErrors, StatisticsSegmentsErrors.GetMaxSizeString());
	sizeofSegmentsErrors = max(sizeofSegmentsErrors, StatisticsPctSegmentsErrors.GetMaxSizeString());
	
	if(isStarDiezePlus)
	{
		++sizeofCorrectWords;
		++sizeofSubstitutions;
		++sizeofSpeakerErrors;
		++sizeofDeletions;
		++sizeofInsertions;
		++sizeofErrors;
	}
	
	sizeoffullline = 19+(incSpErr?1:0)+sizeofSpeaker+sizeofSegments+sizeofRefWords+sizeofCorrectWords+sizeofSubstitutions+(incSpErr ? sizeofSpeakerErrors : 0)+sizeofDeletions+sizeofInsertions+sizeofErrors+sizeofSegmentsErrors+sizeofNCE;
	
	AddStringText(output, "SYSTEM SUMMARY PERCENTAGE by SPEAKER", "middle", sizeoffullline);
	output << endl;
	output << ".";
	AddChar(sizeoffullline-2, "-", output);
	output << "." << endl;
	
	output << "|"; 
	AddStringText(output, GetFileNameFromPath(systm), "middle", sizeoffullline-2);
	output << "|" << endl;
	
	AddSeparator(output, "-", sizeoffullline);
	
	output << "| ";
	AddStringText(output, "Speaker", "left", sizeofSpeaker);
	output << " | ";
	AddStringText(output, "#Snt", "right", sizeofSegments);
	output << " ";
	AddStringText(output, "#Wrd", "right", sizeofRefWords);
	output << " | ";
	AddStringText(output, "Corr", "right", sizeofCorrectWords);
	output << " ";
	AddStringText(output, "Sub", "right", sizeofSubstitutions);
	if (incSpErr)
	{
		output << " ";
		AddStringText(output, "SpSub", "right", sizeofSpeakerErrors);
	}
	output << " ";
	AddStringText(output, "Del", "right", sizeofDeletions);
	output << " ";
	AddStringText(output, "Ins", "right", sizeofInsertions);
	output << " ";
	AddStringText(output, "Err", "right", sizeofErrors);
	output << " ";
	AddStringText(output, "S.Err", "right", sizeofSegmentsErrors);
	output << " | ";
	AddStringText(output, "NCE", "middle", sizeofNCE);	
	output << " |" << endl;
	
	i = m_MapDatas.begin();
	ei = m_MapDatas.end();
	
	while(i != ei)
	{
		AddSeparator(output, "-", sizeoffullline);
		
		output << "| ";
		AddStringText(output, i->first, "left", sizeofSpeaker);
		
		output << " | ";
		AddStringText(output, static_cast<int>(i->second->GetNumberSegments()), "right", sizeofSegments);
		
		output << " ";
		AddStringText(output, static_cast<int>(i->second->GetNumberRefWords()), "right", sizeofRefWords);
		
		output << " | ";
		
		if(isStarDiezePlus)
		{
			stradd = " ";
			
			if( (i->second->GetNumberRefWords() == 0) && (m_RawSys != 1) )
				stradd = "*";
		}
		else
		{
			stradd = "";
		}
		
		if(m_RawSys == 1)	// Raw
			AddStringText(output, static_cast<int>(i->second->GetNumberCorrectWords()), "right", sizeofCorrectWords, stradd);
		else // Sys - percentage
		{
			if(i->second->GetNumberRefWords() == 0)
				AddStringText(output, static_cast<int>(i->second->GetNumberCorrectWords()), "right", sizeofCorrectWords, stradd);
			else
				AddStringText(output, i->second->GetPercentCorrectWords(), 1, "right", sizeofCorrectWords, stradd);
		}
		
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, static_cast<int>(i->second->GetNumberSubstitutions()), "right", sizeofSubstitutions, stradd);
		else // Sys - percentage
		{
			if(i->second->GetNumberRefWords() == 0)
				AddStringText(output, static_cast<int>(i->second->GetNumberSubstitutions()), "right", sizeofSubstitutions, stradd);
			else
				AddStringText(output, i->second->GetPercentSubstitutions(), 1, "right", sizeofSubstitutions, stradd);
		}
		if (incSpErr)
		{
			output << " ";
			if(m_RawSys == 1)	// Raw
				AddStringText(output, static_cast<int>(i->second->GetNumberSpeakerErrors()), "right", sizeofSpeakerErrors, stradd);
			else // Sys - percentage
			{
				if(i->second->GetNumberRefWords() == 0)
					AddStringText(output, static_cast<int>(i->second->GetNumberSpeakerErrors()), "right", sizeofSpeakerErrors, stradd);
				else
					AddStringText(output, i->second->GetPercentSpeakerErrors(), 1, "right", sizeofSpeakerErrors, stradd);
			}
		}
		
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, static_cast<int>(i->second->GetNumberDeletions()), "right", sizeofDeletions, stradd);
		else // Sys - percentage
		{
			if(i->second->GetNumberRefWords() == 0)
				AddStringText(output, static_cast<int>(i->second->GetNumberDeletions()), "right", sizeofDeletions, stradd);
			else
				AddStringText(output, i->second->GetPercentDeletions(), 1, "right", sizeofDeletions, stradd);
		}
		
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, static_cast<int>(i->second->GetNumberInsertions()), "right", sizeofInsertions, stradd);
		else // Sys - percentage
		{
			if(i->second->GetNumberRefWords() == 0)
				AddStringText(output, static_cast<int>(i->second->GetNumberInsertions()), "right", sizeofInsertions, stradd);
			else
				AddStringText(output, i->second->GetPercentInsertions(), 1, "right", sizeofInsertions, stradd);
		}
		
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, static_cast<int>(i->second->GetNumberErrors()), "right", sizeofErrors, stradd);
		else // Sys - percentage
		{
			if(i->second->GetNumberRefWords() == 0)
				AddStringText(output, static_cast<int>(i->second->GetNumberErrors()), "right", sizeofErrors, stradd);
			else
				AddStringText(output, i->second->GetPercentErrors(), 1, "right", sizeofErrors, stradd);
		}
		
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, static_cast<int>(i->second->GetNumberSegmentsErrors()), "right", sizeofSegmentsErrors);
		else // Sys - percentage
			AddStringText(output, i->second->GetPercentSegmentsErrors(), 1, "right", sizeofSegmentsErrors);
		
		output << " | ";
		if( (i->second->GetNumberRefWords() != 0) && (fabs(i->second->GetNCE()) <= 1000.0) )
			AddStringText(output, i->second->GetNCE(), 3, "middle", sizeofNCE);
		else if( i->second->GetNumberRefWords() == 0 )
			AddStringText(output, "#", "middle", sizeofNCE);
		else if(i->second->GetNCE() <= -1000.0)
			AddStringText(output, "-inf", "middle", sizeofNCE);
		else if(i->second->GetNCE() >= 1000.0)
			AddStringText(output, "+inf", "middle", sizeofNCE);
		
		output << " |" << endl;
		
		++i;
	}
	
	AddSeparator(output, "=", sizeoffullline);
	
	output << "| ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, "Sum", "left", sizeofSpeaker);
	else // Sys - percentage
		AddStringText(output, "Sum/Avg", "left", sizeofSpeaker);
	
	output << " | ";
	AddStringText(output, static_cast<int>(StatisticsSegments.GetSum()), "right", sizeofSegments);
	
	output << " ";
	AddStringText(output, static_cast<int>(StatisticsRefWords.GetSum()), "right", sizeofRefWords);
	
	if(isStarDiezePlus)
		stradd = " ";
	else
		stradd = "";
	
	output << " | ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, static_cast<int>(StatisticsCorrectWords.GetSum()), "right", sizeofCorrectWords, stradd);
	else // Sys - percentage
		AddStringText(output, (StatisticsRefWords.GetSum(1) == 0) ? 0.0 : 100.0*StatisticsCorrectWords.GetSum()/StatisticsRefWords.GetSum(1), 1, "right", sizeofCorrectWords, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, static_cast<int>(StatisticsSubstitutions.GetSum()), "right", sizeofSubstitutions, stradd);
	else // Sys - percentage
		AddStringText(output, (StatisticsRefWords.GetSum(1) == 0) ? 0.0 : 100.0*StatisticsSubstitutions.GetSum()/StatisticsRefWords.GetSum(1), 1, "right", sizeofSubstitutions, stradd);
	
	if (incSpErr)
	{
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, static_cast<int>(StatisticsSpeakerErrors.GetSum()), "right", sizeofSpeakerErrors, stradd);
		else // Sys - percentage
			AddStringText(output, (StatisticsRefWords.GetSum(1) == 0) ? 0.0 : 100.0*StatisticsSpeakerErrors.GetSum()/StatisticsRefWords.GetSum(1), 1, "right", sizeofSpeakerErrors, stradd);
	}
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, static_cast<int>(StatisticsDeletions.GetSum()), "right", sizeofDeletions, stradd);
	else // Sys - percentage
		AddStringText(output, (StatisticsRefWords.GetSum(1) == 0) ? 0.0 : 100.0*StatisticsDeletions.GetSum()/StatisticsRefWords.GetSum(1), 1, "right", sizeofDeletions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, static_cast<int>(StatisticsInsertions.GetSum()), "right", sizeofInsertions, stradd);
	else // Sys - percentage
		AddStringText(output, (StatisticsRefWords.GetSum(1) == 0) ? 0.0 : 100.0*StatisticsInsertions.GetSum()/StatisticsRefWords.GetSum(1), 1, "right", sizeofInsertions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, static_cast<int>(StatisticsErrors.GetSum()), "right", sizeofErrors, stradd);
	else // Sys - percentage
		AddStringText(output, (StatisticsRefWords.GetSum(1) == 0) ? 0.0 : 100.0*StatisticsErrors.GetSum()/StatisticsRefWords.GetSum(1), 1, "right", sizeofErrors, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, static_cast<int>(StatisticsSegmentsErrors.GetSum()), "right", sizeofSegmentsErrors);
	else // Sys - percentage
		AddStringText(output, 100.0*StatisticsSegmentsErrors.GetSum()/StatisticsSegments.GetSum(1), 1, "right", sizeofSegmentsErrors);
	
	output << " | ";
	AddStringText(output, GetTotalNCE(StatisticsCorrectWords.GetSum(), StatisticsInsertions.GetSum(), StatisticsSubstitutions.GetSum(), StatisticsSpeakerErrors.GetSum()), 3, "middle", sizeofNCE);
	
	output << " |" << endl;
	
	AddSeparator(output, "=", sizeoffullline);
	
	output << "| ";
	AddStringText(output, "Mean", "middle", sizeofSpeaker);
	
	output << " | ";
	AddStringText(output, StatisticsSegments.GetMean(), 1, "right", sizeofSegments);
	
	output << " ";
	AddStringText(output, StatisticsRefWords.GetMean(), 1, "right", sizeofRefWords);
	
	if(isStarDiezePlus)
		stradd = "+";
	else
		stradd = "";
	
	output << " | ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsCorrectWordsNotNull.GetMean(), 1, "right", sizeofCorrectWords, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctCorrectWordsNotNull.GetMean(), 1, "right", sizeofCorrectWords, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsSubstitutionsNotNull.GetMean(), 1, "right", sizeofSubstitutions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctSubstitutionsNotNull.GetMean(), 1, "right", sizeofSubstitutions, stradd);
	
	if (incSpErr){
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, StatisticsSpeakerErrorsNotNull.GetMean(), 1, "right", sizeofSpeakerErrors, stradd);
		else // Sys - percentage
			AddStringText(output, StatisticsPctSpeakerErrorsNotNull.GetMean(), 1, "right", sizeofSpeakerErrors, stradd);
	}
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsDeletionsNotNull.GetMean(), 1, "right", sizeofDeletions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctDeletionsNotNull.GetMean(), 1, "right", sizeofDeletions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsInsertionsNotNull.GetMean(), 1, "right", sizeofInsertions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctInsertionsNotNull.GetMean(), 1, "right", sizeofInsertions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsErrorsNotNull.GetMean(),  1,"right", sizeofErrors, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctErrorsNotNull.GetMean(), 1, "right", sizeofErrors, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsSegmentsErrors.GetMean(), 1, "right", sizeofSegmentsErrors);
	else // Sys - percentage
                AddStringText(output, StatisticsPctSegmentsErrors.GetMean(), 1, "right", sizeofSegmentsErrors);
	
	output << " | ";
	AddStringText(output, StatisticsNCENotNull.GetMean(), 3, "middle", sizeofNCE);
	
	output << " |" << endl;
	
	output << "| ";
	AddStringText(output, "S.D.", "middle", sizeofSpeaker);
	
	output << " | ";
	AddStringText(output, StatisticsSegments.GetSD(), 1, "right", sizeofSegments);
	
	output << " ";
	AddStringText(output, StatisticsRefWords.GetSD(), 1, "right", sizeofRefWords);
	
	if(isStarDiezePlus)
		stradd = "+";
	else
		stradd = "";
	
	output << " | ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsCorrectWordsNotNull.GetSD(), 1, "right", sizeofCorrectWords, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctCorrectWordsNotNull.GetSD(), 1, "right", sizeofCorrectWords, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsSubstitutionsNotNull.GetSD(), 1, "right", sizeofSubstitutions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctSubstitutionsNotNull.GetSD(), 1, "right", sizeofSubstitutions, stradd);
	
	if (incSpErr)
	{
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, StatisticsSpeakerErrorsNotNull.GetSD(), 1, "right", sizeofSpeakerErrors, stradd);
		else // Sys - percentage
			AddStringText(output, StatisticsPctSpeakerErrorsNotNull.GetSD(), 1, "right", sizeofSpeakerErrors, stradd);
	}
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsDeletionsNotNull.GetSD(), 1, "right", sizeofDeletions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctDeletionsNotNull.GetSD(), 1, "right", sizeofDeletions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsInsertionsNotNull.GetSD(), 1, "right", sizeofInsertions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctInsertionsNotNull.GetSD(), 1, "right", sizeofInsertions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsErrorsNotNull.GetSD(), 1, "right", sizeofErrors, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctErrorsNotNull.GetSD(), 1, "right", sizeofErrors, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsSegmentsErrors.GetSD(), 1, "right", sizeofSegmentsErrors);
	else // Sys - percentage
		AddStringText(output, StatisticsPctSegmentsErrors.GetSD(), 1, "right", sizeofSegmentsErrors);
	
	output << " | ";
	AddStringText(output, StatisticsNCENotNull.GetSD(), 3, "middle", sizeofNCE);
	
	output << " |" << endl;
	
	output << "| ";
	AddStringText(output, "Median", "middle", sizeofSpeaker);
	
	output << " | ";
	AddStringText(output, StatisticsSegments.GetMedian(), 1, "right", sizeofSegments);
	
	output << " ";
	AddStringText(output, StatisticsRefWords.GetMedian(), 1, "right", sizeofRefWords);
	
	if(isStarDiezePlus)
		stradd = "+";
	else
		stradd = "";
	
	output << " | ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsCorrectWordsNotNull.GetMedian(), 1, "right", sizeofCorrectWords, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctCorrectWordsNotNull.GetMedian(), 1, "right", sizeofCorrectWords, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsSubstitutionsNotNull.GetMedian(), 1, "right", sizeofSubstitutions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctSubstitutionsNotNull.GetMedian(), 1, "right", sizeofSubstitutions, stradd);
	
	if (incSpErr)
	{
		output << " ";
		if(m_RawSys == 1)	// Raw
			AddStringText(output, StatisticsSpeakerErrorsNotNull.GetMedian(), 1, "right", sizeofSpeakerErrors, stradd);
		else // Sys - percentage
			AddStringText(output, StatisticsPctSpeakerErrorsNotNull.GetMedian(), 1, "right", sizeofSpeakerErrors, stradd);
	}
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsDeletionsNotNull.GetMedian(), 1, "right", sizeofDeletions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctDeletionsNotNull.GetMedian(), 1, "right", sizeofDeletions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsInsertionsNotNull.GetMedian(), 1, "right", sizeofInsertions, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctInsertionsNotNull.GetMedian(), 1, "right", sizeofInsertions, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsErrorsNotNull.GetMedian(), 1, "right", sizeofErrors, stradd);
	else // Sys - percentage
		AddStringText(output, StatisticsPctErrorsNotNull.GetMedian(), 1, "right", sizeofErrors, stradd);
	
	output << " ";
	if(m_RawSys == 1)	// Raw
		AddStringText(output, StatisticsSegmentsErrors.GetMedian(), 1, "right", sizeofSegmentsErrors);
	else // Sys - percentage
		AddStringText(output, StatisticsPctSegmentsErrors.GetMedian(), 1, "right", sizeofSegmentsErrors);
	
	output << " | ";
	AddStringText(output, StatisticsNCENotNull.GetMedian(), 3, "middle", sizeofNCE);
	
	output << " |" << endl;
	
	output << "'";
	AddChar(sizeoffullline-2, "-", output);
	output << "'" << endl;
	
	if(isStarDiezePlus)
	{
		output << "* No Reference words for this(these) speaker(s). Word counts supplied" << endl;
		output << "  rather than percents." << endl;
		output << "# No Reference words for this(these) speaker(s). NCE not computable." << endl;
		output << "+ Speaker(s) with no reference data is(are) ignored." << endl;
	}
	
	VecSegments.clear();
	VecRefWords.clear();
	VecRefWordsNotNull.clear();
	VecCorrectWords.clear();
	VecCorrectWordsNotNull.clear();
	VecSubstitutions.clear();
	VecSubstitutionsNotNull.clear();
	VecSpeakerErrors.clear();
	VecSpeakerErrorsNotNull.clear();
	VecDeletions.clear();
	VecDeletionsNotNull.clear();
	VecInsertions.clear();
	VecInsertionsNotNull.clear();
	VecErrors.clear();
	VecErrorsNotNull.clear();
	VecSegmentsErrors.clear();
	VecNCENotNull.clear();
	VecPctCorrectWordsNotNull.clear();
	VecPctSubstitutionsNotNull.clear();
	VecPctSpeakerErrorsNotNull.clear();
	VecPctDeletionsNotNull.clear();
	VecPctInsertionsNotNull.clear();
	VecPctErrorsNotNull.clear();
	VecSegmentsErrors.clear();
	VecPctSegmentsErrors.clear();

}

/** Generate the SYS report */
void RAWSYSReportGenerator::Generate(Alignment* alignment, int where)
{
	string extensionfile;
	
	if(m_RawSys == 1) // Raw
		extensionfile = "raw";
	else // Sys
		extensionfile = "sys";
	
	for(size_t i=0; i<alignment->GetNbOfSystems(); ++i)
	{
		ofstream file;
		
		if(where == 1)
		{
			string filename;
			
			if(Properties::GetProperty("report.outputdir") == string(""))
				filename = alignment->GetSystemFilename(i) + "." + extensionfile;
			else
				filename = Properties::GetProperty("report.outputdir") + "/" + GetFileNameFromPath(alignment->GetSystemFilename(i)) + "." + extensionfile;
				
			file.open(filename.c_str());
			
			if(! file.is_open())
			{
				LOG_ERR(logger, "Could not open file '" + filename + "' for " + extensionfile + " report, the output will be redirected in the stdout to avoid any lost.");
				where = 1;
			}
			else
			{
				LOG_INFO(logger, "Generating " + extensionfile + " report file '" + filename + "'.");
			}
		}
		else
		{
			LOG_INFO(logger, "Generating " + extensionfile + " report in the stdout.");
		}
		
		ostream output(where == 1 ? file.rdbuf() : cout.rdbuf());
		GenerateSystem(alignment, alignment->GetSystem(i), output);
		
		if(where == 1)
			file.close(); 
	}
}

double RAWSYSReportGenerator::GetTotalNCE(const double& numcorrects, const double& numinsertions, const double& numsubstitutions, const double& numspeakererrors)
{
	if((numcorrects + numinsertions + numsubstitutions + numspeakererrors) == 0)
		return -2000.0;
	
	double pc = min(max(0.0000001, numcorrects/(numcorrects + numinsertions + numsubstitutions + numspeakererrors)), 0.9999999);
	
	double Hmax = -numcorrects*log(pc)*log2e-(numinsertions + numsubstitutions + numspeakererrors)*log(1.0-pc)*log2e;
	
	if(Hmax == 0.0)
		return -2000.0;
	else
		return((Hmax+m_SumTotConfidenceCorrect+m_SumTotConfidenceIncorrect)/Hmax);
}
