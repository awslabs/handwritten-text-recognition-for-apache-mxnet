
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

#include "sgml_generic_reportgenerator.h"

Logger* SGMLGenericReportGenerator::m_pLogger = Logger::getLogger();

SGMLGenericReportGenerator::~SGMLGenericReportGenerator()
{
	for(size_t i=0; i<m_vGAS.size(); ++i)
		delete m_vGAS[i];
	
	m_vGAS.clear();
	m_vTitle.clear();
	m_vFilename.clear();
}

/** Generate the XML report */
void SGMLGenericReportGenerator::Generate(int where)
{
	ofstream file;
			
	if(where == 1)
	{
		string filename;
		
		if(Properties::GetProperty("report.outputdir") == string(""))
			filename = "generic-report.xml";
		else
			filename = Properties::GetProperty("report.outputdir") + "/generic-report.xml";
		
		file.open(filename.c_str());
	
		if(! file.is_open())
		{
			LOG_ERR(m_pLogger, "Could not open file '" + filename + "' for XML Generic report, the output will be redirected in the stdout to avoid any lost.");
			where = 1;
		}
		else
		{
			LOG_INFO(m_pLogger, "Generating XML Generic report file '" + filename + "'.");
		}
	}
	else
	{
		LOG_INFO(m_pLogger, "Generating XML Generic report in the stdout.");
	}
	
	ostream output(where == 1 ? file.rdbuf() : cout.rdbuf());
	
	output << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
	output << "<rover>\n";
	
	for(size_t i=0; i<GetNbOfSystems(); ++i)
		output << "\t<input title=\"" << GetTitle(i) << "\" filename=\"" << GetFilename(i) << "\" />\n";
	
	for(size_t i=0; i<m_vGAS.size(); ++i)
	{
		for(size_t j=0; j<m_vGAS[i]->GetNbOfGraphAlignedToken(); ++j)
		{
			string file = "";
			string channel = "";
		
			for(size_t k=0; k<m_vGAS[i]->GetGraphAlignedToken(j)->GetDimension(); ++k)
			{
				Token* toke = m_vGAS[i]->GetGraphAlignedToken(j)->GetToken(k);
				
				if(toke != NULL)
				{
					file = toke->GetParentSegment()->GetSource();
					channel = toke->GetParentSegment()->GetChannel();
				}
			}
		
			output << "\t<putative";
		
			if(file != "")
				output << " file=\"" << file << "\"";
			
			if(channel != "")
				output << " channel=\"" << channel << "\"";
			
			output << ">\n";

			GraphAlignedToken* gat = m_vGAS[i]->GetGraphAlignedToken(j);
			
			for(size_t k=0; k<gat->GetDimension(); ++k)
			{
				output << "\t\t<word";
				
				Token* tok = gat->GetToken(k);
				string srctext = string("");
				
				if(tok != NULL)
				{
					srctext = tok->GetSourceText();
					
					if(tok->GetStartTime() >= 0)
					{
						char buffer[BUFFER_SIZE];
						sprintf(buffer, "%.3f", ((double)(tok->GetStartTime()))/1000.0);
						output << " bt=\"" << string(buffer) << "\"";
					}
					
					if(tok->GetEndTime() >= 0)
					{
						char buffer[BUFFER_SIZE];
						sprintf(buffer, "%.3f", ((double)(tok->GetEndTime()))/1000.0);
						output << " et=\"" << string(buffer) << "\"";
					}
					
					if(tok->IsConfidenceSet())
					{
						char buffer [BUFFER_SIZE];
						sprintf(buffer, "%.6f", tok->GetConfidence());
						output << " conf=\"" << string(buffer) << "\"";
					}
					
					if(tok->GetParentSegment()->GetSpeakerId() != "")
						output << " speaker=\"" << tok->GetParentSegment()->GetSpeakerId() << "\"";
					
					output << " input=\"" << tok->GetParentSegment()->GetParentSpeech()->GetParentSpeechSet()->GetTitle() << "\"";
				}
				
				output << ">" << srctext << "</word>\n";
			}
			
			if(gat->GetDimension() < GetNbOfSystems())
				for(size_t k=0; k<(GetNbOfSystems()-gat->GetDimension()); ++k)
					output << "\t\t<word></word>\n";
					
			output << "\t</putative>\n";
		}
	}
	
	output << "</rover>\n";
	
	if(where == 1)
		file.close();
}
