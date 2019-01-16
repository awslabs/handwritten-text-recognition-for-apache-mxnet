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
 * Class that handle the parsing of TRN encoded file
 */
 
#include "trn_inputparser.h" // class's header file

Logger* TRNInputParser::logger = Logger::getLogger();

/**
 * Load the named file into a vector of Speech element.
 * @todo Finish this :P
 */
SpeechSet* TRNInputParser::loadFile(const string& name)
{
	string line;
	long int lineNum = -1;
	long int elementNum = 0;
	ifstream file;
	file.open(name.c_str(), ifstream::in);
    
	if (! file.is_open())
	{ 
		LOG_FATAL(logger, "Error opening file " + name); 
		exit (E_LOAD); 
	}
	
	map<string, Speech*> spkr_list;
	SpeechSet* vec = new SpeechSet(name);
	
	while (getline(file,line,'\n'))
	{
		++lineNum;
		//cout << "prec:"<< prec_seg->isEmpty() << endl;
		if (line.find_first_of(";;") == 0)
		{
			//comment so skip (for now)
		}
		else
		{
			size_t uidindex = line.find_last_of("(")+1;
			size_t uidsize = line.find_last_of(")")-uidindex;
			
			string uid = line.substr(uidindex, uidsize);
			string spkr = "undefined";
            
			if (Properties::GetProperty("inputparser.trn.uid").compare("spu_id") == 0)
			{
				spkr = uid.substr(0, uid.find_last_of("_-"));
			} 
			else
			{
				LOG_ERR(logger, "trn_importer : unknown uterance id type : " + Properties::GetProperty("inputparser.trn.uid"));
			}
            
			Speech* speech = spkr_list[spkr];
            
			if (!speech)
			{
				spkr_list[spkr] = new Speech(vec);
				speech = spkr_list[spkr];
			}    
                  
			Segment* seg = ParseWords(string(""), string(""), string(""), -1, -1, speech, line.substr(0, uidindex-1));
			seg->SetId("(" + uid + ")");
			seg->SetSourceLineNum(lineNum);
			seg->SetSourceElementNum(elementNum++);
			seg->SetSpeakerId(spkr);
			speech->AddSegment(seg);
		}
	}
    
	LOG_INFO(logger, "loading of file '" + name + "' done");
	file.close();
	
	map<string, Speech*>::iterator i = spkr_list.begin();
	map<string, Speech*>::iterator ei = spkr_list.end();
    
	bool emptyFile = true;
	
	while (i != ei)
	{
		vec->AddSpeech(i->second);
		emptyFile = false;
		++i;
	}
	
	if(emptyFile)
	{
		LOG_FATAL(logger, "TRN file '" + name + "' contains no data!");
		exit(E_MISSINFO);
	}
    
	return vec;
}
