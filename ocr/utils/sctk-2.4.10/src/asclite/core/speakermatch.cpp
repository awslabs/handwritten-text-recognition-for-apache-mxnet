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
 * Database for matching speaker optimization
 */
 
#include "speakermatch.h" // class's header file

Logger* SpeakerMatch::logger = Logger::getLogger();

/** Constructor */
SpeakerMatch::SpeakerMatch()
{
	m_pMapSourceChannelSysRef = new map < string, string >;
}

/** Destructor */
SpeakerMatch::~SpeakerMatch()
{
	m_pMapSourceChannelSysRef->clear();
	
	if(m_pMapSourceChannelSysRef)
		delete m_pMapSourceChannelSysRef;
}

/** Load the file */
void SpeakerMatch::LoadFile(const string& filename)
{
	string line;
	ifstream file;
	long int lineNum = -1;
	
	file.open(filename.c_str(), ifstream::in);
    
	if (! file.is_open())
	{ 
		LOG_FATAL(logger, "Error opening SpeakerMatch file " + filename); 
		exit(E_LOAD); 
	}
	
	while (getline(file,line,'\n'))
    {
		++lineNum;
		
        if (line.find_first_of("#") == 0)
		{
			//comment so skip (for now)
		}
		else if(line == string("File,Channel,RefSpeaker,SysSpeaker,isMapped,timeOverlap"))
		{
			// CSV header
		}		
        else
        // if Type is 'LEXEME' we take it, unless we drop the line
        {
			size_t index = 0;
			size_t lpos = 0;
			string delim = string(",");
			size_t pos = line.find_first_of(delim, lpos);
			string tokens[6];
						
			do
			{
				if(index == 6)
				{
					char buffer[BUFFER_SIZE];
					sprintf(buffer, "Too much: Error parsing the line %li in file %s", lineNum, filename.c_str());
					LOG_ERR(logger, buffer);
				}
				
				tokens[index] = line.substr(lpos, pos - lpos);
				++index;
				lpos = ( pos == string::npos ) ?  string::npos : pos + 1;
				pos = line.find_first_of(delim, lpos);
			}
			while( lpos != string::npos );
			
			if(index != 6)
			{
				char buffer[BUFFER_SIZE];
				sprintf(buffer, "Too few: Error parsing the line %li in file %s", lineNum, filename.c_str());
				LOG_ERR(logger, buffer);
			}
			
			if( tokens[4] == string("mapped") )
			{
				//SetSysRef(string(l_file), string(l_channel), string(l_sys), string(l_ref));
				//transform(tokens[3].begin(), tokens[3].end(), tokens[3].begin(), (int(*)(int)) toupper);
				//transform(tokens[2].begin(), tokens[2].end(), tokens[2].begin(), (int(*)(int)) toupper);
				SetSysRef(tokens[0], tokens[1], tokens[3], tokens[2]);
			}
		}
    }
    
    file.close();
    LOG_INFO(logger, "loading of file " + filename + " done");
}

/** Initialize a couple, sys-ref */
void SpeakerMatch::SetSysRef(const string& source, const string& channel, const string& sys, const string& ref)
{
	string index("");
	index.append(source);
	index.append("|");
	index.append(channel);
	index.append("|");
	index.append(sys);
	
	if(m_pMapSourceChannelSysRef->find(index) == m_pMapSourceChannelSysRef->end())
	{
		m_pMapSourceChannelSysRef->insert(pair < string, string > (index, ref));
	}
	else
	{
		cerr << source << " " << channel << " " << sys << " exists! -> IGNORED" << endl;
		LOG_WARN(logger, source + " " + channel + " " + sys + " exists! -> IGNORED"); 
	}	
}

/** Return the ref coupled to the sys */
string SpeakerMatch::GetRef(const string& source, const string& channel, const string& sys)
{
	string index("");
	index.append(source);
	index.append("|");
	index.append(channel);
	index.append("|");
	index.append(sys);
	
	map < string, string >::iterator iter = m_pMapSourceChannelSysRef->find(index);
	
	if( iter != m_pMapSourceChannelSysRef->end() )
	{
		return(iter->second);
	}
	else
	{
		return( string("") );
	}
}
