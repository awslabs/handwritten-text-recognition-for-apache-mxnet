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
 
#include "stdinc.h"
#include "logger.h"
#include "recording.h"
#include "properties.h"

#define ASC_VERSION "1.10"
#define ASC_NAME "Ma Douce"

struct inputfilename
{
	string filename, fileformat, title;
};

void PrintHelp()
{
	cout << "Asclite version: " << ASC_VERSION << " (\"" << ASC_NAME << "\")" << endl;
	cout << "Usage: asclite <Input Options> [<Alignment Options>] [<Output Options>] [Scoring Report Options>]" << endl;
	cout << "Input Options:" << endl;
	cout << "    -r <reffilename> [ rttm | stm | trn ]" << endl;
	cout << "                  Defines the reference file and its format." << endl;
	cout << "                  The default format is 'trn'." << endl;
	cout << "    -h <hypfilename> [ ctm | rttm | trn ] [ <title> ]" << endl;
	cout << "                  Defines the hypothesis file, its format, and the 'title' used" << endl;
	cout << "                  for reports. The default title is the filename." << endl;
	cout << "                  This option may be used more than once." << endl;
	cout << "                  The default format is 'trn'." << endl;
	cout << "    -i <ids>      Set the utterance id type (for trn mode only)" << endl;
	cout << "Filter Options:" << endl;
	cout << "    -spkrautooverlap [ ref | hyp | both ]" << endl;
	cout << "                  Check if the speakers are self-overlaping or not." << endl;
	cout << "    -uem <uemfilename> [ ref | hyp | both ]" << endl;
	cout << "                  Apply the UEM rules." << endl;
	cout << "                  The default value is 'both'." << endl;
	cout << "    -noisg        Do not creates the Inter Segment Gaps." << endl;
	cout << "    -spkrmap      Check speaker map coincide with the Speaker ref and Hyp." << endl;
	cout << "Alignment Options:" << endl;
	cout << "    -s            Do case-sensitive alignments." << endl;
	cout << "    -F            Score fragments as correct." << endl;
	cout << "    -D [ ref | hyp | both ]" << endl;
	cout << "                  Score optional tokens as correct based on the flag's value:" << endl;
	cout << "                  hyp: a hypothesis word is optional and the work is deleted," << endl;
	cout << "                  ref: a reference word is optional and the word is inserted," << endl;
	cout << "                  both: both 'ref' and 'hyp' are in effect." << endl;
	cout << "                  The default value is 'both'." << endl;
	cout << "    -time-base-cost" << endl;
	cout << "                  Used the time base cost model for the alignment." << endl;
	cout << "    -time-prune <time_threshold>" << endl;
	cout << "                  Activates the time pruning optimization with allowed error of <time_threshold> millisecond(s)." << endl;
	cout << "    -word-time-align <time_threshold>" << endl;
	cout << "                  Activates the time optimization for word alignment with allowed error of <time_threshold> millisecond(s)." << endl;
    cout << "    -spkr-align <mdm_spkr_align_file_csv>" << endl;
    cout << "                  Activates the speaker alignment optimization regarding the <mdm_spkr_align_file_csv> csv file." << endl;
	cout << "    -adaptive-cost" << endl;
    cout << "                  Activates the adaptive cost based on the time." << endl;
	cout << "    -wordalign-cost" << endl;
    cout << "                  Activates the word align cost based on the edit distance." << endl;
    cout << "    -overlap-limit <max_nb_of_overlaping speaker>" << endl;
    cout << "                  Change the maximum number of overlaping speaker (default: 4)." << endl;
    cout << "    -overlap-min <min_nb_of_overlaping speaker>" << endl;
    cout << "                  Change the minimum number of overlaping speaker (default: 0)." << endl;
    cout << "    -only-SG <segment_group_ID>" << endl;
    cout << "                  Only score the segment group specified (default: disable, scoring all Segment Groups)." << endl;
	cout << "    -memory-compression <block_KB>" << endl;
    cout << "                  Set the memory compression with compressed <block_KB> KB block (default: off / recommanded: 256)." << endl;
	cout << "    -force-memory-compression" << endl;
    cout << "                  Force the memory compression." << endl;
	cout << "    -memory-limit <max_GB>" << endl;
    cout << "                  Set the maximum memory allocation in GB for the LCM (default: 2.0)." << endl;
	cout << "                  If <max_GB> is smaller then 2 GB then every segmentation above <max_GB> will not be aligned." << endl;
	cout << "                  If <max_GB> is bigger 2 GB and memory compression has been activated" << endl;
	cout << "                    then every segmentation above <max_GB> will not be aligned" << endl;
	cout << "                     and every segmentation between 2 GB and <max_GB> will be aligned." << endl;
	cout << "                  If <max_GB> is bigger 2 GB and memory compression has not been activated" << endl;
	cout << "                    every segmentation above 2 GB will not be aligned." << endl;
	cout << "    -difficulty-limit <max_GB>" << endl;
    cout << "                  Set the maximum difficulty limit in GB for the LCM (disabled)." << endl;
	cout << "                  Every segmentation above this limit will not be aligned." << endl;
	cout << "    -min-difficulty-limit <max_GB>" << endl;
    cout << "                  Set the min difficulty limit in GB for the LCM (disabled)." << endl;
	cout << "                  Every segmentation below this limit will not be aligned." << endl;
	cout << "    -generic-cost This options trigger the generic cost model computing the results disregarding the notion" << endl;
	cout << "                  of hyp and ref, the results will be a multiple-alignment." << endl;
	cout << "                  The hypotheses files are the only input." << endl;
	cout << "    -threads <nb_threads>" << endl;
	cout << "                  Set the number of threads to <nb_threads> for parallel computation (default: 1)." << endl;
	cout << "Output Options:" << endl;
	cout << "    -O <output_dir>" << endl;
	cout << "                  Writes all output files into output_dir." << endl;
	cout << "                  The default directory is the hypfile's." << endl;
	cout << "    -f <level>    Defines feedback mode." << endl;
	cout << "                  The default value is 4." << endl;
    cout << "                  Silent: 0." << endl;
	cout << "                  Alignment: 6." << endl;
	cout << "                  Alignment only: 7." << endl;
	cout << "    -l <width>    Defines the line width used for reports." << endl;
	cout << "Scoring Report Options:" << endl;
	cout << "    -o [ sum | rsum | sgml ] [ sdtout ]" << endl;
	cout << "                  Defines the output reports." << endl;
	cout << "                  The default value is 'sum stdout'." << endl;
	cout << endl;
	cout << "Note:" << endl;
	cout << "LCM:              Levenshtein Cost Matrix." << endl;
	exit(E_OK);
}

int main(int argc, char **argv)
{
	Logger* logger = Logger::getLogger();
	
	inputfilename reffile;
	bool arg_bref = false;
	int arg_index = 1;
	bool arg_ok = true;
	bool arg_butterance = false;
	string arg_utterance = "spu_id";
	bool arg_bcasesensitive = false;
	bool arg_bfragcorrect = false;
	bool arg_boptionaltoken = false;
	bool arg_btimepruneoptimization = false;
	string arg_timepruneoptimizationthreshold = "0";
	bool arg_btimewordoptimization = false;
	string arg_timewordoptimizationthreshold = "0";
    bool arg_bspeakeroptimization = false;
	string arg_speakeroptimizationfilename = "";
	bool arg_bmaxnboverlapingspkr = false;
	string arg_maxnboverlapingspkr = "4";
	bool arg_bminnboverlapingspkr = false;
	string arg_minnboverlapingspkr = "0";
	bool arg_bonlysg = false;
	string arg_onlysg = "";
	bool arg_bmemorycompression = false;
	string arg_memorycompression = "256";
	bool arg_bforcememorycompression = false;
	string arg_optionaltoken = "none";
	bool arg_bproperty = false;
	bool arg_badaptivecost = false;
	bool arg_bwordaligncost = false;
	string arg_outputdir = "";
	int arg_feedback = 4;
	int arg_width = 27;
    bool arg_bmaxgb = false;
    string arg_maxgb = "1";
	bool arg_bdifficultygb = false;
    string arg_difficultygb = "16";
	bool arg_bmindifficultygb = false;
    string arg_mindifficultygb = "0";
   
    string arg_uemfilename = "";
    string arg_uemoption = "";
    bool arg_buem = false;
    
    bool arg_buseISG = true;
    
    bool arg_bcheckspkrmap = false;
    
    string arg_spkrautooverlapoption = "";
    bool arg_bspkrautooverlap = false;
    
    bool arg_btimebasecost = false;
    bool arg_bgenericcost = false;
	
	string arg_nbthreads = "1";
	
	vector<string> arg_vecouput;
	vector<string> arg_filters;
	vector<inputfilename> vecHyps;
	map<string, string> arg_properties;
	
	if(argc == 1)
		PrintHelp();
	
	while( (arg_index < argc) && arg_ok)
	{
		// Hypothesis
		if(strcmp(argv[arg_index], "-h") == 0)
		{
			if(arg_index < argc-1)
			{
				inputfilename hypfile;
				bool titlefound = false;
				arg_index++;
				hypfile.filename = string(argv[arg_index]);
				
				if(arg_index < argc-1)
				{
					arg_index++;
					
					if(strcmp(argv[arg_index], "ctm") == 0)
						hypfile.fileformat = string("ctm");
					else if(strcmp(argv[arg_index], "rttm") == 0)
						hypfile.fileformat = string("rttm");
					else if(strcmp(argv[arg_index], "trn") == 0)
						hypfile.fileformat = string("trn");
					else if(argv[arg_index][0] != '-')
					{
						arg_index--;
						titlefound = true;
						hypfile.fileformat = string("trn");
						hypfile.title = string(argv[arg_index]);
					}
					else
					{
						arg_index--;
						hypfile.fileformat = string("trn");
					}
					
					if(!titlefound && (arg_index < argc-1) )
						if(argv[arg_index+1][0] != '-')
						{
							titlefound = true;
							arg_index++;
							hypfile.title = string(argv[arg_index]);
						}
				}
				else
					hypfile.fileformat = string("trn");
				
				if(!titlefound)
					hypfile.title = string("");

				vecHyps.push_back(hypfile);
			}
			else
			{
				cerr << "[  ERROR  ] Hypothesis filename missing!" << endl;
				arg_ok = false;
			}
		}
		else
		// References
		if(strcmp(argv[arg_index], "-r") == 0)
		{
			if(arg_index < argc-1)
			{
				arg_index++;
				reffile.filename = string(argv[arg_index]);
				
				if(arg_index < argc-1)
				{
					arg_index++;
					
					if(strcmp(argv[arg_index], "stm") == 0)
						reffile.fileformat = string("stm");
					else if(strcmp(argv[arg_index], "rttm") == 0)
						reffile.fileformat = string("rttm");
					else if(strcmp(argv[arg_index], "trn") == 0)
						reffile.fileformat = string("trn");
					else if(argv[arg_index][0] != '-')
					{
						arg_index--;
						reffile.fileformat = string("trn");
					}
					else
					{
						arg_index--;
						reffile.fileformat = string("trn");
					}
				}
				else
					reffile.fileformat = string("trn");
				
				reffile.title = string("");
				arg_bref = true;
			}
			else
			{
				cerr << "[  ERROR  ] Reference filename missing!" << endl;
				arg_ok = false;
			}
		}
		else
        // uem
        if(strcmp(argv[arg_index], "-uem") == 0)
		{
			if(arg_index < argc-1)
			{
				arg_index++;
				arg_uemfilename = string(argv[arg_index]);
				
				if(arg_index < argc-1)
				{
					arg_index++;
					
					if(strcmp(argv[arg_index], "ref") == 0)
						arg_uemoption = string("ref");
					else if(strcmp(argv[arg_index], "hyp") == 0)
						arg_uemoption = string("hyp");
					else if(strcmp(argv[arg_index], "both") == 0)
						arg_uemoption = string("both");
					else if(argv[arg_index][0] != '-')
					{
						arg_index--;
						arg_uemoption = string("both");
					}
					else
					{
						arg_index--;
						arg_uemoption = string("both");
					}
				}
				else
					arg_uemoption = string("both");
				
				arg_buem = true;
			}
			else
			{
				cerr << "[  ERROR  ] UEM filename missing!" << endl;
				arg_ok = false;
			}
		}
		else
		// Speaker Auto-overlap
        if(strcmp(argv[arg_index], "-spkrautooverlap") == 0)
		{
			if(arg_index < argc-1)
			{
				arg_index++;
				
				if(strcmp(argv[arg_index], "ref") == 0)
					arg_spkrautooverlapoption = string("ref");
				else if(strcmp(argv[arg_index], "hyp") == 0)
					arg_spkrautooverlapoption = string("hyp");
				else if(strcmp(argv[arg_index], "both") == 0)
					arg_spkrautooverlapoption = string("both");
				else if(argv[arg_index][0] != '-')
				{
					arg_index--;
					arg_spkrautooverlapoption = string("both");
				}
				else
				{
					arg_index--;
					arg_spkrautooverlapoption = string("both");
				}
			}
			else
				arg_spkrautooverlapoption = string("both");
			
			arg_bspkrautooverlap = true;
		}
		else
		// noisg
		if(strcmp(argv[arg_index], "-noisg") == 0)
		{
			arg_buseISG = false;
		}
		else
		// spkrmap
		if(strcmp(argv[arg_index], "-spkrmap") == 0)
		{
			arg_bcheckspkrmap = false;
		}
		else
		// Utterance
		if(strcmp(argv[arg_index], "-i") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_butterance = true;
					arg_utterance = string(argv[arg_index]);
				}
			}
			
			if(!arg_butterance)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Utterance missing!" << endl;
			}
		}
		else
		// Time prune optimization
		if(strcmp(argv[arg_index], "-time-prune") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_btimepruneoptimization = true;
					arg_timepruneoptimizationthreshold = string(argv[arg_index]);
				}
			}
			
			if(!arg_btimepruneoptimization)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Optimization prune threshold missing!" << endl;
			}
		}
		else
		// Time word optimization
		if(strcmp(argv[arg_index], "-word-time-align") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_btimewordoptimization = true;
					arg_timewordoptimizationthreshold = string(argv[arg_index]);
				}
			}
			
			if(!arg_btimewordoptimization)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Optimization word threshold missing!" << endl;
			}
		}
		else
        // Speaker optimization
		if(strcmp(argv[arg_index], "-spkr-align") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_bspeakeroptimization = true;
					arg_speakeroptimizationfilename = string(argv[arg_index]);
				}
			}
			
			if(!arg_bspeakeroptimization)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Speaker Optimization file missing!" << endl;
			}
		}
		else
		// Memory Limit
		if(strcmp(argv[arg_index], "-memory-limit") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
                    arg_bmaxgb = true;
					arg_maxgb = string(argv[arg_index]);
				}
			}
			
			if(!arg_bmaxgb)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Max GB missing!" << endl;
			}
		}
		else
		// Difficulty Limit
		if(strcmp(argv[arg_index], "-difficulty-limit") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
                    arg_bdifficultygb = true;
					arg_difficultygb = string(argv[arg_index]);
				}
			}
			
			if(!arg_bdifficultygb)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Difficulty GB missing!" << endl;
			}
		}
		else
		// Min Difficulty Limit
		if(strcmp(argv[arg_index], "-min-difficulty-limit") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
                    arg_bmindifficultygb = true;
					arg_mindifficultygb = string(argv[arg_index]);
				}
			}
			
			if(!arg_bmindifficultygb)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Min Difficulty GB missing!" << endl;
			}
		}
		else
		// Memory Compression
		if(strcmp(argv[arg_index], "-memory-compression") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
                    arg_bmemorycompression = true;
					arg_memorycompression = string(argv[arg_index]);
				}
			}
			
			if(!arg_bmemorycompression)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] block KB missing!" << endl;
			}
		}
		else
        // Max overlap
		if(strcmp(argv[arg_index], "-overlap-limit") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_bmaxnboverlapingspkr = true;
					arg_maxnboverlapingspkr = string(argv[arg_index]);
				}
			}
			
			if(!arg_bmaxnboverlapingspkr)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Max number of overlaping speaker missing!" << endl;
			}
		}
		else
		// Min overlap
		if(strcmp(argv[arg_index], "-overlap-min") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_bminnboverlapingspkr = true;
					arg_minnboverlapingspkr = string(argv[arg_index]);
				}
			}
			
			if(!arg_bminnboverlapingspkr)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Min number of overlaping speaker missing!" << endl;
			}
		}
		else
		// Only Segment Group
		if(strcmp(argv[arg_index], "-only-SG") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_bonlysg = true;
					arg_onlysg = string(argv[arg_index]);
				}
			}
			
			if(!arg_bonlysg)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Segment Group ID missing!" << endl;
			}
		}
		else
		// Set Property
		if(strcmp(argv[arg_index], "--set-property") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					string p_name = string(argv[arg_index]);
					
					if(arg_index < argc-1)
					{
						if(argv[arg_index+1][0] != '-')
						{
              				arg_index++;
							arg_bproperty = true;
					    	arg_properties[p_name] = string(argv[arg_index]);
						}
					}
				}
			}
			
			if(!arg_bproperty)
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Property need to be set with 2 args, name => value" << endl;
			}
		}
		else
		// Case-sensitive
		if(strcmp(argv[arg_index], "-s") == 0)
		{
			arg_bcasesensitive = true;
		}
		else
		// Adaptive cost
		if(strcmp(argv[arg_index], "-adaptive-cost") == 0)
		{
			arg_badaptivecost = true;
		}
		else
		// Generic cost
		if(strcmp(argv[arg_index], "-generic-cost") == 0)
		{
			arg_bgenericcost = true;
		}
		else
		// Time Base Cost
		if(strcmp(argv[arg_index], "-time-base-cost") == 0)
		{
			arg_btimebasecost = true;
		}
		else
		// Word Align cost
		if(strcmp(argv[arg_index], "-wordalign-cost") == 0)
		{
			arg_bwordaligncost = true;
		}
		else
		// Force Memory Compression
		if(strcmp(argv[arg_index], "-force-memory-compression") == 0)
		{
			arg_bforcememorycompression = true;
			arg_bmemorycompression = true;
		}
		else
		// Fragments
		if(strcmp(argv[arg_index], "-F") == 0)
		{
			arg_bfragcorrect = true;
		}
		else
		// Optional token
		if(strcmp(argv[arg_index], "-D") == 0)
		{
			if(arg_index < argc-1)
			{
				if( (argv[arg_index+1][0] != '-') &&
						( (strcmp(argv[arg_index+1], "ref")) ||
							(strcmp(argv[arg_index+1], "hyp")) ||
							(strcmp(argv[arg_index+1], "both")) ) )
				{
					arg_index++;
					arg_boptionaltoken = true;
					arg_optionaltoken = string(argv[arg_index]);
				}
				else
				{
					arg_optionaltoken = string("both");
				}
			}
			else
			{
				arg_optionaltoken = string("both");
			}
		}
		else
		// Output dir
		if(strcmp(argv[arg_index], "-O") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_outputdir = string(argv[arg_index]);
				}
			}
			else
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Ouput dir missing!" << endl;
			}
		}
		else
		// Feedback
		if(strcmp(argv[arg_index], "-f") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_feedback = atoi(argv[arg_index]);
				}
			}
			else
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Feedback value missing!" << endl;
			}
		}
		else
		// Width
		if(strcmp(argv[arg_index], "-l") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_width = atoi(argv[arg_index]);
				}
			}
			else
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Width value missing!" << endl;
			}
		}
		else
		// Output
		if(strcmp(argv[arg_index], "-o") == 0)
		{
			int j=0;
			
			while( (arg_index < argc-1) && (j<4) && arg_ok)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					
					if(strcmp(argv[arg_index], "sum") == 0)
					{
						arg_vecouput.push_back("sum");
					}
					else if(strcmp(argv[arg_index], "rsum") == 0)
					{
						arg_vecouput.push_back("rsum");
					}
					else if(strcmp(argv[arg_index], "sgml") == 0)
					{
						arg_vecouput.push_back("sgml");
					}
					else if(strcmp(argv[arg_index], "stdout") == 0)
					{
						arg_vecouput.push_back("stdout");
					}
					else
					{
						arg_index--;
						arg_ok = false;
						cerr << "[  ERROR  ] Parameter incorrect!" << endl;
					}
				}
				
				j++;
			}
			
			if(arg_vecouput.empty())
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Report missing!" << endl;
			}
		}
		else
		// Number of threads
		if(strcmp(argv[arg_index], "-threads") == 0)
		{
			if(arg_index < argc-1)
			{
				if(argv[arg_index+1][0] != '-')
				{
					arg_index++;
					arg_nbthreads = string(argv[arg_index]);
				}
			
				if(atoi(arg_nbthreads.c_str()) < 1)
				{
					arg_ok = false;
					cerr << "[  ERROR  ] Number of threads must be >= 1!" << endl;
				}
			}
			else
			{
				arg_ok = false;
				cerr << "[  ERROR  ] Number of threads missing!" << endl;
			}
		}
		else
		{
			arg_ok = false;
			cerr << "[  ERROR  ] Parameter unknown: " << argv[arg_index] << endl;
		}
		
		arg_index++;
	}
	
	if(!arg_bgenericcost)
	{
		// Hyp-Ref
		if(!arg_bref || vecHyps.empty())
		{
			cerr << "[  ERROR  ] A least one hypothesis and one reference are needed!" << endl;
			arg_ok = false;
		}
		
		arg_ok = arg_ok && arg_bref && !vecHyps.empty();
		
		if( arg_btimebasecost && ( (vecHyps.begin()->fileformat == "trn") || (reffile.fileformat == "trn") || (reffile.fileformat == "stm") ) )
		{
			cerr << "[  ERROR  ] Time based alignment can only be done with input format allowing token time: (ctm and rttm)." << endl;
			arg_ok = false;
		}
		
		if(!vecHyps.empty() && arg_bref)
		{
			vector<inputfilename>::iterator i = vecHyps.begin();
			vector<inputfilename>::iterator ei = vecHyps.end();
			
			bool bformat = true;
			
			while(i != ei)
			{
				if( !( ( (i->fileformat == "trn") && (reffile.fileformat == "trn") )  ||
					   ( (i->fileformat == "ctm") && (reffile.fileformat == "stm") )  ||
					   ( (i->fileformat == "rttm") && (reffile.fileformat == "stm") ) ||
					   ( (i->fileformat == "rttm") && (reffile.fileformat == "rttm") ) ) )
				{
					cerr << "[  ERROR  ] (H) " << i->fileformat << " vs. (R) " << reffile.fileformat << endl;
					bformat = false;
				}
				
				++i;
			}
			
			if(!bformat)
			{
				cerr << "[  ERROR  ] Hyp must be a ctm  and Ref a stm  or" << endl;
				cerr << "[  ERROR  ] Hyp must be a rttm and Ref a stm  or" << endl;
				cerr << "[  ERROR  ] Hyp must be a rttm and Ref a rttm or" << endl;
				cerr << "[  ERROR  ] Hyp and Ref must be trn's." << endl;
				arg_ok = false;
			}
		}
		
		if(arg_vecouput.empty())
		{
			arg_vecouput.push_back("sum");
			arg_vecouput.push_back("stdout");
		}
	}
	else
	{
		// Generic
		if(vecHyps.empty())
		{
			cerr << "[  ERROR  ] A least one hypothesis is needed!" << endl;
			arg_ok = false;
		}
		
		if(arg_bref)
		{
			arg_bref = false;
			cerr << "[  INFO   ] The generic process requires only hypothesis files, the reference will be ignored." << endl;
		}
		
		bool isStdout = ( arg_vecouput.end() != find(arg_vecouput.begin(), arg_vecouput.end(), "stdout") );

		arg_vecouput.clear();
		//arg_vecouput.push_back("sgml");
		
		if(arg_buseISG)
		{
			arg_buseISG = false;
			cerr << "[  INFO   ] ISG will be ignored." << endl;
		}
		
		if(isStdout)
			arg_vecouput.push_back("stdout");
			
		if(arg_bspkrautooverlap)
		{
			if( (arg_spkrautooverlapoption == string("hyp")) || (arg_spkrautooverlapoption == string("both")) )
				arg_spkrautooverlapoption == string("hyp");
			else
				arg_bspkrautooverlap = false;
		}
		
		if(arg_buem)
		{
			if( (arg_uemoption == string("hyp")) || (arg_uemoption == string("both")) )
				arg_uemoption == string("hyp");
			else
				arg_buem = false;
		}
		
		if( (arg_optionaltoken == string("hyp")) || (arg_optionaltoken == string("both")) )
				arg_optionaltoken == string("hyp");
	}
		
	if(!arg_ok)
	{
		cerr << "[  FATAL  ] type 'asclite' to display the help." << endl;
		exit(E_ARGS);
	}
	else
	{
		logger->setLogLevel(arg_feedback);
		
		// Set properties
		Properties::Initialize();
		Properties::SetProperty("inputparser.trn.uid", arg_utterance);
		Properties::SetProperty("align.case_sensitive", arg_bcasesensitive ? "true" : "false");
		Properties::SetProperty("align.fragment_are_correct", arg_bfragcorrect ? "true" : "false");
		Properties::SetProperty("align.timepruneoptimization", arg_btimepruneoptimization ? "true" : "false");
		Properties::SetProperty("align.timepruneoptimizationthreshold", arg_timepruneoptimizationthreshold);
		Properties::SetProperty("align.timewordoptimization", arg_btimewordoptimization ? "true" : "false");
		Properties::SetProperty("align.timewordoptimizationthreshold", arg_timewordoptimizationthreshold);
		Properties::SetProperty("align.speakeroptimization", arg_bspeakeroptimization ? "true" : "false");
		Properties::SetProperty("align.memorycompression", arg_bmemorycompression ? "true" : "false");
		Properties::SetProperty("align.forcememorycompression", arg_bforcememorycompression ? "true" : "false");
		Properties::SetProperty("align.memorycompressionblock", arg_memorycompression);
		Properties::SetProperty("align.adaptivecost", arg_badaptivecost ? "true" : "false");
		Properties::SetProperty("align.wordaligncost", arg_bwordaligncost ? "true" : "false");
		Properties::SetProperty("align.genericmethod", arg_bgenericcost ? "true" : "false");
		
		if(arg_btimebasecost)
			Properties::SetProperty("align.typecost", "2");
		else
			Properties::SetProperty("align.typecost", "1");
			
		Properties::SetProperty("threads.number", arg_nbthreads);
		
		Properties::SetProperty("recording.maxspeakeroverlaping", arg_maxnboverlapingspkr);
		Properties::SetProperty("recording.minspeakeroverlaping", arg_minnboverlapingspkr);
		Properties::SetProperty("recording.bonlysg", arg_bonlysg ? "true" : "false");
		Properties::SetProperty("recording.onlysg", arg_onlysg);
        Properties::SetProperty("recording.maxnbofgb", arg_maxgb);
		Properties::SetProperty("recording.difficultygb", arg_bdifficultygb ? "true" : "false");
		Properties::SetProperty("recording.nbrdifficultygb", arg_difficultygb);
		Properties::SetProperty("recording.mindifficultygb", arg_bmindifficultygb ? "true" : "false");
		Properties::SetProperty("recording.minnbrdifficultygb", arg_mindifficultygb);
		Properties::SetProperty("recording.maxoverlapinghypothesis", "1");
		
		if(vecHyps.begin()->fileformat == "rttm")
            Properties::SetProperty("recording.maxoverlapinghypothesis", arg_maxnboverlapingspkr);

		Properties::SetProperty("align.optionally", arg_optionaltoken);
		Properties::SetProperty("general.feedback.level", ""+arg_feedback);
        Properties::SetProperty("report.outputdir", arg_outputdir);
		Properties::SetProperty("align.type", "lev");
		Properties::SetProperty("score.type", "stt");
        
        Properties::SetProperty("filter.uem.isg", arg_buseISG ? "true" : "false");
        
        // Speaker Auto Overlap
        if(arg_bspkrautooverlap)
        {
        	Properties::SetProperty("filter.spkrautooverlap", "true");
            Properties::SetProperty("filter.spkrautooverlap.option", arg_spkrautooverlapoption);
			arg_filters.push_back("filter.spkrautooverlap");
		}
		
		//Spkr mapping Check
		Properties::SetProperty("filter.spkrmap", arg_bcheckspkrmap ? "true" : "false");
        
        // UEM
        if(reffile.fileformat == "trn")
        {
        	LOG_INFO(logger, "TRN file forces UEM and ISG to none!");
        	Properties::SetProperty("filter.uem.isg", "false");
        }
        else
        {
			if(arg_buem || arg_buseISG)
				arg_filters.push_back("filter.uem");
			
			if(arg_buem)
			{
				Properties::SetProperty("filter.uem", "true");
				Properties::SetProperty("filter.uem.option", arg_uemoption);
				Properties::SetProperty("filter.uem.arg_uemfilename", arg_uemfilename);
			}
			else
				Properties::SetProperty("filter.uem", "false");
        }
        
		map<string, string>::iterator i = arg_properties.begin();
		map<string, string>::iterator ei = arg_properties.end();
		
		while (i != ei)
		{
            Properties::SetProperty((*i).first, (*i).second);
            ++i;
        }
        
        ullint max_gb = static_cast<ullint>(ceil(1024*1024*1024*atof(Properties::GetProperty("recording.maxnbofgb").c_str())/sizeof(int)));
        
        char buffer_mem[BUFFER_SIZE];
        sprintf(buffer_mem, "Using %s GB of memory for computation graph. (%llu cells)", Properties::GetProperty("recording.maxnbofgb").c_str(), max_gb );
        LOG_INFO(logger, buffer_mem);
        
		//launch the processing with the prec args
		LOG_INFO(logger, "Initialize Recording");
		Recording* recording = new Recording;
		
		LOG_INFO(logger, "Loading Input files");
		
		vector<string> hyps_files;
		vector<string> hyps_titles;
		vector<string> hyps_formats;
		
		for (size_t i=0 ; i < vecHyps.size() ; ++i)
		{
            hyps_files.push_back(vecHyps[i].filename);
            hyps_titles.push_back(vecHyps[i].title);
            hyps_formats.push_back(vecHyps[i].fileformat);
        }
        
		if(!arg_bgenericcost)
			recording->Load(reffile.filename, reffile.fileformat, hyps_files, hyps_titles, hyps_formats, arg_uemfilename, arg_speakeroptimizationfilename);
		else
			recording->Load(hyps_files, hyps_titles, hyps_formats, arg_uemfilename, arg_speakeroptimizationfilename);
		
		LOG_INFO(logger, "Filter Input data");
		recording->Filter(arg_filters);
		
		LOG_INFO(logger, "Align Input data");
		recording->Align();
		
		LOG_INFO(logger, "Score Input data");
		recording->Score();
		
		LOG_INFO(logger, "Generate reports");
		recording->GenerateReport(arg_vecouput);
		
		//Deletion process
		delete recording;
		
		hyps_files.clear();
		hyps_titles.clear();
		hyps_formats.clear();
		
		LOG_INFO(logger, "Job complete");
	}
	
	arg_vecouput.clear();
	arg_filters.clear();
	vecHyps.clear();
	arg_properties.clear();
	
	delete logger;
	
    return E_OK;
}
