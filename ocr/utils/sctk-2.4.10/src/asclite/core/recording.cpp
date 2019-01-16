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
 * A recording contain all the data needed to score a testset.
 * From the argument to the list of parser (input) and generator (output report).
 * This is the main entry to the aligner.
 */

#include "recording.h" // class's header file

Logger* Recording::logger = Logger::getLogger();

// class constructor
Recording::Recording()
{   
	//init the input parsers
	inputParsers["ctm"] = new CTMInputParser;
	inputParsers["stm"] = new STMInputParser;
	inputParsers["trn"] = new TRNInputParser;
    inputParsers["rttm"] = new RTTMInputParser;
	
	//init the report generator
	reportGenerators["rsum"] = new RAWSYSReportGenerator(1);
	reportGenerators["sum"] = new RAWSYSReportGenerator(2);
	reportGenerators["sgml"] = new SGMLReportGenerator();
	pSGMLGenericReportGenerator = new SGMLGenericReportGenerator();
	
	//init the segmentor
	segmentors["trn"] = new TRNTRNSegmentor();
	segmentors["ctmstmrttm"] = new CTMSTMRTTMSegmentor();
	
	//init the Aligner
	aligner["lev"] = new Levenshtein();
  
	//init the Scorer
	scorer["stt"] = new STTScorer();
  
	//init the Filters
	filters["filter.spkrautooverlap"] = new SpkrAutoOverlap();
	filters["filter.uem"] = new UEMFilter();
  
	//init the Alignment result
	alignments = new Alignment();
	
	//Init SpeakerMatch
	m_pSpeakerMatch = new SpeakerMatch();
}

// class destructor
Recording::~Recording()
{
	if(m_pSpeakerMatch)
		delete m_pSpeakerMatch;
	
	map<string, InputParser*>::iterator ipi, ipe;
	
	ipi = inputParsers.begin();
	ipe = inputParsers.end();
	
	while(ipi != ipe)
	{
		InputParser* ptr_elt = ipi->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++ipi;
	}
	
	inputParsers.clear();
	
	map<string, ReportGenerator*>::iterator rgi, rge;
	
	rgi = reportGenerators.begin();
	rge = reportGenerators.end();
	
	while(rgi != rge)
	{
		ReportGenerator* ptr_elt = rgi->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++rgi;
	}
	
	reportGenerators.clear();
	
	delete pSGMLGenericReportGenerator;
	
	map<string, Aligner*>::iterator ai, ae;
	
	ai = aligner.begin();
	ae = aligner.end();
	
	while(ai != ae)
	{
		Aligner* ptr_elt = ai->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++ai;
	}
	
	aligner.clear();
	
	map<string, Scorer*>::iterator si, se;
	
	si = scorer.begin();
	se = scorer.end();
	
	while(si != se)
	{
		Scorer* ptr_elt = si->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++si;
	}
	
	scorer.clear();
	
	map<string, Segmentor*>::iterator segi, sege;
	
	segi = segmentors.begin();
	sege = segmentors.end();
	
	while(segi != sege)
	{
		Segmentor* ptr_elt = segi->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++segi;
	}
	
	segmentors.clear();
	
	map<string, ::Filter*>::iterator fi, fe;
	
	fi = filters.begin();
	fe = filters.end();
	
	while(fi != fe)
	{
		::Filter* ptr_elt = fi->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++fi;
	}
	
	filters.clear();
	delete alignments;
	delete references;
	
	map<string, SpeechSet* >::iterator ssi, sse;
	
	ssi = hypothesis.begin();
	sse = hypothesis.end();
	
	while(ssi != sse)
	{
		SpeechSet* ptr_elt = ssi->second;
		
		if(ptr_elt)
			delete ptr_elt;
		
		++ssi;
	}
	
	hypothesis.clear();
}

/**
 * Load the reference & Hypothesis files into the system.
 * use the right loader based on the type.
 */
void Recording::Load(const string& _references, const string& _refType, const vector<string> & _hypothesis_files, const vector<string> & _hypothesis_titles, const vector<string> & _hypothesis_types, const string& uemFile, const string& speakeralignfile)
{
	m_bGenericAlignment = false;
	
	if(string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0)
		m_pSpeakerMatch->LoadFile(speakeralignfile);
		
	if(string("true").compare(Properties::GetProperty("filter.uem")) == 0)
		filters["filter.uem"]->LoadFile(uemFile);

    //init the segmentor
    if(_refType == string("trn"))
    	segmentor = segmentors["trn"];
    else
    	segmentor = segmentors["ctmstmrttm"];    	
  
	//load reference
	LOG_INFO(logger, "Load 1 reference of type "+_refType);
	references = inputParsers[_refType]->loadFile(_references);
    references->SetOrigin("ref");
	
	//load hypothesis
    char buffer[BUFFER_SIZE];
    sprintf(buffer, "Load %lu hypothesis", _hypothesis_files.size());
 	LOG_INFO(logger, buffer);
 	string title_temp;
    
    for (size_t i=0 ; i < _hypothesis_files.size() ; ++i)
	{
		if(string("rttm") == _hypothesis_types[i])
			inputParsers[_hypothesis_types[i]]->SetOneTokenPerSegment(true);
		
        SpeechSet* hyps_loaded = inputParsers[_hypothesis_types[i]]->loadFile(_hypothesis_files[i]);
		
        hyps_loaded->SetOrigin("hyp");
        
        if (_hypothesis_titles[i] == "")
            title_temp = _hypothesis_files[i];
        else
			title_temp = _hypothesis_titles[i];
		
		hyps_loaded->SetTitle(title_temp);
		
		hypothesis[title_temp] = hyps_loaded;
		alignments->AddSystem(_hypothesis_files[i], title_temp);
    }    
}

void Recording::Load(const vector<string> & _hypothesis_files, const vector<string> & _hypothesis_titles, const vector<string> & _hypothesis_types, const string& _uemFile, const string& _speakeralignfile)
{
	m_bGenericAlignment = true;
	
	if(string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0)
		m_pSpeakerMatch->LoadFile(_speakeralignfile);
		
	if(string("true").compare(Properties::GetProperty("filter.uem")) == 0)
		filters["filter.uem"]->LoadFile(_uemFile);

	//init the segmentor
    segmentor = segmentors["ctmstmrttm"];
    
    //load hypothesis
    char buffer[BUFFER_SIZE];
    sprintf(buffer, "Load %lu hypothesis", _hypothesis_files.size());
 	LOG_INFO(logger, buffer);
 	string title_temp;
    
    for (size_t i=0 ; i < _hypothesis_files.size() ; ++i)
	{
		if(string("rttm") == _hypothesis_types[i])
			inputParsers[_hypothesis_types[i]]->SetOneTokenPerSegment(true);
		
        SpeechSet* hyps_loaded = inputParsers[_hypothesis_types[i]]->loadFile(_hypothesis_files[i]);
		
        hyps_loaded->SetOrigin("gen");
        
        if (_hypothesis_titles[i] == "")
            title_temp = _hypothesis_files[i];
        else
			title_temp = _hypothesis_titles[i];
			
		hyps_loaded->SetTitle(title_temp);
		
		hypothesis[title_temp] = hyps_loaded;
		pSGMLGenericReportGenerator->AddTitleAndFilename(_hypothesis_files[i], title_temp);
    }
}

/**
 * Filter the references and hypothesis with the availables filters.
 */
void Recording::Filter(const vector<string> & _filters)
{
    unsigned long int nbErr = 0;
  
    for (size_t i=0 ; i < _filters.size() ; ++i)
    {
    	LOG_INFO(logger, "Filtering ==> " +_filters[i] + " - pass 1");
    
    	string refhypboth = Properties::GetProperty(_filters[i] + ".option");
    	    
    	if( (string(refhypboth).compare("ref") == 0) || (string(refhypboth).compare("both") == 0) )
    	{
    		LOG_INFO(logger, "Filtering ==> " + _filters[i] + " references");

			for (size_t j=0 ; j < references->GetNumberOfSpeech() ; ++j)
				nbErr += filters[_filters[i]]->ProcessSingleSpeech(references->GetSpeech(j));
        }
           	
        if( (string(refhypboth).compare("hyp") == 0) || (string(refhypboth).compare("both") == 0) )
        {
        	LOG_INFO(logger, "Filtering ==> " + _filters[i] + " hypotheses");
        	
        	map<string, SpeechSet* >::iterator hi  = hypothesis.begin();
			map<string, SpeechSet* >::iterator hei = hypothesis.end();
			
			while(hi != hei)
			{
				SpeechSet* spkset = hi->second;
				
				for(size_t j = 0; j < spkset->GetNumberOfSpeech(); ++j)
					nbErr += filters[_filters[i]]->ProcessSingleSpeech(spkset->GetSpeech(j));

				++hi;
			}
        }
    }
    
    for (size_t i=0 ; i < _filters.size() ; ++i)
    {
    	LOG_INFO(logger, "Filtering ==> " +_filters[i] + " - pass 2");
    
    	if(filters[_filters[i]]->isProcessAllSpeechSet())
    	{
    		LOG_INFO(logger, "Filtering ==> " + _filters[i] + " processing");
    		nbErr += filters[_filters[i]]->ProcessSpeechSet(references, hypothesis);
    	}
    }
    
    // Check the speaker mapping
    if(string("true").compare(Properties::GetProperty("align.speakeroptimization")) == 0)
    {
    	bool foundmatch = false;
    
    	for(size_t i=0 ; i < references->GetNumberOfSpeech() ; ++i)
    	{
    		for(size_t j=0 ; j < references->GetSpeech(i)->NbOfSegments() ; ++j)
    		{
    			string file1 = references->GetSpeech(i)->GetSegment(j)->GetSource();
				string channel1 = references->GetSpeech(i)->GetSegment(j)->GetChannel();
				string speaker1 = references->GetSpeech(i)->GetSegment(j)->GetSpeakerId();
				//transform(speaker1.begin(), speaker1.end(), speaker1.begin(), (int(*)(int)) toupper);
				
				map<string, SpeechSet* >::iterator hi  = hypothesis.begin();
				map<string, SpeechSet* >::iterator hei = hypothesis.end();
				
				while(hi != hei)
				{
					SpeechSet* spkset = hi->second;
					
					for(size_t k = 0; k < spkset->GetNumberOfSpeech(); ++k)
					{
						for(size_t l=0 ; l < spkset->GetSpeech(k)->NbOfSegments() ; ++l)
						{
							string file2 = spkset->GetSpeech(k)->GetSegment(l)->GetSource();
							string channel2 = spkset->GetSpeech(k)->GetSegment(l)->GetChannel();
							string speaker2 = spkset->GetSpeech(k)->GetSegment(l)->GetSpeakerId();
							//transform(speaker2.begin(), speaker2.end(), speaker2.begin(), (int(*)(int)) toupper);
							
							if( (file1 == file2) && (channel1 == channel2) && (m_pSpeakerMatch->GetRef(file1, channel1, speaker1) == speaker2) )
								foundmatch = true;
						}
					}						
	
					++hi;
				}
    		}
    	}
    	
    	if(!foundmatch)
    	{
    		char buffer[BUFFER_SIZE];
        	sprintf(buffer, "Speaker map checking - No matching Detected");
        	LOG_WARN(logger, buffer);
        	
        	if(string("true").compare(Properties::GetProperty("filter.spkrmap")) == 0)
        		nbErr++;
    	}
    }
  
    if (nbErr != 0)
    {
        char buffer[BUFFER_SIZE];
        sprintf(buffer, "%li Error(s) in the input files during the filtering", nbErr);
        LOG_FATAL(logger, buffer);
        exit(E_FILTER);
    }
}

/**
* Align the ref/gen with the select align algo
 */
void Recording::AlignGeneric()
{
	uint max_spkrOverlaping = atoi(Properties::GetProperty("recording.maxspeakeroverlaping").c_str());
	uint min_spkrOverlaping = atoi(Properties::GetProperty("recording.minspeakeroverlaping").c_str());
	ulint OnlySG_ID = 0;
	bool bOnlySG = (string("true").compare(Properties::GetProperty("recording.bonlysg")) == 0);
	
	if(bOnlySG)
		OnlySG_ID = atoi(Properties::GetProperty("recording.onlysg").c_str());
		
    ullint max_nb_gb = static_cast<ullint>(ceil(1024*1024*1024*atof(Properties::GetProperty("recording.maxnbofgb").c_str())/sizeof(int)));
	ullint max_nb_2gb = static_cast<ullint>(ceil(2.0*1024*1024*1024/sizeof(int)));
	bool bForceMemoryCompression = (string("true").compare(Properties::GetProperty("align.forcememorycompression")) == 0);
	bool bDifficultyLimit = (string("true").compare(Properties::GetProperty("recording.difficultygb")) == 0);
	bool bMinDifficultyLimit = (string("true").compare(Properties::GetProperty("recording.mindifficultygb")) == 0);
	
    if(logger->isAlignLogON())
		LOG_ALIGN(logger, "Aligned,SegGrpID,File,Channel,Eval[,RefSegID,RefSegBT,RefSegET,RefSpkrID,RefTknID,RefTknBT,RefTknET,RefTknTxt,RefTknConf,RefTknPrev,RefTknNext]*");
		
	LOG_INFO(logger, "Processing system");
	SegmentsGroup* segmentsGroup;
	segmentor->ResetGeneric(hypothesis);
		
	while (segmentor->HasNext())
	{
		segmentsGroup = segmentor->NextGeneric();
		ullint cellnumber = segmentsGroup->GetDifficultyNumber();

		double KBused = ((static_cast<double>(cellnumber))/1024.0) * (static_cast<double>(sizeof(int)));
		double MBused = KBused/1024.0;
		double GBused = MBused/1024.0;
		double TBused = GBused/1024.0;

		char buffer_size[BUFFER_SIZE];

		if(TBused > 1.0)
			sprintf(buffer_size, "%.2f TB", TBused);
		else if(GBused > 1.0)
			sprintf(buffer_size, "%.2f GB", GBused);
		else if(MBused > 1.0)
			sprintf(buffer_size, "%.2f MB", MBused);
		else
			sprintf(buffer_size, "%.2f kB", KBused);
  
		char buffer[BUFFER_SIZE];

		sprintf(buffer, "Align SG %lu [%s] %lu dimensions, Difficulty: %llu (%s) ---> bt=%.3f et=%.3f", 
						static_cast<ulint>(segmentsGroup->GetsID()), 
						segmentsGroup->GetDifficultyString().c_str(),
						static_cast<ulint>(segmentsGroup->GetNumberOfHypothesis()),
						cellnumber,
						buffer_size,
						segmentsGroup->GetMinTime()/1000.0,
						segmentsGroup->GetMaxTime()/1000.0);
   
		LOG_DEBUG(logger, buffer);
  
		bool ignoreSegs = false;
		bool buseCompArray = false;
		
		if(segmentsGroup->GetNumberOfHypothesis() <= /*1*/ 0)
		{
			//ignoreSegs = true;
			sprintf(buffer, "Skip this group of segments (%lu): no hypothesis/dimension", static_cast<ulint>(segmentsGroup->GetsID()) );
			LOG_WARN(logger, buffer);
		}
		
		if(!ignoreSegs && (segmentsGroup->isIgnoreInScoring()) )
		{
			ignoreSegs = true;
			sprintf(buffer, "Skip this group of segments (%lu): Ignore this time segments in scoring set into the references", static_cast<ulint>(segmentsGroup->GetsID()) );
			LOG_WARN(logger, buffer);
		}
		
		if(!ignoreSegs && bOnlySG )
		{
			if(segmentsGroup->GetsID() != OnlySG_ID)
			{
				ignoreSegs = true;
				sprintf(buffer, "Skip this group of segments (%lu): Only scoring %lu", static_cast<ulint>( segmentsGroup->GetsID()), OnlySG_ID);
				LOG_WARN(logger, buffer);
			}
		}
		
		if(!ignoreSegs &&  (segmentsGroup->GetNumberOfReferences() > max_spkrOverlaping) )
		{
			ignoreSegs = true;        
			sprintf(buffer, "Skip this group of segments (%lu): nb of reference speaker (%lu) overlaping to high (limit: %lu)", static_cast<ulint>(segmentsGroup->GetsID()), 
																																static_cast<ulint>(segmentsGroup->GetNumberOfReferences()), 
																																static_cast<ulint>(max_spkrOverlaping));
			LOG_WARN(logger, buffer);
		}
		
		if(!ignoreSegs &&  (segmentsGroup->GetNumberOfReferences() < min_spkrOverlaping) )
		{
			ignoreSegs = true;        
			sprintf(buffer, "Skip this group of segments (%lu): nb of reference speaker (%lu) overlaping to small (limit: %lu)", static_cast<ulint>(segmentsGroup->GetsID()), 
																																 static_cast<ulint>(segmentsGroup->GetNumberOfReferences()), 
																																 static_cast<ulint>(min_spkrOverlaping));
			LOG_WARN(logger, buffer);
		}
		
		if(!ignoreSegs && (bDifficultyLimit) )
		{
			ullint difficulty_nb_gb = static_cast<ullint>(ceil(1024*1024*1024*atof(Properties::GetProperty("recording.nbrdifficultygb").c_str())/sizeof(int)));
			
			if(cellnumber > difficulty_nb_gb)
			{
				ignoreSegs = true;
				sprintf(buffer, "Skip this group of segments (%lu): the graph size will be too large (%llu [%s]) regarding the difficulty limit (%llu [%.2f GB])", 
								static_cast<ulint>(segmentsGroup->GetsID()),
								cellnumber,
								buffer_size,
								difficulty_nb_gb,
								atof(Properties::GetProperty("recording.nbrdifficultygb").c_str()));
				LOG_WARN(logger, buffer);
			}
		}
		
		if(!ignoreSegs && (bMinDifficultyLimit) )
		{
			ullint mindifficulty_nb_gb = static_cast<ullint>(ceil(1024*1024*1024*atof(Properties::GetProperty("recording.minnbrdifficultygb").c_str())/sizeof(int)));
							
			if(cellnumber < mindifficulty_nb_gb)
			{
				ignoreSegs = true;
				sprintf(buffer, "Skip this group of segments (%lu): the graph size will be too small (%llu [%s]) regarding the min difficulty limit (%llu [%.2f GB])", 
								static_cast<ulint>(segmentsGroup->GetsID()),
								cellnumber,
								buffer_size,
								mindifficulty_nb_gb,
								atof(Properties::GetProperty("recording.minnbrdifficultygb").c_str()));
				LOG_WARN(logger, buffer);
			}
		}
		
		if(!ignoreSegs && ( (cellnumber > max_nb_gb) || (cellnumber > max_nb_2gb) ) )
		{
			bool m_bUseCompression = (string("true").compare(Properties::GetProperty("align.memorycompression")) == 0) || bForceMemoryCompression;
			
			if(!m_bUseCompression)
			{
				ignoreSegs = true;
				sprintf(buffer, "Skip this group of segments (%lu): the graph size will be too large (%llu [%s]) regarding the memory limit (%llu [%.2f GB]) and no compression have been set", 
								static_cast<ulint>(segmentsGroup->GetsID()),
								cellnumber,
								buffer_size,
								(cellnumber > max_nb_gb) ? max_nb_gb : max_nb_2gb,
								(cellnumber > max_nb_gb) ? atof(Properties::GetProperty("recording.maxnbofgb").c_str()) : 2.0);
				
				LOG_WARN(logger, buffer);
			}
			else
			{
				buseCompArray = true;
				sprintf(buffer, "Using Levenshtein Matrix Compression Algorithm for group of segments (%lu)", static_cast<ulint>(segmentsGroup->GetsID()));
				LOG_INFO(logger, buffer);
			}
		}
		
		if(!ignoreSegs && (bForceMemoryCompression) )
		{
			buseCompArray = true;
			sprintf(buffer, "Forcing Levenshtein Matrix Compression Algorihm for group of segments (%lu)", static_cast<ulint>(segmentsGroup->GetsID()));
			LOG_INFO(logger, buffer);
		}
		
		if(segmentsGroup)
		{
			if(logger->isAlignLogON())
				segmentsGroup->LoggingAlignment(m_bGenericAlignment, string("SG"));
		}
		
		//Special case where the SegmentGroup contain a segment time to ignore
		if (!ignoreSegs)
		{
			Aligner* aligner_instance = aligner["lev"];
			aligner_instance->SetSegments(segmentsGroup, m_pSpeakerMatch, buseCompArray);
			aligner_instance->Align();
			GraphAlignedSegment* gas = aligner_instance->GetResults();
			
			//cout << gas->ToString() << endl;
			pSGMLGenericReportGenerator->AddGraphAlignSegment(gas);

			sprintf(buffer, "%li GAS cost: (%d)", gas->GetNbOfGraphAlignedToken(), (dynamic_cast<Levenshtein*>(aligner_instance)->GetCost()));
			LOG_DEBUG(logger, buffer);
			
			if(logger->isAlignLogON())
				gas->LoggingAlignment(segmentsGroup->GetsID());
							
			//delete gas;
		}
		else //segmentsGroup ignored
		{
			if(logger->isAlignLogON())
				segmentsGroup->LoggingAlignment(m_bGenericAlignment, string("NO"));
		}
		
		if(segmentsGroup)
			delete segmentsGroup;
	}
}

void Recording::AlignHypRef()
{
	uint max_spkrOverlaping = atoi(Properties::GetProperty("recording.maxspeakeroverlaping").c_str());
	uint min_spkrOverlaping = atoi(Properties::GetProperty("recording.minspeakeroverlaping").c_str());
	ulint OnlySG_ID = 0;
	bool bOnlySG = (string("true").compare(Properties::GetProperty("recording.bonlysg")) == 0);
	
	if(bOnlySG)
		OnlySG_ID = atoi(Properties::GetProperty("recording.onlysg").c_str());
	
    ullint max_nb_gb = static_cast<ullint>( ceil(1024*1024*1024*atof(Properties::GetProperty("recording.maxnbofgb").c_str())/sizeof(int)) );
	ullint max_nb_2gb = static_cast<ullint>( ceil(2.0*1024*1024*1024/sizeof(int)) );
	bool bForceMemoryCompression = (string("true").compare(Properties::GetProperty("align.forcememorycompression")) == 0);
	bool bDifficultyLimit = (string("true").compare(Properties::GetProperty("recording.difficultygb")) == 0);
	bool bMinDifficultyLimit = (string("true").compare(Properties::GetProperty("recording.mindifficultygb")) == 0);
	
    map<string, SpeechSet* >::iterator i = hypothesis.begin();
    map<string, SpeechSet* >::iterator ei = hypothesis.end();

	if(logger->isAlignLogON())
		LOG_ALIGN(logger, "Aligned,SegGrpID,File,Channel,Eval,RefSegID,RefSegBT,RefSegET,RefSpkrID,RefTknID,RefTknBT,RefTknET,RefTknTxt,RefTknConf,RefTknPrev,RefTknNext,HypSegID,HypSegBT,HypSegET,HypSpkrID,HypTknID,HypTknBT,HypTknET,HypTknTxt,HypTknConf,HypTknPrev,HypTknNext");
	
    while(i != ei)
    {		
        string hyp_name = i->first;
        LOG_INFO(logger, "Processing system "+hyp_name);
        SegmentsGroup* segmentsGroup;
        segmentor->Reset(references, hypothesis[hyp_name]);
		        
        while (segmentor->HasNext())
        {
            segmentsGroup = segmentor->Next();
            
            //LOG_INFO(logger, segmentsGroup->ToString());
			
            //cerr << "Nbr ref: " << segmentsGroup->GetNumberOfReferences() << endl;
            //cerr << "Nbr hyp: " << segmentsGroup->GetNumberOfHypothesis() << endl;
			//cerr << "SG:" << segmentsGroup->GetsID() << endl;
			//cerr << segmentsGroup->ToString() << endl;
			
			bool emptyHypvsISG = false;
            
            //Special case where there is no hyp
            if (segmentsGroup->GetNumberOfHypothesis() == 0)
            {
                //May need to put the segment time
                Segment* emptyseg = Segment::CreateWithEndTime(0, 0, NULL);
                emptyseg->SetAsSegmentExcludeFromScoring();
                segmentsGroup->AddHypothesis(emptyseg);
				
				if(segmentsGroup->GetNumberOfReferences() == 1)
				{
					if( segmentsGroup->GetReference(0)[0]->GetSpeakerId() == string("inter_segment_gap") )
						emptyHypvsISG = true;
				}
            }
      
            ullint cellnumber = segmentsGroup->GetDifficultyNumber();

            double KBused = (static_cast<double>(cellnumber)/1024.0) * static_cast<double>(sizeof(int));
            double MBused = KBused/1024.0;
            double GBused = MBused/1024.0;
			double TBused = GBused/1024.0;

            char buffer_size[BUFFER_SIZE];

            if(TBused > 1.0)
                sprintf(buffer_size, "%.2f TB", TBused);
            else if(GBused > 1.0)
                sprintf(buffer_size, "%.2f GB", GBused);
            else if(MBused > 1.0)
                sprintf(buffer_size, "%.2f MB", MBused);
            else
                sprintf(buffer_size, "%.2f kB", KBused);
      
            char buffer[BUFFER_SIZE];

			sprintf(buffer, "Align SG %lu [%s] %lu dimensions, Difficulty: %llu (%s) ---> bt=%.3f et=%.3f", 
							static_cast<ulint>(segmentsGroup->GetsID()), 
                            segmentsGroup->GetDifficultyString().c_str(),
                            static_cast<ulint>(segmentsGroup->GetNumberOfHypothesis()+segmentsGroup->GetNumberOfReferences()),
                            cellnumber,
                            buffer_size,
                            segmentsGroup->GetMinTime()/1000.0,
                            segmentsGroup->GetMaxTime()/1000.0);
	   
            LOG_DEBUG(logger, buffer);
	  
            bool ignoreSegs = false;
			bool buseCompArray = false;
            
			if(emptyHypvsISG)
			{
				//ignoreSegs = true;
				sprintf(buffer, "Skip this group of segments (%lu): Inter Segment Gap versus Empty Hyp", static_cast<ulint>( segmentsGroup->GetsID()) );
                LOG_WARN(logger, buffer);
			}
			
			if(!ignoreSegs && (segmentsGroup->isIgnoreInScoring()) )
            {
                ignoreSegs = true;
				sprintf(buffer, "Skip this group of segments (%lu): Ignore this time segments in scoring set into the references", static_cast<ulint>( segmentsGroup->GetsID()) );
                LOG_WARN(logger, buffer);
            }
            
            if(!ignoreSegs && bOnlySG )
            {
            	if(segmentsGroup->GetsID() != OnlySG_ID)
            	{
					ignoreSegs = true;
					sprintf(buffer, "Skip this group of segments (%lu): Only scoring %lu", static_cast<ulint>( segmentsGroup->GetsID()), OnlySG_ID);
					LOG_WARN(logger, buffer);
            	}
            }
			
			if(!ignoreSegs &&  (segmentsGroup->GetNumberOfReferences() > max_spkrOverlaping) )
            {
                ignoreSegs = true;        
				sprintf(buffer, "Skip this group of segments (%lu): nb of reference speaker (%lu) overlaping to high (limit: %lu)", static_cast<ulint>( segmentsGroup->GetsID() ), 
																																	static_cast<ulint>( segmentsGroup->GetNumberOfReferences() ), 
																																	static_cast<ulint>( max_spkrOverlaping) );
                LOG_WARN(logger, buffer);
            }
			
			if(!ignoreSegs &&  (segmentsGroup->GetNumberOfReferences() < min_spkrOverlaping) )
			{
				ignoreSegs = true;        
				sprintf(buffer, "Skip this group of segments (%lu): nb of reference speaker (%lu) overlaping to small (limit: %lu)", static_cast<ulint>(segmentsGroup->GetsID()), 
																																	 static_cast<ulint>(segmentsGroup->GetNumberOfReferences()), 
																																	 static_cast<ulint>(min_spkrOverlaping));
				LOG_WARN(logger, buffer);
			}
		
			if(!ignoreSegs && (bDifficultyLimit) )
			{
				ullint difficulty_nb_gb = static_cast<ullint>( ceil(1024*1024*1024*atof(Properties::GetProperty("recording.nbrdifficultygb").c_str())/sizeof(int)) );
				
				if(cellnumber > difficulty_nb_gb)
				{
					ignoreSegs = true;
					sprintf(buffer, "Skip this group of segments (%lu): the graph size will be too large (%llu [%s]) regarding the difficulty limit (%llu [%.2f GB])", 
									static_cast<ulint>( segmentsGroup->GetsID() ),
									cellnumber,
									buffer_size,
									difficulty_nb_gb,
									atof(Properties::GetProperty("recording.nbrdifficultygb").c_str()));
					LOG_WARN(logger, buffer);
				}
			}
			
			if(!ignoreSegs && (bMinDifficultyLimit) )
			{
				ullint mindifficulty_nb_gb = static_cast<ullint>( ceil(1024*1024*1024*atof(Properties::GetProperty("recording.minnbrdifficultygb").c_str())/sizeof(int)) );
								
				if(cellnumber < mindifficulty_nb_gb)
				{
					ignoreSegs = true;
					sprintf(buffer, "Skip this group of segments (%lu): the graph size will be too small (%llu [%s]) regarding the min difficulty limit (%llu [%.2f GB])", 
									static_cast<ulint>( segmentsGroup->GetsID() ),
									cellnumber,
									buffer_size,
									mindifficulty_nb_gb,
									atof(Properties::GetProperty("recording.minnbrdifficultygb").c_str()));
					LOG_WARN(logger, buffer);
				}
			}
			
			if(!ignoreSegs && ( (cellnumber > max_nb_gb) || (cellnumber > max_nb_2gb) ) )
            {
				bool m_bUseCompression = (string("true").compare(Properties::GetProperty("align.memorycompression")) == 0) || bForceMemoryCompression;
				
				if(!m_bUseCompression)
				{
					ignoreSegs = true;
					sprintf(buffer, "Skip this group of segments (%lu): the graph size will be too large (%llu [%s]) regarding the memory limit (%llu [%.2f GB]) and no compression have been set", 
									static_cast<ulint>( segmentsGroup->GetsID() ),
									cellnumber,
									buffer_size,
									(cellnumber > max_nb_gb) ? max_nb_gb : max_nb_2gb,
									(cellnumber > max_nb_gb) ? atof(Properties::GetProperty("recording.maxnbofgb").c_str()) : 2.0);
					
					LOG_WARN(logger, buffer);
				}
				else
				{
					buseCompArray = true;
					sprintf(buffer, "Using Levenshtein Matrix Compression Algorithm for group of segments (%lu)", static_cast<ulint>( segmentsGroup->GetsID() ) );
					LOG_INFO(logger, buffer);
				}
            }
			
			if(!ignoreSegs && (bForceMemoryCompression) )
			{
				buseCompArray = true;
				sprintf(buffer, "Forcing Levenshtein Matrix Compression Algorihm for group of segments (%lu)", static_cast<ulint>( segmentsGroup->GetsID() ) );
				LOG_INFO(logger, buffer);
			}
			
            //Special case where the SegmentGroup contain a segment time to ignore
            if (!ignoreSegs)
            {
                //cerr << segmentsGroup->ToString() << endl;
				
                Aligner* aligner_instance = aligner["lev"];
                aligner_instance->SetSegments(segmentsGroup, m_pSpeakerMatch, buseCompArray);
                aligner_instance->Align();
                GraphAlignedSegment* gas = aligner_instance->GetResults();
				
				//cerr << gas->ToString() << endl;
				
                sprintf(buffer, "Store %li GAS for hyp: '%s' cost: (%d)", gas->GetNbOfGraphAlignedToken(), hyp_name.c_str(), (dynamic_cast<Levenshtein*>(aligner_instance))->GetCost());
                LOG_DEBUG(logger, buffer);
				
				alignments->AddGraphAlignedSegment(gas, hyp_name, segmentsGroup);
								
				delete gas;
			}
			else //segmentsGroup ignored
			{
				if(logger->isAlignLogON() && !emptyHypvsISG)
					segmentsGroup->LoggingAlignment(m_bGenericAlignment, string("NO"));
			}
			
			if(segmentsGroup)
			{
				if(logger->isAlignLogON() && !emptyHypvsISG)
					segmentsGroup->LoggingAlignment(m_bGenericAlignment, string("SG"));
				
				delete segmentsGroup;
			}
		}
		
        ++i;
    }
}

/**
 * Score the Alignment with the selected scoring system
 */
void Recording::Score()
{
	if(!m_bGenericAlignment)
	{
		Scorer* scorer_instance = scorer["stt"];
    	scorer_instance->Score(alignments, m_pSpeakerMatch);
    }
}

/**
 * Generate the reports based on the scored alignment.
 */
void Recording::GenerateReport(const vector<string> & reportsType)
{
	int outputint = 1;
	
	for (size_t i=0 ; i < reportsType.size() ; ++i)
		if(reportsType[i] == "stdout")
			outputint = 0;
	
	if(!m_bGenericAlignment)
	{	
		for (size_t i=0 ; i < reportsType.size() ; ++i)
			if(reportsType[i] != "stdout")
				reportGenerators[reportsType[i]]->Generate(alignments, outputint);
	}
	else
	{
		pSGMLGenericReportGenerator->Generate(outputint);
	}
}
