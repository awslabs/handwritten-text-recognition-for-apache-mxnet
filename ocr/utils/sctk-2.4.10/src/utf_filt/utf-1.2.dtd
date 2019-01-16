<!SGML "ISO 8879:1986"
-- 
File:	@(#)utf-1.2.dtd	Mar 31, 2004


Authors: Paul Morgovsky and Milan Young
         Linguistic Data Consortium,
	 University of Pennsylvania.

	 Henry S. Thompson,
	 Language Technology Group
	 University of Edinburgh

	 Jon Fiscus
	 Spoken Natural Language Processing Group
	 NIST

Desc: SGML and DTD declaration for the new specifications for the 
      Transcription of Spoken Language.

      Numerous changes were made to enable named entity tagging
      and ASR tagging to co-exist.  This dtd is also annotated
      with comments, which when ran through the appropriate PERL
      script, will result is a DTD without active shortrefs.

Revision History:
      - nothing yet
      - 11/02/99 Added the NOMEX tag.  JGF
      - 03/31/04 Dave Graff and Jon Fiscus conspired to update the DTD for Arabic

Usage: 
        nsgmls utf.dtd filename
--

CHARSET  BASESET  "ISO 646-1983//CHARSET
                   International Reference Version (IRV)//ESC 2/5 4/0"
         DESCSET  0  9 UNUSED   -- NUL,SOH,STX,ETX,ETO,ENQ,ACK,BEL,BS --
                  9  2  9
                  11  2 UNUSED  -- VT,FF --
                  13  1 13   
                  14 18 UNUSED  -- SO,SI,DLE,DC1,DC2 --
                  32 95 32
                  127  1 UNUSED -- del character --
        BASESET   "ISO 646-1983//CHARSET
International Reference Version (IRV)//ESC 2/5 4/0"
        DESCSET   128 32 UNUSED
                  160 1  UNUSED
                  161 65373   161

CAPACITY PUBLIC   "ISO 8879:1986//CAPACITY Reference//EN"
SCOPE    DOCUMENT

SYNTAX   SHUNCHAR CONTROLS 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
                           18 19 20 21 22 23 24 25 26 27 28 29 30 31 127 160
         BASESET  "ISO 646-1983//CHARSET International Reference 
                   Version (IRV)//ESC 2/5 4/0"
         DESCSET  0 128 0
         FUNCTION RE                    13
                  RS                    10
                  SPACE                 32
                  TAB       SEPCHAR     9
         NAMING   LCNMSTRT  ""
                  UCNMSTRT  ""
                  LCNMCHAR  "_-."
                  UCNMCHAR  "_-."
                  NAMECASE  GENERAL     YES
                            ENTITY      NO
         DELIM    GENERAL   SGMLREF 
		  SHORTREF  NONE "*" "+" "^" "%" "@" "," "." "?" "{" "_" "["
				"&#RE;"
				"&#RE;&#RS;"
				"&#RE;&#RS;B"
				"&#RE;B"
				"&#RE;B&#RS;"
				"&#RS;"
				"&#RS;&#RE;"
				"&#RS;&#RE;B"
				"&#RS;B"
				"&#RS;B&#RE;"
				"B"
				"B&#RE;"
				"B&#RE;&#RS;"
				"B&#RS;"
				"B&#RS;&#RE;"

         NAMES    SGMLREF
         QUANTITY SGMLREF
                  NAMELEN   99999999
                  PILEN     24000
                  TAGLEN    99999999
                  TAGLVL    99999999
   
FEATURES MINIMIZE DATATAG   NO
                  OMITTAG   YES
                  RANK      YES
                  SHORTTAG  YES
         LINK     SIMPLE    YES 1000
                  IMPLICIT  YES
                  EXPLICIT  YES 1
         OTHER    CONCUR    NO
                  SUBDOC    YES 99999999
                  FORMAL    YES
APPINFO  NONE>

<!-- This dtd has bee augmented with comments, which, after applying the following
     filter will dissable shortrefs in the DTD.  Thus, text tokens are not parsed by
     sgml but is instead left to the application.

perl -pe  'if (/DELETE TO DISABLE SHORTREF/) { ($_ = "\n") } elsif (/BEGIN COMMENT TO/) { ($_ = "< ! - -\n")=~s/ //g; } elsif (/END COMMENT TO/) { ($_ = "- ->\n")=~s/ //g; } '

  -->

<!DOCTYPE utf [

<!-- Quick Substitution Entities -->

<!-- BEGIN COMMENT TO DISABLE SHORTREF -->
<!ENTITY % textTokens 		"(separator | pName | mispronounced | misspelling | acronym | idiosyncratic | nonlexeme | nonSpeech | period | qmark | comma | contraction | fragment | hyphen | acousticnoise | wtime | #PCDATA )" >
<!-- END COMMENT TO DISABLE SHORTREF -->

<!-- DELETE TO DISABLE SHORTREF
<!ENTITY % textTokens 		"(#PCDATA | contraction | fragment | hyphen | wtime )" >
     DELETE TO DISABLE SHORTREF -->

<!ENTITY % ne_bound         "( b_enamex | e_enamex | b_timex | e_timex | b_numex | e_numex | b_nomex | e_nomex )" >
<!ENTITY % asr_bound        "( b_foreign | e_foreign | b_unclear | e_unclear | b_overlap | e_overlap | b_noscore | e_noscore | b_aside | e_aside )" >



<!ENTITY NONSPEECH		"<nonSpeech>" >
<!ENTITY ACOUSTICNOISE		"<acousticnoise>" >
<!ENTITY SEP			"<separator>" >
<!ENTITY PNAME			"<pName>" >
<!ENTITY MISPRONOUNCED		"<mispronounced>" >
<!ENTITY MISSPELLING		"<misspelling>" >
<!ENTITY ACRONYM		"<acronym>" >
<!ENTITY IDIOSYNCRATIC		"<idiosyncratic>" >
<!ENTITY NONLEXEME		"<nonlexeme>" >
<!ENTITY PERIOD			"<period>" >
<!ENTITY QMARK			"<qmark>" >
<!ENTITY COMMA			"<comma>" >
<!ENTITY IGNORE			"">

<!-- Document Grammar Specifications -->
<!-- Structural definition -->
<!ELEMENT utf    	- - 	( bn_episode_trans | conversation_trans ) >
<!ELEMENT bn_episode_trans
	 	    	- - 	(section | recording_change | background)+ >
<!ELEMENT section    	- - 	(turn | background)* >

<!ELEMENT conversation_trans    	- - 	(turn | background)* >

<!ELEMENT recording_change - O 	EMPTY >
<!ELEMENT turn		- - 	( %textTokens; | time | background | %ne_bound; | %asr_bound; )+ >
<!ELEMENT separator	- O	EMPTY >

<!-- Floating elements -->
<!ELEMENT background	- O	EMPTY >
<!ELEMENT time		  - O 	EMPTY >
<!ELEMENT wtime		  - O 	EMPTY >

<!-- Bouunding tags made explicitly -->
<!ELEMENT b_foreign       - O   EMPTY >
<!ELEMENT b_unclear       - O   EMPTY >
<!ELEMENT b_overlap       - O   EMPTY >
<!ELEMENT b_noscore       - O   EMPTY >
<!ELEMENT b_aside         - O   EMPTY >
<!ELEMENT e_foreign       - O   EMPTY >
<!ELEMENT e_unclear       - O   EMPTY >
<!ELEMENT e_overlap       - O   EMPTY >
<!ELEMENT e_noscore       - O   EMPTY >
<!ELEMENT e_aside         - O   EMPTY >

<!ELEMENT b_enamex        - O   EMPTY >
<!ELEMENT b_timex	  - O   EMPTY >
<!ELEMENT b_numex	  - O   EMPTY >
<!ELEMENT b_nomex	  - O   EMPTY >
<!ELEMENT e_enamex        - O   EMPTY >
<!ELEMENT e_timex	  - O   EMPTY >
<!ELEMENT e_numex	  - O   EMPTY >
<!ELEMENT e_nomex	  - O   EMPTY >

<!-- Applied word tags -->
<!ELEMENT fragment	- O	EMPTY >
<!ELEMENT contraction	- O	EMPTY >

<!-- Shortref elements -->
<!ELEMENT pName		- O	EMPTY >
<!ELEMENT mispronounced	- O	EMPTY >
<!ELEMENT misspelling	- O	EMPTY >
<!ELEMENT acronym	- O	EMPTY >
<!ELEMENT idiosyncratic	- O	EMPTY >
<!ELEMENT nonlexeme	- O	EMPTY >
<!ELEMENT nonSpeech	- O	EMPTY >
<!ELEMENT acousticnoise - O	EMPTY >
<!ELEMENT period	- O	EMPTY >
<!ELEMENT qmark		- O	EMPTY >
<!ELEMENT comma		- O	EMPTY >
<!ELEMENT hyphen	- O	EMPTY >

<!-- Attributes of the Tags -->
<!ATTLIST utf    dtd_version      (utf-1.0|utf-1.1|utf-1.2) #REQUIRED
	 	 audio_filename	  CDATA #REQUIRED
                 language	  CDATA #REQUIRED 
                 scribe	          CDATA #IMPLIED
                 version	  NUMBER #IMPLIED
 	         version_date     CDATA #IMPLIED>

<!ATTLIST bn_episode_trans
                     program	  CDATA #REQUIRED
                     air_date	  CDATA #IMPLIED >

<!ATTLIST conversation_trans
                     recording_date	  CDATA #IMPLIED >

<!ATTLIST section    type      	(report|filler|nontrans) #REQUIRED
                     startTime 	CDATA #REQUIRED
                     endTime   	CDATA #REQUIRED 
                     id	 	CDATA #IMPLIED 
                     topic 	CDATA #IMPLIED >

<!ATTLIST recording_change show CDATA #REQUIRED
                     date       CDATA #REQUIRED
                     sec 	CDATA #REQUIRED >

<!ATTLIST turn       speaker   	CDATA #REQUIRED
                     spkrtype (male|female|child|unknown) #REQUIRED
		     dialect    CDATA #IMPLIED
                     startTime 	CDATA #REQUIRED
                     endTime   	CDATA #REQUIRED 
		     mode 	(planned|spontaneous) #IMPLIED
		     channel 	CDATA #IMPLIED
                     fidelity 	(low|medium|high) #IMPLIED >

<!ATTLIST b_noscore           startTime 	CDATA #REQUIRED
                            endTime 	CDATA #REQUIRED 
                            reason 	CDATA CDATA >

<!ATTLIST b_foreign    language   CDATA #REQUIRED >

<!ATTLIST contraction e_form     CDATA #REQUIRED >

<!ATTLIST b_overlap    startTime 	CDATA #IMPLIED
                     endTime   	CDATA #IMPLIED >

<!ATTLIST time       sec      	CDATA #REQUIRED >

<!ATTLIST wtime      startTime 	CDATA #REQUIRED 
                     endTime	CDATA #REQUIRED 
                     clust     	CDATA #IMPLIED 
                     conf      	CDATA #IMPLIED >

<!ATTLIST background        startTime 	CDATA #REQUIRED 
                            type 	(music|speech|other) #REQUIRED 
                            level 	(off|low|high) #REQUIRED >

<!ATTLIST b_enamex   type	CDATA  #REQUIRED
                     status	(opt)  #IMPLIED
		     alt	CDATA  #IMPLIED >

<!ATTLIST b_timex    type	CDATA  #REQUIRED
        	     status	(opt)  #IMPLIED
		     alt	CDATA  #IMPLIED >

<!ATTLIST b_numex    type	CDATA  #REQUIRED
                     status	(opt)  #IMPLIED
		     alt	CDATA  #IMPLIED >

<!ATTLIST b_nomex    type	CDATA  #REQUIRED
                     status	(opt)  #IMPLIED
		     min	CDATA  #IMPLIED >


<!-- Short Refference Mappings -->

<!-- BEGIN COMMENT TO DISABLE SHORTREF -->

<!SHORTREF TURN		'.'	PERIOD
			'?'	QMARK
			','	COMMA
			'+'	MISPRONOUNCED
			'@'	MISSPELLING
			'_'	ACRONYM
			'^'	PNAME
			'*'	IDIOSYNCRATIC
			'%'	NONLEXEME
			'{'	NONSPEECH
			'['	ACOUSTICNOISE
			'&#RS;B&#RE;'	IGNORE
			'&#RS;&#RE;'	IGNORE
			'&#RE;&#RS;'	SEP
			'&#RE;&#RS;B'	SEP
			'&#RE;'		SEP
			'&#RE;B&#RS;'	SEP
			'&#RE;B'	SEP
			'&#RS;&#RE;B'	SEP
			'&#RS;'		SEP
			'&#RS;B'	SEP
			'B&#RE;&#RS;'	SEP
			'B&#RE;'	SEP
			'B&#RS;&#RE;'	SEP
			'B&#RS;'	SEP
                        'B'		SEP   >

<!USEMAP TURN turn >

<!-- END COMMENT TO DISABLE SHORTREF -->

]>
