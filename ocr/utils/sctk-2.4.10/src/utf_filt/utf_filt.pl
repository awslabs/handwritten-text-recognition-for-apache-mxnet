#!/usr/bin/perl -w

########################################################
# File: utf_filt_v55.pl
#
# History:
#
#  Version 1-24 Debugging versions non release
#
#  Version 25 Released 8/14/98
#          This version converts the output of a nsgmls to various formats. 
#          These formats include:
#          1) stm without NE tags for ASR processing,
#          2) stm with NE tags for NE processing
#          3) ctm with NE tags for NE processing (word time is obtained 
#             from averaging number of words in a turn),
#          4) ctm with NE tags for NE processing (word time is obtained 
#             from word time tag).
#          Several issues are still unresolved in this version: 
#          1) period in Ms.
#          2) turn end time is marked when music ends not when speaker
#             stops talking
#
#  Version 26
#          Added stm_sm_u format
#          Fixed usage statement and change <format> values to be more
#          readable
#
#  Version 27
#          Added the nsgmls conversion to be done internally so that
#          the input to the filter is the sgml transcript not the output
#          of the nsgmls program
#          sgml --> filter ---> output in specified format
# 
#  Version 28
#          Added the -o flag to indicate if overlap is to be processed
#          or ignored
#          Added the -e option for user to specify the dtd file to be used
#
#  Version 29
#          Accomondated the latest version of dtd. Attributes of the old
#          episode tag is divided between a utf tag and episode tag
#
#  Version 30
#          Modified usage statement. Inproved command line interface.
#          Corrected the output NE tags to include double quotes.
#
#  Version 31-32
#          Fixed the warning messaged generated on implied data elements
#          in turns and section.
#          Added handlers for 'conversation' and 'b_aside'/'e_aside'
#          Modified the routine to parenthize words so that sgml tags are handled
#
#  Version 33-34
#          Added the CTM output format
#          Modified the tokenization of stm2ctm to be more robust.
#          
#  Version 35
#          modified the IECTM output format to make a single token out of 
#          all named entity tags so that sclite doesn't divide them
#
#  Version 36
#          Added the option -k to keep the text in the excluded text regions
#             (rather than 'IGNORE_TIME_...'
#          Modified the way IECTMs are made when the -w tag is not used in
#             that overlap tags with times are used to divide turns so that
#             the inferred word times next to overlapped regions are correct.
#  Version 37 
#          Fixed the handling if possessive acronyms.  The pattern match was
#          case sensitive
#
#  Version 38
#          Modified the way overlaps are diagnosed so that identical numbers
#          are tolerated.
#          Changed the sex label for inter_segment_gaps to be blank.
#          Removed the child designation for a possoble label
#          Fixed the contraction parser to handle upper case letters, and use
#             either => or = in the eform rule
#
#  Version 39
#          Incorporated changes from Aaron Douthat that handled back-to-back NE tags.
#
#  Version 40
#          Fixed an interaction with acronyms and contractions.
#          ASR and PUNCTUATION tags now function as WORD separators.
#
#  Version 41
#          Fixed handling of optionally deletable text in the ctm output.
#             - all opt-del tags are removed, thus making them normal lexical items
#
#  Version 42
#          Fixed how fragments are handled if they are ne tagged
#
#  Version 43
#          Fixed the interaction between netags and the -w switch
#
#  Version 44
#          Fixed a problem with a local scoping bug that showed up with the
#          newest version of perl.
#
#  Version 45
#          Added the sex designation "unknown" the the LUR labels
#
#  Version 46
#          Added code to usure the asr attribute stack orders the start and 
#          end times appropriately.  It's clear that codes needs rewritten.  It
#          was done before JGF knew about hash of hashes
#
#  Version 47
#          Added -c flag to disable focus conditions
#
#  Version 48
#          Fixed the unknown gender tag to match for the lur conditions.
#
#  Version 49
#          Corrected the handling of the process_overlap flag
#
#          Fixed a problem in ie_split().  The [^\\s<>] expressiuon should
#          not have been escaped for new versions of perl.
#
#  Version 50
#         1. Fixed a problem with splitting uncleare tokens.  There was never a space put between them.
#         2. Fixed problem with plural ancronyms
#         3. Fixed problem with unclear spoken letters.
#         4. Fixed problem with pre-token fragments (-ord)
#
#  Version 51
#         Fixed problem with reading UTF-8 character
#         Use Getopt::Long instead of getopt
#
#  Version 52
#         Added the option to not expand contractions
#
#  Version 53
#         Added code to warn IF there are non-posessive contractions that are not annotated.
#
#  Version 54
#         Added code to not translate non-lexemes to %hesitation.
#         Added code to not translate non-lexemes to %hesitation.
#
#  Version 55          
#         Don't lowercase the speaker and channel of a turn
#
#  Version 56          
#         Added support for onsgmls
#
########################################################


########################################################
# constant definitions
########################################################

# fields for items in a stm turn
#
$FILENAME_KEY = "FILENAME";
$NUM_CHAN_KEY = "NUM_CHAN";
$SPEAKER_KEY = "SPEAKER";
$SPKRTYPE_KEY = "SPKRTYPE";
$TURN_STIME_KEY = "STARTTIME";
$TURN_ETIME_KEY = "ENDTIME";
$TURN_CHANNEL_KEY = "CHANNEL";
$FOCUS_COND_KEY = "FOCUS_COND";
$TEXT_KEY = "TEXT";
$SECTION_STIME_KEY = "SECTION_STARTTIME";
$SECTION_ETIME_KEY = "SECTION_ENDTIME";
$SECTION_TYPE_KEY = "SECTION_TYPE";
$SECTION_ID_KEY = "SECTION_ID";

# fields for determining the focus condition
#
$DIALECT_KEY = "DIALECT";
$MODE_KEY = "MODE";
$FIDELITY_KEY = "FIDELITY";
$MUSIC_KEY = "MUSIC";
$SPEECH_KEY = "SPEECH";
$OTHER_KEY = "OTHER";
$LEVEL_KEY = "LEVEL";

# noise and fidelity levels
#
$HIGH = "high";
$LOW = "low";
$ON = "on";
$OFF = "off";

# modes
#
$PLANNED = "planned";
$SPONTANEOUS = "spontaneous";

# dialect types
#
$NATIVE = "native";
#$NONNATIVE = "nonnative";

# constant strings in lower case. Must convert to lower case before
# comparison
#
$IGNORE_TIME_SEGMENT_IN_SCORING = "ignore_time_segment_in_scoring";
$EXCLUDED_REGION = "excluded_region";
$INTER_SEGMENT_GAP = "inter_segment_gap";
$UNKNOWN = "unk";

# misc. constants
#
$OVERLAP = 1;
$NO_OVERLAP = -2;
$THRESHOLD = 1;

# output format
#
$STM = "STM";
$IESTM = "IESTM";
$IECTM = "IECTM";
$CTM = "CTM";
$IESM = "IESM";
$SM = "SM";

# debug level
#
$DEBUG_NONE = 0;
$DEBUG_BRIEF = 1;
$DEBUG_DETAIL = 2;
#$DEBUG_FULL = 4;

# define usage statement
#
$USAGE = "\n\n$0 [-wkph] -f <format> -e <dtd_file> -i <input file> -o <output file>\n\n".
"Desc:  This script reformats an input UTF transcript file to various formats\n".
"       The script uses the SGML DTD, and nsgmls to read the <input_file> and write\n".
"       the file reformatted to <format> into <output_file>\n".
"\n".
"Required arguments:\n".
"   -f <format> indicates the format of the output\n".
"      where <format> can be one of the following:\n".
"         IESTM: stm format with IE tags\n".
"         STM:   stm format for ASR--default\n".
"         CTM:   ctm format\n".
"         IECTM: ctm format with IE tags\n".
"         IESM:  Story marked STM format with IE tags\n".
"         SM:    Story marked STM format for ASR\n".
"   -e <dtd file> is the dtd file\n".
"   -i <input file>  is the sgml transcript\n".
"   -o <output file> is the output of this script\n".
"Optional Arguments\n".
"   -d <debug level> prints out debug information\n".
"   -c flag to disable focus condition tagging\n".
"   -h flag to display this message\n".
"   -k flag to keep text within excluded regions in the output stream.  only\n".
"      This option valid only if -p is specified\n".
"   -p flag to process overlap instead of ignoring it\n".
"   -w flag to use word time tag instead of average word duration for CTM format\n".
"   -t do not translate contractions to their expanded form\n".
"   -n do not delete acousticnoise and nonspeech events.  Do not translate nonlexemes\n".
"      to the %hesitation symbol\n".
"   -s 'prog' set the SGML parsing program name to 'prog'".
"\n";

@TextAttribs = ('ACOUSTICNOISE', 'NONSPEECH', 'ACRONYM', 'NONLEXEME', 'MISPRONOUNCED', 'MISSPELLING', 'PNAME', 'IDIOSYNCRATIC');

@NE_beg_tags = ('B_ENAMEX', 'B_TIMEX', 'B_NUMEX');
@NE_end_tags = ('E_ENAMEX', 'E_TIMEX', 'E_NUMEX');

@ASR_beg_tags = ('B_FOREIGN', 'B_UNCLEAR', 'B_OVERLAP', 'B_NOSCORE', 'B_ASIDE');
@ASR_end_tags = ('E_FOREIGN', 'E_UNCLEAR', 'E_OVERLAP', 'E_NOSCORE', 'E_ASIDE');

@PUNCT_tags = ('PERIOD', 'QMARK', 'COMMA');

# generate regular expressions once for speed
#
$TextAttribs_re = join("|",@TextAttribs);
$NE_beg_tags_re = join("|",@NE_beg_tags);
$NE_end_tags_re = join("|",@NE_end_tags);
$ASR_beg_tags_re = join("|",@ASR_beg_tags);
$ASR_end_tags_re = join("|",@ASR_end_tags);
$PUNCT_tags_re = join("|",@PUNCT_tags);

########################################################
# global variables
########################################################

$conformance = 0;
$incomment = 0;
%attr_stack = ();
%ne_attr_stack = ();
%asr_attr_stack = ();
%bkgrnd_attr_stack = ();
%wtime_attr_stack = ();
@sorted_section_stack = ();
@section_stack = ();
%section_attr = ();
%turn_attr = ();
@turn_stack = ();
%turn_elems = ();
@text_elem = ();
$text = "";
$text_flag = 0;
$filename = "";
$music_level = $OFF;  # default
$speech_level = $OFF; # default
$other_level = $OFF;  # default
$num_chan = 1;        # default
$in_turn = 0;         # 0 not in the middle of a turn, 1 otherwise
$kept_turn = 1;       # 0 turn is not kept, 1 otherwise
$format = $STM;       # default
$use_wtag = 0;        # flag to use average word duration
$process_overlap = 0; # flag to ignore overlap
$input = "";
$output = "";
$dtd_file = "/home1/le/su_98/dtd/utf-1.0.dtd";
$debug_level = $DEBUG_NONE;     # default no debugging info
$display_help = 0;              # default no display of help file
$focus_condition_enabled = 1;   # maek dat with focus conditions
$inter_segment_gap_enabled = 1; #enable/disable marking inter-segment gaps
$DBoff = "";
$keep_exclude_text = 0;
$translateContractions = 1;
$translateNonlexemes = 1;
$deleteNonspeech = 1;
$deleteAcousticnoise = 1;
$NSGMLS = "nsgmls";

########################################################
# include extended perl libraries
########################################################

# use the SGMLS class
#
use SGMLS;

########################################################
# main program
########################################################

#require "getopts.pl";
#&Getopts('pwkchd:f:e:i:o:');
use Getopt::Long;
my $ret = GetOptions ("p", "w", "k", "c", "h", "d:i", "f=s", "e=s", "i=s", "o=s", "t", "n", "s=s");

$use_wtag = $opt_w if (defined($opt_w));
$process_overlap = $opt_p if (defined($opt_p));
$display_help = $opt_h if (defined($opt_h));
$output = $opt_o if (defined($opt_o));
$debug_level = $opt_d if (defined($opt_d));
$focus_condition_enabled = (! $opt_c) if (defined($opt_c));
$translateContractions = (!$opt_t) if (defined($opt_t));
if (defined($opt_n)){
    $translateNonlexemes = (!$opt_n);
    $deleteNonspeech = (!$opt_n);
    $deleteAcousticnoise = (!$opt_n);
}
if (defined($opt_k)){
    $keep_exclude_text = $opt_k;
    die "${USAGE}Error: option -k only valid with option -p"
	if (!defined($opt_p));
}

if (defined($opt_f)){
    ($format = $opt_f) =~ tr/a-z/A-Z/;
    die "${USAGE}Error: Unknown format '$format'" if ($format !~ /^(IESTM|STM|IECTM|CTM|IESM|SM)$/);
}
if (defined($opt_e)){
    die "Error: DTD file '$opt_e' is not a readable file" if (! -r $opt_e);
    $dtd_file = $opt_e;
}
if (defined($opt_i)){
    die "Error: Input file '$opt_i' is not a readable file" if (! -r $opt_i);
    $input = $opt_i;
}
$NSGMLS = $opt_s if (defined($opt_s));

# check command line arguments
#
die "${USAGE}Error: Output format required" if (!defined($opt_f));
die "${USAGE}Error: Output file required" if (!defined($opt_o));
die "${USAGE}Error: Input UTF file required" if (!defined($opt_i));
die "${USAGE}Error: SGML DTD file required" if (!defined($opt_e));

die ("$USAGE") if ($display_help) ;
die ("ERROR: use -w only with IECTM format\n") 
    if ($use_wtag == 1 && $format !~ /^($CTM|$IECTM)$/);


# open files
#
open(IN, "cat $dtd_file $input | $NSGMLS |") or 
    die("Unable to open nsgmls parsed data $input");
binmode IN;

open (OUT, ">$output") or die ("Unable to open $output for writing\n");
binmode OUT;

# allocate an object of SGMLS class 
#
$this_parse = new SGMLS(IN);

while ($this_event = $this_parse->next_event) {
    local $type = $this_event->type;
    local $data = $this_event->data;

    print "${DBoff}Main-WHILE  \$text='$text'  \@text_elem = '".join(" ",@text_elem)."'\n"
	if ($debug_level > $DEBUG_DETAIL) ;

   SWITCH: {
      
      $type eq 'start_element' && do {
	  if ($debug_level > $DEBUG_BRIEF) {
	      print "${DBoff}start ".$data->name."\n"; ${DBoff} .= "  ";
          }

	  if ($data->name eq "UTF"){
	      &proc_beg_utf($data);
	  }
	  
	  elsif ($data->name eq "BN_EPISODE_TRANS"){
	      &proc_beg_episode($data);
	  }

	  elsif ($data->name eq "CONVERSATION_TRANS"){
	      &proc_beg_converse($data);
	  }

	  elsif ($data->name eq "SECTION"){
	      &proc_beg_section($data);
	  }

	  elsif ($data->name eq "TURN"){
	      &proc_beg_turn($data);
	  }

	  elsif ($data->name eq "SEPARATOR"){
	      &proc_beg_separator($data);
	  }

	  elsif ($data->name eq "WTIME"){
	      &proc_beg_wtime($data);
	  }

	  elsif ($data->name eq "TIME"){
	      ; ## ignore for now
	  }

	  elsif ($data->name eq "COMMENT"){
	      $incomment = 1;
	  }

	  elsif ($data->name eq "BACKGROUND"){
	      &proc_beg_bkgrnd($data);
	  }

	  elsif ($data->name eq "CONTRACTION"){
	      &proc_beg_cont($data);
	  }

	  elsif ($data->name eq "FRAGMENT"){
	      &proc_beg_frag($data);
	  }

	  elsif ($data->name eq "HYPHEN"){
	      &proc_beg_hyphen($data);
	  }

	  elsif ($data->name =~ /($PUNCT_tags_re)/){
	      ### Pre-check, ASR tags ARE word separators
	      &proc_beg_separator("") if ($text ne "");
	  }

	  elsif ($data->name =~ /($TextAttribs_re)/){
	      &proc_beg_TextAttrib($1);
	  }

	  elsif ($data->name =~ /($NE_beg_tags_re)/){
	      &proc_beg_NE_tag($1,$data);
	  }

	  elsif ($data->name =~ /($NE_end_tags_re)/){
	      &proc_end_NE_tag($1,$data);
	  }

	  elsif ($data->name =~ /($ASR_beg_tags_re)/){
	      &proc_beg_ASR_tag($1,$data);
	  }

	  elsif ($data->name =~ /($ASR_end_tags_re)/){
	      &proc_end_ASR_tag($1,$data);
	  }

	  else {
	      die "undefined start tag ".$data->name;
	  }

	  last SWITCH;
      };

      $type eq 'end_element' && do {
	  if ($debug_level > $DEBUG_BRIEF) {
	      ${DBoff} =~ s/..$//; print "${DBoff}end ".$data->name."\n";
          }

	  if ($data->name eq "UTF"){
	      &proc_end_utf($data);
	  }
	  
	  elsif ($data->name eq "BN_EPISODE_TRANS"){
	      &proc_end_episode();
	  }

	  elsif ($data->name eq "CONVERSATION_TRANS"){
	      &proc_end_converse();
	  }

	  elsif ($data->name eq "SECTION"){
	      &proc_end_section();
	  }

	  elsif ($data->name eq "TURN"){
	      &proc_end_turn();
	  }

	  elsif ($data->name eq "SEPARATOR"){
	      ; ## ignore for now
	  }

	  elsif ($data->name eq "WTIME"){
	      ; ## ignore for now
	  }

	  elsif ($data->name eq "TIME"){
	      ; ## ignore for now
	  }

	  elsif ($data->name eq "COMMENT"){
	      $incomment = 0;
	  }

	  elsif ($data->name eq "BACKGROUND"){
	      if ($debug_level > $DEBUG_DETAIL){
         	  print "${DBoff}  After BKG music=$bkgrnd_attr_stack{$MUSIC_KEY} ".
                      "speech=$bkgrnd_attr_stack{$SPEECH_KEY} ".
		      "other=$bkgrnd_attr_stack{$OTHER_KEY}\n"
	      }
	      ; ## ignore for now
	  }

	  elsif ($data->name eq "CONTRACTION"){
	      ; ## ignore for now
	  }

	  elsif ($data->name eq "FRAGMENT"){
	      ; ## ignore for now
	  }

	  elsif ($data->name eq "HYPHEN"){
	      ; ## ignore for now
	  }

	  elsif ($data->name =~ /($PUNCT_tags_re)/){
	      ; ## ignore for now
	  }

	  elsif ($data->name =~ /($TextAttribs_re)/){
	      ; ## ignore for now
	  }

	  elsif ($data->name =~ /($NE_beg_tags_re)/){
	      ; ## ignore for now
	  }

	  elsif ($data->name =~ /($NE_end_tags_re)/){
	      ; ## ignore for now
	  }

	  elsif ($data->name =~ /($ASR_beg_tags_re)/){
	      ; ## ignore for now
	  }

	  elsif ($data->name =~ /($ASR_end_tags_re)/){
	      ; ## ignore for now
	  }

	  else {
	      die "undefined end tag ".$data->name;
	  }

	  last SWITCH;
      };

      $type eq 'cdata' && do {
	  print "${DBoff}  CDATA ".$data."\n"
	      if ($debug_level > $DEBUG_BRIEF);

	  if ($incomment){
	      ; #### Skipping Comment 
	  }

	  else { 
	      &proc_cdata($data);
	  }

	  last SWITCH;
      };

      $type eq 're' && do {
	  if ($incomment){
	      ;  
	  }

	  last SWITCH;
      };

      $type eq 'conforming' && do {
	  $conformance = 1;
	  last SWITCH;
      };

      die "Undefined event occurred '$type' in file";
  }
}

if ($conformance == 0) {
    die("Transcript file '$input' does not conform to DTD '$dtd_file'");
}

# close input and output files
#
close (IN);
close (OUT);

# exit normally
#
exit 0;

########################################################
# subroutines
########################################################

sub proc_beg_utf{
    local($data) = @_;
    local($key);
    
    for $key ($data->attribute_names) {

	if ($debug_level > $DEBUG_DETAIL) {
	    print "${DBoff}Attribute: $key=" . (($data->attribute($key)->is_implied) ? "" : $data->attribute($key)->value) . "\n";
	}

	# extract the filename for one of the fields in an stm turn
	#
	if ($key eq "AUDIO_FILENAME") {
	    $filename = $data->attribute($key)->value;
	    $filename =~ s/(\S+)\.\S+/$1/;

	    # if value of FILENAME key is UNK, use the input filename
	    #
	    if (lc($filename) eq $UNKNOWN) {
		$filename = $input;
	    }
	}
    }

    # initialize background conditions
    #
    $bkgrnd_attr_stack{$MUSIC_KEY} = $OFF;
    $bkgrnd_attr_stack{$SPEECH_KEY} = $OFF;
    $bkgrnd_attr_stack{$OTHER_KEY} = $OFF;
    $bkgrnd_attr_stack{$TURN_STIME_KEY} = 0;
    $music_level = $OFF;
    $speech_level = $OFF;
    $other_level = $OFF;

    # reset a variable that holds all the turns (of all the sections) 
    # in an episode
    #
    @section_stack = ();
}

sub proc_end_utf{
    if ($debug_level > $DEBUG_BRIEF) {
	print "${DBoff}End of Utf -----------\n";
    }

    # sort the turn by start time
    #
    @sorted_section_stack = sort proc_bystarttime @section_stack;

    # eliminate turns whose text field is empty
    #
    # &proc_del_turn(*sorted_section_stack);

    # fill in the gap between turns
    #
    &proc_fill_gap(*sorted_section_stack) if ($inter_segment_gap_enabled);

    # print the content of the stack
    #
    &proc_print_stack(OUT, $format, *sorted_section_stack);
}

sub proc_beg_episode{

    local($data) = @_;
    local($key);

    for $key ($data->attribute_names) {

	if ($debug_level > $DEBUG_DETAIL) {
	    print "${DBoff}Attribute: $key=" . $data->attribute($key)->value . "\n";
	}
    }
}

sub proc_end_episode{
    ;
}

sub proc_beg_section{

    local($data) = @_;
    local $key;
    local $temp_key;

    %section_attr = ();

    for $key ($data->attribute_names) {
	if ($debug_level > $DEBUG_DETAIL) {	
	    print "${DBoff}Attribute: $key=" . (($data->attribute($key)->is_implied) ? "" : $data->attribute($key)->value) . "\n";
	}
	
	# add the "SECTION_" in front of the original key to differentiate
	# key in a turn
	#
	$temp_key = "SECTION_" . $key;

	$section_attr{$temp_key} = ($data->attribute($key)->is_implied) ? "" : $data->attribute($key)->value;

    }

    # reset a variable that holds all the turns in a section
    #
    @turn_stack = ();
}

sub proc_end_section{

    local %tmp_turn = ();

    # eliminate turns whose text field is empty
    #
    &proc_del_turn(*turn_stack);

    # check for overlap and zap it
    #
    if ($process_overlap) {
	&proc_elim_overlap(*turn_stack);
    }

    # save all the turns in a section in another variable
    #
    for ($i = 0; $i <= $#turn_stack; $i++) {
	%tmp_turn = ();
	for $key (keys %{$turn_stack[$i]}) {
	    $tmp_turn{$key} = $turn_stack[$i]{$key};
	}
	push @section_stack, { % tmp_turn };
    }
}

sub proc_beg_converse{

    local($data) = @_;
    local($key);

    ### Disable focus conditions
    $focus_condition_enabled = 0;
    $inter_segment_gap_enabled = 0;

    %section_attr = ();

    for $key ($data->attribute_names) {
	if ($debug_level > $DEBUG_DETAIL) {	
	    print "${DBoff}Attribute: $key=" . (($data->attribute($key)->is_implied) ? "" : $data->attribute($key)->value) . "\n";
	}
	
	# add the "CONVERSATION_" in front of the original key to differentiate
	# key in a turn
	#
	$temp_key = "CONVERSATION_" . $key;

	$section_attr{$temp_key} = ($data->attribute($key)->is_implied) ? "" : $data->attribute($key)->value;

    }

    # reset a variable that holds all the turns in a section
    #
    @turn_stack = ();
}

sub proc_end_converse{

    local %tmp_turn = ();

    # check for overlap and zap it
    #
    if ($process_overlap) {
	&proc_elim_overlap(*turn_stack);
    }

    # save all the turns in a section in another variable
    #
    for ($i = 0; $i <= $#turn_stack; $i++) {
	%tmp_turn = ();
	for $key (keys %{$turn_stack[$i]}) {
	    $tmp_turn{$key} = $turn_stack[$i]{$key};
	}
	push @section_stack, { % tmp_turn };
    }
}

sub proc_beg_turn{

    local($data) = @_;
    local ($key);

    for $key ($data->attribute_names) {
	if (! $data->attribute($key)->is_implied){
	    if ($debug_level > $DEBUG_DETAIL) {
		print "${DBoff}Attribute: $key=" . (($data->attribute($key)->is_implied) ? "" : $data->attribute($key)->value) . "\n";
	    }
	}
	else {
	    if ($key eq "CHANNEL") {
		$num_chan = 1;
	    }
	}
    }

    # reset all the variables that hold info for a turn
    #
    %turn_attr = ();
    %turn_elems = ();
    @text_elem = ();

    # keep the attributes of a turn
    #
    foreach $key (keys %section_attr) {
	$turn_elems{$key} = $section_attr{$key};
    }
    foreach $key($data->attribute_names){
	$turn_attr{$key} = ($data->attribute($key)->is_implied) ? "" :
	    lc($data->attribute($key)->value);
	$turn_elems{$key} = $turn_attr{$key};
    }
    $turn_elems{$FILENAME_KEY} = $filename;
    $turn_elems{$TURN_CHANNEL_KEY} = "1" 
	if ($turn_elems{$TURN_CHANNEL_KEY} eq "");
    $turn_attr{$TURN_CHANNEL_KEY} = "1"
	if ($turn_attr{$TURN_CHANNEL_KEY} eq "");
    $turn_elems{$NUM_CHAN_KEY} = ($turn_elems{$TURN_CHANNEL_KEY} eq "") ?
	$num_chan : $turn_elems{$TURN_CHANNEL_KEY};
    ### Ignore children for lur report
    if ($turn_elems{$SPKRTYPE_KEY} =~ /child/i){
	$turn_elems{$SPKRTYPE_KEY} = "";
	$turn_attr{$SPKRTYPE_KEY} = "";
    }    
    $turn_elems{$FOCUS_COND_KEY} = &proc_get_focus_cond(
                      $turn_attr{$DIALECT_KEY}, 
		      $turn_attr{$FIDELITY_KEY}, 
                      $turn_attr{$MODE_KEY}, 
                      $bkgrnd_attr_stack{$MUSIC_KEY}, 
                      $bkgrnd_attr_stack{$SPEECH_KEY},
                      $bkgrnd_attr_stack{$OTHER_KEY}); 
    $turn_elems{$TEXT_KEY} = "";

    # we are in a middle of a turn
    #
    $in_turn = 1;

    # keep the turn unless otherwise
    #
    $kept_turn = 1;
}

sub proc_end_turn{

    local $i;

    $turn_elems{$TEXT_KEY} = "";

    # if cdata is preceded by the separator tag then we can put a space
    # after it, else do not but a space after.  However, an acronym
    # is an exception to this rule.  No separator precedes cdata but
    # we must put a space after each acronym.
    # For example, B. B. C.'s not B.B.C.'s. So if we put a space after
    # each acronym then when we get to 's the space after C. will show.
    # There is no way we can tell when the last acronym is read as we do
    # not do any lookahead but process each event as it is encountered.
    # So we output B.B.C.'s and go back and postprocess it to be B. B. C.'s
    #
    &proc_preproc_text (*text_elem);

    for ($i = 0; $i <= $#text_elem; $i++) {
	$turn_elems{$TEXT_KEY} .= $text_elem[$i] . " ";
    }

    if ($kept_turn) {
	push @turn_stack, { %turn_elems };
    }

    # we are not in a turn
    #
    $in_turn = 0;
}

sub proc_beg_separator{

    local($data) = @_;

    local $kept_text = 1;

    if ($text eq "") {
	$kept_text = 0;
    }

    if ($debug_level > $DEBUG_DETAIL) {
	print "${DBoff}   ( ";
    }

    # Look for un-annotated contractions
    if ($translateContractions && ! defined($attr_stack{"CONTRACTION"})){
	print "Warning: Possible non-annotated contraction \"$text\"\n" if ($text =~ /\w'\w+/ && $text !~ /\w's$/);
    }
    
    for $key(keys %attr_stack){

	if ($debug_level > $DEBUG_DETAIL) {
	    print "$key=".$attr_stack{$key} . " ";
	}

	# do not keep nonspeech 
	#
	if ($key eq "NONSPEECH") {	    
	    if ($deleteNonspeech){
		$kept_text = 0;		
	    } else {
		$text = "{".$text;
	    }
	} elsif ($key eq "ACOUSTICNOISE") {	    
	    if ($deleteAcousticnoise){
		$kept_text = 0;		
	    } else {
		$text = "\[".$text;
	    }
	}

	# output (%hesitation) for nonlexeme
	#
	elsif ($key eq "NONLEXEME") {
	    if ($translateNonlexemes){
		### This perserves any named entity tag appended to a hesitation
		$text =~ s/^[^<>]+/(%hesitation)/;
		$text =~ s/[^<>\(\)]+$/(%hesitation)/;
	    } else {
		$text = "%".$text;
	    }
	}

	# change contraction to full word
	#
	elsif ($key eq "CONTRACTION") {
	    if ($translateContractions){
		local $tmp_str = $attr_stack{$key};
		$tmp_str =~ s/\s//g;
		$tmp_str =~ s/\S+=(\[.*)/$1/g;
		$tmp_str =~ s/\]\[/\] \[/g;
		local @tmp_data1 = split (/\s+/, $tmp_str);
		local $word;
		local $tag;
		local $val;
		foreach $word (@tmp_data1) {
		    $word =~ s/\[(\S+)=>*(.*)\]/$1 $2/g;
		    ($tag, $val) = split (/\s/, $word);
		    $text =~ s/(.*)$tag(\.?)(.*)/$1 $val$2 $3/i;
		}
		$text =~ s/\s$//;
	    }
	}
    }

    if ($debug_level > $DEBUG_DETAIL) {    
        print "NE: ";
	foreach $key(sort(keys %ne_attr_stack)){
	    print "$key=".$ne_attr_stack{$key}." ";
	}
	
        print "ASR: ";
	foreach $key(sort(keys %asr_attr_stack)){
	    print "$key=".$asr_attr_stack{$key}." ";
	}
	
        print "BG: ";
	foreach $key(sort(keys %bkgrnd_attr_stack)){
	    print "$key=".$bkgrnd_attr_stack{$key}. " ";
	}
	
	print "\$text=$text kept_text=$kept_text)\n";
    }
    if ($kept_text) {
	if ($parenthesize_text) {

	    # check if there is a () around the word(s) already.
	    # If so, take () off, split if multiple words, and put ()
	    # around each word.
	    #
	    if ($text =~ /\(([^\)]*)\)/) {
		$text = $1;
	    }

	    ### JGF: parenthisis fix
	    local $newtext = "";
	    local $jword = "";
	    foreach $jword(ie_split ($text)){
		$jword =~ s/^([^<> ]+)$/($1)/;
		$newtext .= " ".$jword;
	    }
  	    $text = $newtext;
	}
	if ($use_wtag) {
	    push (@text_elem, "STIME=$wtime_attr_stack{$TURN_STIME_KEY}");
	    push (@text_elem, "ETIME=$wtime_attr_stack{$TURN_ETIME_KEY}");
	}
	push (@text_elem, $text);
    }
    
    # reset the text within separator tag
    #
    $text = "";

    # reset attribute stack
    #
    %attr_stack = ();

    # reset wtime attribute stack
    #
    %wtime_attr_stack = ();

    print "${DBoff}  final text arr '".join(" ",@text_elem)."'\n"
	if ($debug_level > $DEBUG_DETAIL);

}

sub proc_beg_wtime{

    local($data) = @_;
    local ($key);

    foreach $key($data->attribute_names){
	if (! $data->attribute($key)->is_implied){
	    $wtime_attr_stack{$key} = $data->attribute($key)->value;
	    if ($debug_level > $DEBUG_DETAIL) {
		print "$key = $wtime_attr_stack{$key} ";
	    }
	}
    }
    if ($debug_level > $DEBUG_DETAIL) {
	print "\n";
    }
}

sub proc_beg_TextAttrib{

    local($attrib) = @_;
    if ($debug_level > $DEBUG_BRIEF) {
	print "   $attrib\n";
    }

    # reset the attribute stack
    #
    $attr_stack{$attrib} = "";
}

sub proc_cdata{
    local($data) = @_;
    local($key);

    # add the text
    #
    $text .= $data;

    $text_flag = 1;

    for $key(keys %attr_stack){

	# add ".@" after each letter of an acronym.  Will convert @ to blank
	# later
	#
	if ($key eq "ACRONYM") {
	    delete $attr_stack{$key};

	    # can't seem to make this work
	    #
#	    if ($text =~ /(.+)('ve|'s|'re|'m|n't|'ll|'d)/) {
#		$text =~ s/(.+)('ve|'s|'re|'m|n't|'ll|'d)/$1\.('ve|'s|'re|'m|n't|'ll|'d)/;
#	    }
#	    else {
#		$text .= ".@";
#	    }
	    if ($text =~ /(.+)'s/i) {
		$text =~ s/(.+)('s)/$1\.$2/i;
	    } elsif ($text =~ /(.+)s/i) {
		$text =~ s/(.+)(s)/$1\.$2/i;
	    }
	    else {
		$text .= ".@";
	    }
        } elsif ($key eq "FRAGMENT"){
	    $text = "(-$text)";
	}
    }
}

sub proc_beg_frag{
    local($data) = @_;
    local($newtext) = "";

    &proc_beg_TextAttrib($data->name);
    foreach $word(ie_split($text)) {
	
	$newtext .= ($word =~ /^<.*>$/) ? $word : "($word-)";
    }
    $text = $newtext;
}

sub proc_beg_hyphen{
    local($data) = @_;

    # substitute hyphen for space.  However, it's the same doing nothing
    # and treat the next cdata as a new word
    #
    if ($text !~ /\s$/) { $text .= " "; }
    $text =~ s:(<[^/<>]+>)\s+$:$1:;
}

sub proc_beg_NE_tag{
    local($tag,$data) = @_;
    local($norm_tag) = $tag;
    $norm_tag =~ s/B_//;

    $ne_attr_stack{$norm_tag} = "<$norm_tag";
    foreach $key($data->attribute_names){
	if (! $data->attribute($key)->is_implied){
	    $ne_attr_stack{$norm_tag} .= " $key=\"".$data->attribute($key)->value."\"";
	}
    }
    $ne_attr_stack{$norm_tag} .= ">";
    if ($format eq $IESTM || $format eq $IECTM || $format eq $IESM) {
	if ($text ne "" && $text !~ /\s$/){
	    $text .=  " ";
	}
	$text .= $ne_attr_stack{$norm_tag};
    }
}

sub proc_end_NE_tag{
    local($tag,$data) = @_;
    local($norm_tag) = $tag;
    $norm_tag =~ s/E_//;

    delete $ne_attr_stack{$norm_tag};
    if ($format eq $IESTM || $format eq $IECTM || $format eq $IESM) {
	if ($text =~ /\s$/){
	    $text =~ s/\s+$// ;
	    $text .= "\<\/$norm_tag\> ";
	} else {
	    $text .= "\<\/$norm_tag\>";
	}
    }
}

sub proc_beg_ASR_tag{
    local($tag,$data) = @_;
    local($norm_tag) = $tag;

    ### Pre-check, ASR tags ARE word separators
    &proc_beg_separator("") if ($text ne "");

    $norm_tag =~ s/B_//;

    $asr_attr_stack{$norm_tag} = "<$norm_tag";
    foreach $key($data->attribute_names){
	next if ($key eq "STARTTIME" || $key eq "ENDTIME");
	$asr_attr_stack{$norm_tag} .= " $key=" .
	    ((! $data->attribute($key)->is_implied) ? 
	     $data->attribute($key)->value : "" );
    }
    foreach $key($data->attribute_names){
	next unless ($key eq "STARTTIME");
	$asr_attr_stack{$norm_tag} .= " $key=" .
	    ((! $data->attribute($key)->is_implied) ? 
	     $data->attribute($key)->value : "" );
    }
    foreach $key($data->attribute_names){
	next unless ($key eq "ENDTIME");
	$asr_attr_stack{$norm_tag} .= " $key=" .
	    ((! $data->attribute($key)->is_implied) ? 
	     $data->attribute($key)->value : "" );
    }
    $asr_attr_stack{$norm_tag} .= ">";
    # print "${DBoff}  ASR stack $asr_attr_stack{$norm_tag}\n"

    # process overlap if flagged
    #
#    if ($process_overlap) {
    if (1) {

	# if overlap or noscore, divide each into individual turn and 
	# eliminate later
	#
	if (($norm_tag eq "OVERLAP" && $process_overlap) || $norm_tag eq "NOSCORE" ){ 
	    local $starttime_tag;
	    local $starttime;
	    local $endtime_tag;
	    local $endtime;
	    
	    if ($asr_attr_stack{$norm_tag} =~ /(STARTTIME)=(\S+)\s*(ENDTIME)=(\S+)\>/) {	
		$starttime_tag = $1;
		$starttime = $2;
		$endtime_tag = $3;
		$endtime = $4;
	    } else {
		die "Unable to process '$norm_tag' tags w/o times"
	    }
	    
	    if ($turn_elems{$starttime_tag} != $starttime) {
		$turn_elems{$endtime_tag} = $starttime;
		
		$turn_elems{$TEXT_KEY} = "";
		
		# see note in proc_end_turn routine
		#
		&proc_preproc_text (*text_elem);
		
		for ($i = 0; $i <= $#text_elem; $i++) {
		    $turn_elems{$TEXT_KEY} .= $text_elem[$i] . " ";
		}

		push @turn_stack, { %turn_elems };
	    }

	    %turn_elems = ();
	    @text_elem = ();
	    
	    # copy prev attribute over
	    #
	    for $key (keys %section_attr) {
		$turn_elems{$key} = $section_attr{$key};
	    }
	    
	    for $key (keys %turn_attr) {
		$turn_elems{$key} = $turn_attr{$key};
	    }
	    
	    $turn_elems{$starttime_tag} = $starttime;
	    $turn_elems{$endtime_tag} = $endtime;
	    $turn_elems{$FILENAME_KEY} = $filename;
	    $turn_elems{$NUM_CHAN_KEY} = $num_chan;
	    $turn_elems{$FOCUS_COND_KEY} = &proc_get_focus_cond(
		   $turn_attr{$DIALECT_KEY}, 
		   $turn_attr{$FIDELITY_KEY}, 
		   $turn_attr{$MODE_KEY}, 
		   $bkgrnd_attr_stack{$MUSIC_KEY}, 
		   $bkgrnd_attr_stack{$SPEECH_KEY},
                   $bkgrnd_attr_stack{$OTHER_KEY}); 
 	    $turn_elems{$TEXT_KEY} = "";
	}
    } elsif ($format eq $IECTM &&
	     (!$use_wtag) &&
	     ($norm_tag eq "OVERLAP" || $norm_tag eq "NOSCORE") &&
	     $#text_elem >= 0) {
	### divide the turn here so that the inferred word time work out
	if ($asr_attr_stack{$norm_tag} =~
	    /(STARTTIME)=(\S+)\s*(ENDTIME)=(\S+)\>/) {	
	    $starttime_tag = $1;
	    $starttime = $2;
	    $endtime_tag = $3;
	    $endtime = $4;

	    #### flush the residual data 
	    $turn_elems{$endtime_tag} = $starttime;
	    &proc_preproc_text (*text_elem);
	    $turn_elems{$TEXT_KEY} .= join(" ",@text_elem) . " ";
	    push @turn_stack, { %turn_elems };

	    # copy prev attribute over
	    #
	    for $key (keys %section_attr) {
		$turn_elems{$key} = $section_attr{$key};
	    }
	    
	    for $key (keys %turn_attr) {
		$turn_elems{$key} = $turn_attr{$key};
	    }
	    
	    $turn_elems{$starttime_tag} = $starttime;
	    $turn_elems{$endtime_tag} = $endtime;
	    $turn_elems{$FILENAME_KEY} = $filename;
	    $turn_elems{$NUM_CHAN_KEY} = $num_chan;
	    $turn_elems{$FOCUS_COND_KEY} = &proc_get_focus_cond(
		   $turn_attr{$DIALECT_KEY}, 
		   $turn_attr{$FIDELITY_KEY}, 
		   $turn_attr{$MODE_KEY}, 
		   $bkgrnd_attr_stack{$MUSIC_KEY}, 
		   $bkgrnd_attr_stack{$SPEECH_KEY},
                   $bkgrnd_attr_stack{$OTHER_KEY}); 
 	    $turn_elems{$TEXT_KEY} = "";
	    @text_elem = ();
	}
    }
    
    if ($norm_tag eq "FOREIGN" || $norm_tag eq "UNCLEAR") {
	$parenthesize_text = 1;
    }

    &dump_turn_stack() 	  if ($debug_level > $DEBUG_DETAIL) ;
}

sub proc_end_ASR_tag{
    local($tag,$data) = @_;
    local($norm_tag) = $tag;
    $norm_tag =~ s/E_//;

    ### Pre-check, ASR tags ARE word separators
    &proc_beg_separator("") if ($text ne "");

    # print STDERR "==== text arr '".join(" ",@text_elem)."'\n";

#    if ($process_overlap) {
    if (1) {
	if (($norm_tag eq "OVERLAP" && $process_overlap) || $norm_tag eq "NOSCORE") {
	    $turn_elems{$SPEAKER_KEY} = $EXCLUDED_REGION;
	    $turn_elems{$SPKRTYPE_KEY} = $UNKNOWN;
	    $turn_elems{$FOCUS_COND_KEY} = "";
	    if ($keep_exclude_text){
		$turn_elems{$TEXT_KEY} = join(" ",@text_elem)." ";
	    } else {
		$turn_elems{$TEXT_KEY} = $IGNORE_TIME_SEGMENT_IN_SCORING;
	    }
	    push @turn_stack, { %turn_elems };
	    
	    %turn_elems = ();
	    @text_elem = ();
	    
	    # copy prev attribute over
	    #
	    for $key (keys %section_attr) {
		$turn_elems{$key} = $section_attr{$key};
	    }
	    for $key (keys %turn_attr) {
		$turn_elems{$key} = $turn_attr{$key};
	    }

	    if ($asr_attr_stack{$norm_tag} =~ /(STARTTIME)=(\S+)\s*(ENDTIME)=(\S+)\>/) {
		$starttime_tag = $1;
		$starttime = $2;
		$endtime_tag = $3;
		$endtime = $4;
	    }
	    
	    if ($turn_elems{$endtime_tag} != $endtime) {
		
		$turn_elems{$starttime_tag} = $endtime;
		$turn_elems{$FILENAME_KEY} = $filename;
		$turn_elems{$NUM_CHAN_KEY} = $num_chan;
		$turn_elems{$FOCUS_COND_KEY} = &proc_get_focus_cond(
			    $turn_attr{$DIALECT_KEY}, 
                            $turn_attr{$FIDELITY_KEY}, 
                            $turn_attr{$MODE_KEY}, 
                            $bkgrnd_attr_stack{$MUSIC_KEY}, 
                            $bkgrnd_attr_stack{$SPEECH_KEY},
                            $bkgrnd_attr_stack{$OTHER_KEY}); 
	        $turn_elems{$TEXT_KEY} = "";
	    }
	    else {
		$kept_turn = 0;
	    }
	}
    } elsif ($format eq $IECTM &&
	     (!$use_wtag) &&
	     ($norm_tag eq "OVERLAP" || $norm_tag eq "NOSCORE") &&
	     $#text_elem >= 0) {
	### divide the turn here so that the inferred word time work out
	if ($asr_attr_stack{$norm_tag} =~
	    /(STARTTIME)=(\S+)\s*(ENDTIME)=(\S+)\>/) {	
	    $starttime_tag = $1;
	    $starttime = $2;
	    $endtime_tag = $3;
	    $endtime = $4;

	    #### flush the residual data 
	    $turn_elems{$endtime_tag} = $endtime;
	    &proc_preproc_text (*text_elem);
	    $turn_elems{$TEXT_KEY} .= join(" ",@text_elem) . " ";
	    push @turn_stack, { %turn_elems };

	    # copy prev attribute over
	    #
	    for $key (keys %section_attr) {
		$turn_elems{$key} = $section_attr{$key};
	    }
	    
	    for $key (keys %turn_attr) {
		$turn_elems{$key} = $turn_attr{$key};
	    }
	    
	    $turn_elems{$starttime_tag} = $endtime;
	    $turn_elems{$endtime_tag} = $turn_attr{$endtime_tag};
	    $turn_elems{$FILENAME_KEY} = $filename;
	    $turn_elems{$NUM_CHAN_KEY} = $num_chan;
	    $turn_elems{$FOCUS_COND_KEY} = &proc_get_focus_cond(
		   $turn_attr{$DIALECT_KEY}, 
		   $turn_attr{$FIDELITY_KEY}, 
		   $turn_attr{$MODE_KEY}, 
		   $bkgrnd_attr_stack{$MUSIC_KEY}, 
		   $bkgrnd_attr_stack{$SPEECH_KEY},
                   $bkgrnd_attr_stack{$OTHER_KEY}); 
 	    $turn_elems{$TEXT_KEY} = "";
	    @text_elem = ();
	}
    }

    # parenthesize the text if foreign or unclear
    #
    if ($norm_tag eq "FOREIGN" || $norm_tag eq "UNCLEAR") {
	$parenthesize_text = 0;
    }
    
    delete $asr_attr_stack{$norm_tag};

    # print "==== exit text arr '".join(" ",@text_elem)."'\n";

    &dump_turn_stack() 	  if ($debug_level > $DEBUG_DETAIL) ;
}

sub dump_turn_stack{
    my $ji;
    my $jkey;

    #### Turn stack

    for ($ji=0; $ji<=$#turn_stack; $ji++){
	print "Turn $ji\n";
	foreach $jkey(sort(keys %{ $turn_stack[$ji] } )){
	    print "      $jkey=$turn_stack[$ji]{$jkey}\n";
	}
    }
}

sub proc_beg_cont{
    local($data) = @_;

    $attr_stack{$data->name} = "";
    foreach $key($data->attribute_names){
	if (! $data->attribute($key)->is_implied){
	    $attr_stack{$data->name} .= "$key=" . $data->attribute($key)->value . " ";
	}
    }    
}


sub proc_beg_bkgrnd{

    local($data) = @_;
    my $cur_focus_cond;

    ###  my $x1; print "Begin_bkgnd:\n"; foreach $x1(sort(keys %turn_elems)){print "   $x1 -> '$turn_elems{$x1}'\n"; } }

    # get the current focus condition. The reason is that sometimes
    # the level of noise changing from high->low or low->high is not
    # enough to split the turn.  When they do split the turn (high/low->off 
    # or off->high/low) the focus condition is still the 
    # condition before high->low or low->high changes. We need to get the 
    # focus condition after the high->low and vice versa changes but before
    # the high/low->off and vice versa changes
    #
      if ($debug_level > $DEBUG_DETAIL){
       	  print "${DBoff}  Before BKG music=$bkgrnd_attr_stack{$MUSIC_KEY} ".
                "speech=$bkgrnd_attr_stack{$SPEECH_KEY} ".
                "other=$bkgrnd_attr_stack{$OTHER_KEY}\n"
      }

    if ($in_turn) {
	$cur_focus_cond = &proc_get_focus_cond(
                      $turn_attr{$DIALECT_KEY}, 
                      $turn_attr{$FIDELITY_KEY}, 
                      $turn_attr{$MODE_KEY}, 
                      $bkgrnd_attr_stack{$MUSIC_KEY}, 
                      $bkgrnd_attr_stack{$SPEECH_KEY},
		      $bkgrnd_attr_stack{$OTHER_KEY}); 
    }

    ### I believe this is BAD, it should be ...
    #local $mlevel = $music_level;
    #local $slevel = $speech_level;
    #local $olevel = $other_level;
    local $mlevel = ($bkgrnd_attr_stack{$MUSIC_KEY} eq "off") ? "off" : "on";
    local $slevel = ($bkgrnd_attr_stack{$SPEECH_KEY} eq "off") ? "off" : "on";
    local $olevel = ($bkgrnd_attr_stack{$OTHER_KEY} eq "off") ? "off" : "on";
    ####
    local $bkgrnd_prev_stime = $bkgrnd_attr_stack{$TURN_STIME_KEY};

    foreach $key($data->attribute_names){
	if (! $data->attribute($key)->is_implied){
	    if (lc($data->attribute($key)->value) eq "music") {
		$bkgrnd_attr_stack{$MUSIC_KEY} = 
		    lc($data->attribute($LEVEL_KEY)->value);
		if ($bkgrnd_attr_stack{$MUSIC_KEY} eq $HIGH ||
		    $bkgrnd_attr_stack{$MUSIC_KEY} eq $LOW) {
		    $mlevel = $ON;
		}
		else {
		    $mlevel = $OFF;
		}
	    }
	    elsif (lc($data->attribute($key)->value) eq "speech") {
		$bkgrnd_attr_stack{$SPEECH_KEY} = 
		    lc($data->attribute($LEVEL_KEY)->value);
		if ($bkgrnd_attr_stack{$SPEECH_KEY} eq $HIGH ||
		    $bkgrnd_attr_stack{$SPEECH_KEY} eq $LOW) {
		    $slevel = $ON;
		}
		else {
		    $slevel = $OFF;
		}
	    }
	    elsif (lc($data->attribute($key)->value) eq "other") {
		$bkgrnd_attr_stack{$OTHER_KEY} = 
		    lc($data->attribute($LEVEL_KEY)->value);
		if ($bkgrnd_attr_stack{$OTHER_KEY} eq $HIGH ||
		    $bkgrnd_attr_stack{$OTHER_KEY} eq $LOW) {
		    $olevel = $ON;
		}
		else {
		    $olevel = $OFF;
		}
	    }
	    elsif ($key eq $TURN_STIME_KEY) {
		$bkgrnd_attr_stack{$TURN_STIME_KEY} = 
		    $data->attribute($key)->value;
	    }
	}
    }

    if ($in_turn) {
        if ($debug_level > $DEBUG_DETAIL){
	    print "     ??? should i Dividing Turn\n";
            print "	if ($music_level ne $mlevel || $speech_level ne $slevel ||".
	        " $other_level ne $olevel || (!$text_flag &&".
                " ($bkgrnd_attr_stack{$TURN_STIME_KEY} - ".
                "$turn_elems{$TURN_ETIME_KEY} > $THRESHOLD)))\n";
        }
	if ($music_level ne $mlevel || $speech_level ne $slevel ||
	    $other_level ne $olevel || (!$text_flag && ($bkgrnd_attr_stack{$TURN_STIME_KEY} - $turn_elems{$TURN_ETIME_KEY} > $THRESHOLD))) {

            print "     Dividing Turn\n"  if ($debug_level > $DEBUG_DETAIL);
	    local $starttime = $turn_elems{$TURN_ETIME_KEY};
	    if ($text_flag == 0) {
		$turn_elems{$TURN_ETIME_KEY} = $bkgrnd_prev_stime;
	    }
	    else {
		$turn_elems{$TURN_ETIME_KEY} = $bkgrnd_attr_stack{$TURN_STIME_KEY};
	    }
	    local $i;
	    $turn_elems{$FOCUS_COND_KEY} = $cur_focus_cond;
	    $turn_elems{$TEXT_KEY} = "";
	    
	    # see note in proc_end_turn routine
	    #
	    &proc_preproc_text (*text_elem);
	    
	    for ($i = 0; $i <= $#text_elem; $i++) {
		$turn_elems{$TEXT_KEY} .= $text_elem[$i] . " ";
	    }
	    
	    push @turn_stack, { %turn_elems };
	    ### { my $x1; print "About to push:\n"; foreach $x1(sort(keys %turn_elems)){print "   $x1 -> '$turn_elems{$x1}'\n"; } }

	    %turn_elems = ();
	    @text_elem = ();
	    
	    # copy prev attribute over
	    #
	    for $key (keys %section_attr) {
		$turn_elems{$key} = $section_attr{$key};
	    }
	    for $key (keys %turn_attr) {
		$turn_elems{$key} = $turn_attr{$key};
	    }
	    
	    $turn_elems{$TURN_STIME_KEY} = $bkgrnd_attr_stack{$TURN_STIME_KEY};
	    $turn_elems{$TURN_ETIME_KEY} = $starttime;
	    $turn_elems{$FILENAME_KEY} = $filename;
	    $turn_elems{$NUM_CHAN_KEY} = $num_chan;
	    $turn_elems{$FOCUS_COND_KEY} = &proc_get_focus_cond(
                      $turn_attr{$DIALECT_KEY}, 
                      $turn_attr{$FIDELITY_KEY}, 
                      $turn_attr{$MODE_KEY}, 
                      $bkgrnd_attr_stack{$MUSIC_KEY}, 
                      $bkgrnd_attr_stack{$SPEECH_KEY},
		      $bkgrnd_attr_stack{$OTHER_KEY}); 
	    
	}
	
    }
    $music_level = $mlevel;
    $speech_level = $slevel;
    $other_level = $olevel;
    $text_flag = 0;
}

# determine if two segments, i and j, are overlapping.
# two segments can be of the situations below:
#
#  i           ###############
#1 j  ##########                        no overlapped
#  j  ####
#2 j  ############                      overlap begin j end i
#  j  ################
#3 j  ########################          j
#  j  ##############################    j
#4 j           #########                i
#  j           ###############          i
#5 j           ###################      j
#6 j               #######              i
#  j               ###########          i
#7 j               ###############      overlap begin i end j
#8 j                         #####      no overlapped
# 
# return : an array of 3 items
#  no overlap or overlap, begin time, and end time
#  
sub proc_is_overlap{

    local ($btime_i, $etime_i, $chan_i, $btime_j, $etime_j, $chan_j) = @_;

#    print "($btime_i, $etime_i, $chan_i, $btime_j, $etime_j, $chan_j)\n";

    local @return_val = ();
    
    # Initial case
    #
    if ($chan_i ne $chan_j) {
	push(@return_val, $NO_OVERLAP);
	push(@return_val, 0);
	push(@return_val, 0);
	return (@return_val);
    }
    

    # case 1
    #
    if ($btime_i eq $etime_j || ($btime_i - $etime_j) > -0.0001) {
	push(@return_val, $NO_OVERLAP);
	push(@return_val, 0);
	push(@return_val, 0);
	return (@return_val);
    }
    # case 8
    #
    elsif ($btime_j eq $etime_i || ($btime_j - $etime_i) > -0.0001) {
	push(@return_val, $NO_OVERLAP);
	push(@return_val, 0);
	push(@return_val, 0);
	return (@return_val);
    }
    # case 2,3,4,5,6,7
    #
    else {
	# case 2,3
	#
	if ($btime_j < $btime_i) {
	    # case 2
	    #
	    if ($etime_j < $etime_i) {
		push(@return_val, $OVERLAP);
		push(@return_val, $btime_j);
		push(@return_val, $etime_i);
		return (@return_val);
	    }
	    # case 3
	    #
	    else {
		push(@return_val, $OVERLAP);
		push(@return_val, $btime_j);
		push(@return_val, $etime_j);
		return (@return_val);
	    }
	}
	# case 4,5,6,7
	#
	else {
	    # case 4,5
	    #
	    if ($btime_j == $btime_i) {
		# case 4
		#
		if ($etime_j < $etime_i) {
		    push(@return_val, $OVERLAP);
		    push(@return_val, $btime_i);
		    push(@return_val, $etime_i);
		    return (@return_val);
		}
		# case 5
		#
		else {
		    push(@return_val, $OVERLAP);
		    push(@return_val, $btime_j);
		    push(@return_val, $etime_j);
		    return (@return_val);
		}
	    }
	    # case 6,7
	    #
	    else {
		# case 6
		#
		if ($etime_j <= $etime_i) {
		    push(@return_val, $OVERLAP);
		    push(@return_val, $btime_i);
		    push(@return_val, $etime_i);
		    return (@return_val);
		}
		# case 7
		#
		else {
		    push(@return_val, $OVERLAP);
		    push(@return_val, $btime_i);
		    push(@return_val, $etime_j);
		    return (@return_val);
		}
	    }
	}
    }
}

sub proc_elim_overlap{
    
    local(*stack) = @_;
    local $i;
    local $j;
    local @status = ();

    # find the maximum overlap if any
    #
    for ($i = 0; $i < $#stack; $i++) {
	for ($j = $i+1; $j <= $#stack &&
	     $stack[$j]{$TURN_STIME_KEY}-5 < $stack[$i]{$TURN_ETIME_KEY};
	     $j++) {
	    @status = &proc_is_overlap($stack[$i]{$TURN_STIME_KEY}, 
				       $stack[$i]{$TURN_ETIME_KEY},
				       $stack[$i]{$TURN_CHANNEL_KEY},
				       $stack[$j]{$TURN_STIME_KEY},
				       $stack[$j]{$TURN_ETIME_KEY},
				       $stack[$j]{$TURN_CHANNEL_KEY});
	    if ($status[0] != $NO_OVERLAP) {
		if ($debug_level > $DEBUG_DETAIL) {
		    print "${DBoff}  Found Overlapped turns\n";
		    print "${DBoff}     ".join(" ",%{ $stack[$i] })."\n";
		    print "${DBoff}     ".join(" ",%{ $stack[$j] })."\n";
		}
		
		### Convert $i to an exclude tag
		$stack[$i]{$SPEAKER_KEY} = $EXCLUDED_REGION;
		$stack[$i]{$SPKRTYPE_KEY} = $UNKNOWN;
		$stack[$i]{$TURN_STIME_KEY} = 
		    &MIN($stack[$i]{$TURN_STIME_KEY}, 
			 $stack[$j]{$TURN_STIME_KEY});
		$stack[$i]{$TURN_ETIME_KEY} = 
		    &MAX($stack[$i]{$TURN_ETIME_KEY}, 
			 $stack[$j]{$TURN_ETIME_KEY});
		$stack[$i]{$FOCUS_COND_KEY} = "";
		if ($keep_exclude_text) {
		    $stack[$i]{$TEXT_KEY} .= " - $stack[$j]{$TEXT_KEY}";
		} else {
		    $stack[$i]{$TEXT_KEY} = $IGNORE_TIME_SEGMENT_IN_SCORING;
		}

		### Delete Stack element $j
		splice(@stack,$j,1);
		
		### Decriment $j to reflect the deletion
		$j--;
	    }
	}
    }
}

sub write_stm_header{
    local ($fp) = @_;
    local $cat = 0;
    print $fp ";; CATEGORY \"$cat\" \"\" \"\"\n";
    print $fp ";; LABEL \"O\" \"Overall\" \"Overall\"\n";
    print $fp ";;\n";
    $cat ++;
    
    if ($focus_condition_enabled) {
	print $fp ";; CATEGORY \"$cat\" \"Hub4 Focus Conditions\" \"\"\n";
	print $fp ";; LABEL \"F0\" \"Baseline//Broadcast//Speech\" \"\"\n";
	print $fp ";; LABEL \"F1\" \"Spontaneous//Broadcast//Speech\" \"\"\n";
	print $fp ";; LABEL \"F2\" \"Speech Over//Telephone//Channels\" \"\"\n";
	print $fp ";; LABEL \"F3\" \"Speech in the//Presence of//Background Music\" \"\"\n";
	print $fp ";; LABEL \"F4\" \"Speech Under//Degraded//Acoustic Conditions\" \"\"\n";
	print $fp ";; LABEL \"F5\" \"Speech from//Non-Native//Speakers\" \"\"\n";
	print $fp ";; LABEL \"FX\" \"All other speech\" \"\"\n";
	$cat ++;
    }
    print $fp ";; CATEGORY \"$cat\" \"Speaker Sex\" \"\"\n";
    print $fp ";; LABEL \"female\" \"Female\" \"\"\n";
    print $fp ";; LABEL \"male\"   \"Male\" \"\"\n";
    print $fp ";; LABEL \"child\"   \"Child\" \"\"\n";
    print $fp ";; LABEL \"$UNKNOWN\"   \"Unknown\" \"\"\n";
    $cat++;
}

sub proc_print_stack{

    local($fp, $format, *stack) = @_;
    local $i;
    
    if ($format eq $IESTM || $format eq $STM) {
	&write_stm_header($fp);
	&proc_print_stm($fp, *stack);
    }

    elsif ($format eq $IESM || $format eq $SM) {
	&proc_print_stm_sm_u($fp, *stack);
    }

    # do some post processing to get ctm format
    #
    elsif ($format eq $IECTM || $format eq $CTM) {
	if ($use_wtag) {
	    &proc_stm2ctm_wspec ($fp, *stack);
	}
	else {
	    &proc_stm2ctm_wavg ($fp, *stack);
	}
    }
}

sub proc_get_focus_cond{
    
    local($dialect, $fidelity, $mode, $music_lvl, $speech_lvl, 
	  $other_lvl) = @_;

    local $focus_cond = "";
	    
    # determine focus condition from dialect, fidelity, mode, and background
    # attributes
    #
    # F0, F1, F2, F3, F4, FX
    #
    if ($dialect eq "$NATIVE") {
	
	# F0, F1, F3, F4, FX
	#
	if ($fidelity eq $HIGH) {
	    
	    # F0, F1, FX
	    #
	    if ($music_lvl eq $OFF && $speech_lvl eq $OFF &&
		$other_lvl eq $OFF) {
		
		# F0
		#
		if ($mode eq "$PLANNED") {
		    $focus_cond = "f0";
		}
		
		# F1
		#
		elsif ($mode eq "$SPONTANEOUS") {
		    $focus_cond = "f1";
		}

		# FX
		#
		else {
		    $focus_cond = "fx";
		}
	    }
	    
	    # F3
	    #
	    elsif ($music_lvl ne $OFF && $speech_lvl eq $OFF && 
		   $other_lvl eq $OFF) {
		$focus_cond = "f3";
	    }
		
	    # F4
	    #
	    elsif ($music_lvl eq $OFF && ($speech_lvl ne $OFF ||
		   $other_lvl ne $OFF)) {
		$focus_cond = "f4";
	    }

	    # FX
	    #
	    else {
		$focus_cond = "fx";
	    }
	}
	
	# F2, FX
	#
	else {

	    # F2
	    #
	    if ($music_lvl eq $OFF && $speech_lvl eq $OFF &&
		$other_lvl eq $OFF) {
		$focus_cond = "f2";
	    }

	    # FX
	    #
	    else {
		$focus_cond = "fx";
	    }
	}
    }
    # F5, FX
    #
    else {
	
	# F5, FX
	#
	if ($fidelity eq $HIGH) {

	    # F5
	    #
	    if ($music_lvl eq $OFF && $speech_lvl eq $OFF &&
		$other_lvl eq $OFF) {

		if ($mode eq "$PLANNED") {
		    $focus_cond = "f5";
		}

		# FX
		#
		else {
		    $focus_cond = "fx";
		}
	    }
	    else {
		$focus_cond = "fx";
	    }
	}
	
	# FX
	#
	else {
	    $focus_cond = "fx";
	}
    }
    die "Error: Focus condition not assigned" if ($focus_cond eq "");
    return ($focus_cond);    
}

sub proc_fill_gap{

    local(*stack) = @_;

    local $i = 0;
    local $key;
    local %tmp_turn = ();

    while ($i < $#stack) {
	
	%tmp_turn = ();
        
        # there's a gap
	#
	if ($stack[$i+1]{$TURN_STIME_KEY} - $stack[$i]{$TURN_ETIME_KEY} > $THRESHOLD) {

	    $tmp_turn{$SECTION_STIME_KEY} = -1;
	    $tmp_turn{$SECTION_ETIME_KEY} = -1;
	    $tmp_turn{$SECTION_TYPE_KEY} = $UNKNOWN;
	    $tmp_turn{$SECTION_ID_KEY} = $UNKNOWN;
	    $tmp_turn{$FILENAME_KEY} = $filename;
	    $tmp_turn{$NUM_CHAN_KEY} = $num_chan;
	    $tmp_turn{$SPEAKER_KEY} = $INTER_SEGMENT_GAP;
	    $tmp_turn{$SPKRTYPE_KEY} = "";
	    $tmp_turn{$FOCUS_COND_KEY} = "fx";
	    $tmp_turn{$TURN_CHANNEL_KEY} = $stack[$i]{$TURN_CHANNEL_KEY};
	    $tmp_turn{$TURN_STIME_KEY} = $stack[$i]{$TURN_ETIME_KEY};
	    $tmp_turn{$TURN_ETIME_KEY} = $stack[$i+1]{$TURN_STIME_KEY};
	    $tmp_turn{$TEXT_KEY} = "";
	    splice @stack, $i+1, 0, { % tmp_turn };
	}
	$i++;
    }
}

sub proc_bystarttime {

    %ah = %{ $a };
    %bh = %{ $b };
    if ($ah{$TURN_CHANNEL_KEY} ne $bh{$TURN_CHANNEL_KEY}){
	$ah{$TURN_CHANNEL_KEY} cmp $bh{$TURN_CHANNEL_KEY};
    } else {
	$ah{$TURN_STIME_KEY} <=> $bh{$TURN_STIME_KEY};
    }
}

sub proc_preproc_text{
    local(*stack) = @_;
    local $i;
    local $j;

    print "---- before preprec_text '".join("^",@stack)."'\n"
	if ($debug_level > $DEBUG_DETAIL) ;
    for ($i = 0; $i <= $#stack; $i++) {
	$stack[$i] =~ s/\.@(['< ])/.$1/g;
#	$stack[$i] =~ s/\.@([^@])/.+$1/g;
	$stack[$i] =~ s/@\)/)/g;
	$stack[$i] =~ s/@/ /g;
    }
    print "---- after preprec_text '".join("^",@stack)."'\n"
	if ($debug_level > $DEBUG_DETAIL) ;
}

sub proc_del_turn{
    local(*stack) = @_;

    local $i = 0;

    # kill turns whose text field is empty
    #
    while ($i <= $#stack) {

	if ($stack[$i]{$TEXT_KEY} =~ /^\s*$/) {
	    splice @stack, $i, 1;
	} 
	else {
	    $i++;
	}
    }
}

sub proc_dump_stack {
    local (*stack) = @_;
    my ($k);
    for ($i = 0; $i <= $#stack; $i++) {
	print "-----------------------------------------------------\n";
	foreach $k(sort(keys %{ $stack[$i] })){
	    print "Stack $i: $k -> '$stack[$i]{$k}'\n";
	}
    }
}


sub proc_print_stm {
    local ($fp, *stack) = @_;

    # proc_dump_stack(*stack);

    for ($i = 0; $i <= $#stack; $i++) {
	print $fp "$stack[$i]{$FILENAME_KEY} ";
	print $fp "$stack[$i]{$TURN_CHANNEL_KEY} ";
	print $fp "$stack[$i]{$SPEAKER_KEY} ";
	printf $fp ("%.6f ", $stack[$i]{$TURN_STIME_KEY});
	printf $fp ("%.6f ", $stack[$i]{$TURN_ETIME_KEY});
	print $fp "<o";
	if ($focus_condition_enabled) {
	    print $fp ",$stack[$i]{$FOCUS_COND_KEY}";
	}
	print $fp ",$stack[$i]{$SPKRTYPE_KEY}";
	print $fp "> ";
	print $fp "$stack[$i]{$TEXT_KEY}\n";
    }
}

sub proc_print_stm_sm_u {
    local ($fp, *stack) = @_;
    
    local $prev_stime = -1;
    local $prev_etime = -1;

    local $temp_focus_cond = "";

    for ($i = 0; $i <= $#stack; $i++) {
	if (($stack[$i]{$SECTION_STIME_KEY} != $prev_stime) &&
	    ($stack[$i]{$SECTION_ETIME_KEY} != $prev_etime) &&
	    ($stack[$i]{$SECTION_STIME_KEY} != -1)) {
	    
	    if ($i != 0) {
		print $fp "</Section>\n";
	    }
	    printf $fp ("<Section Type=%s S_time=%.6f E_time=%.6f id=%s>\n",
			$stack[$i]{$SECTION_TYPE_KEY}, 
			$stack[$i]{$SECTION_STIME_KEY}, 
			$stack[$i]{$SECTION_ETIME_KEY}, 
			$stack[$i]{$SECTION_ID_KEY});
	    $prev_stime = $stack[$i]{$SECTION_STIME_KEY}; 
	    $prev_etime = $stack[$i]{$SECTION_ETIME_KEY}; 
	}
	print $fp "$stack[$i]{$FILENAME_KEY} ";
	print $fp "$stack[$i]{$TURN_CHANNEL_KEY} ";
	print $fp "$stack[$i]{$SPEAKER_KEY} ";
	printf $fp ("%.6f ", $stack[$i]{$TURN_STIME_KEY});
	printf $fp ("%.6f ", $stack[$i]{$TURN_ETIME_KEY});

	$temp_focus_cond = "<o";
	if ($focus_condition_enabled) {
	    $temp_focus_cond .= ",".$stack[$i]{$FOCUS_COND_KEY};
	}	
	$temp_focus_cond .= ",$stack[$i]{$SPKRTYPE_KEY}";
	$temp_focus_cond .= ">";

	print $fp "$temp_focus_cond ";

	print $fp "$stack[$i]{$TEXT_KEY}\n";
    }
    print $fp "</Section>\n";
}

sub proc_stm2ctm_wavg{
    local ($fp, *stack) = @_;
    local $i;
    local $j;
    local $stime;
    local $duration;
    local @num_words = ();
    local $tmp_str = "";

    for ($i = 0; $i <= $#stack; $i++) {
	if (($stack[$i]{$TEXT_KEY} ne $IGNORE_TIME_SEGMENT_IN_SCORING) &&
	    ($stack[$i]{$SPEAKER_KEY} ne $INTER_SEGMENT_GAP)) {
	    $stime = $stack[$i]{$TURN_STIME_KEY};
	    @num_words = ();

	    ### Delete optionally deletable words!!!!	    
	    # ($newtext = $stack[$i]{$TEXT_KEY}) =~ s/\([^\(\)]*\)//g;
	    ### Convert optionally deletable words to normal words
	    ($newtext = $stack[$i]{$TEXT_KEY}) =~ s/[\(\)]//g;
	    
	    foreach $word(ie_split($newtext)) {
		if ($word !~ /^\s*$/){
		    push(@num_words,$word) if ($word !~ /^(-\S+|\S+-)$/);
		}
	    }
	    ### splice back together
#	    for ($j=0; $j<$#num_words; $j++){
#		if ($num_words[$j] =~ /<(ENAMEX|TIMEX|NUMEX)/i){	
#		    if ($num_words[$j] !~ /<\/(ENAMEX|TIMEX|NUMEX)>/i){
#			$num_words[$j] .= "#".splice(@num_words,$j+1,1);
#			$j--;
#		    }
#		}
#	    }


#           ALD FIX	    
#	    for ($j=0; $j<$#num_words; $j++){
#		if ($num_words[$j] =~ /<(ENAMEX|TIMEX|NUMEX)/i){	
#		    $num_words[$j] .= splice(@num_words,$j+1,1);
#		} elsif ($num_words[$j] =~ /<\/(ENAMEX|TIMEX|NUMEX)>/i){
#		    $num_words[$j-1] .= splice(@num_words,$j,1);
#		    $j--;
#		}
#	    }
	    
	    for ($j=0; $j<$#num_words; $j++){
		if ($num_words[$j] =~ /^<(ENAMEX|TIMEX|NUMEX)[^<>]*>$/i){	
		    $num_words[$j] .= splice(@num_words,$j+1,1);
		} elsif ($num_words[$j] =~ /^<\/(ENAMEX|TIMEX|NUMEX)[^<>]*>$/i){
		    $num_words[$j-1] .= splice(@num_words,$j,1);
		    $j--;
		}
	    }

	    print "${DBoff} stm2ctm wavg: ~".join("~",@num_words)."~\n"
		if ($debug_level > $DEBUG_DETAIL) ;

#	    ###
	    $num_words = $#num_words + 1;

	    if ($#num_words < 0) {
		print "Warning: number of words is zero\n";
		next;
	    }

	    $duration = ($stack[$i]{$TURN_ETIME_KEY} - $stack[$i]{$TURN_STIME_KEY})/($#num_words+1);
	    for ($j = 0; $j <= $#num_words; $j++) {
		if ($num_words[$j] =~ /\(\S+\)/) {
		    next;
		}
#		elsif ($num_words[$j] =~ /<ENAMEX|<TIMEX|<NUMEX\s*$/) {
#		    $tmp_str = "$num_words[$j] ";
#		}
		else {
		    print $fp "$stack[$i]{$FILENAME_KEY} ";
		    print $fp "$stack[$i]{$NUM_CHAN_KEY} ";
		    printf $fp ("%.6f ", $stime);
		    printf $fp ("%.6f ", $duration);
		    $tmp_str .= $num_words[$j];
		    print $fp "$tmp_str\n";
		    $tmp_str = "";
		}
		$stime = $stime + $duration;
	    }
	}
    }
}

sub proc_stm2ctm_wspec{
    local ($fp, *stack) = @_;
    local $i;
    local $j;
    local @num_words = ();

    for ($i = 0; $i <= $#stack; $i++) {
	if ($stack[$i]{$TEXT_KEY} ne $IGNORE_TIME_SEGMENT_IN_SCORING) {
	    @num_words = ();
	    @num_words = split(/STIME=/, $stack[$i]{$TEXT_KEY});
	    
	    print "${DBoff} stm2ctm wspec: ~".join("~",@num_words)."~\n"
		if ($debug_level > $DEBUG_DETAIL) ;
	    for ($j = 0; $j <= $#num_words; $j++) {

		### Remove all optionally deletable markers
		$num_words[$j] =~ s/[\(\)]//g;

		$num_words[$j] =~ s/ETIME=//g;
		if ($num_words[$j] =~ /(\d+\.\d+)\s+(\d+\.\d+)\s+\(\S+\)$/) {
		    next;
		}
		elsif ($num_words[$j] =~ /(\d+\.\d+)\s+(\d+\.\d+)\s+(\S+)(<[^<>]+>)/) {
		    print $fp "$stack[$i]{$FILENAME_KEY} ";
		    print $fp "$stack[$i]{$NUM_CHAN_KEY} ";
		    printf $fp ("%.6f %.6f %s\n", $1, $2-$1, $3.$4);
		}
		elsif ($num_words[$j] =~ /(\d+\.\d+)\s+(\d+\.\d+)\s+(<[^<>]+>\S+<[^<>]+>)/) {
		    print $fp "$stack[$i]{$FILENAME_KEY} ";
		    print $fp "$stack[$i]{$NUM_CHAN_KEY} ";
		    printf $fp ("%.6f %.6f %s\n", $1, $2-$1, $3);
		}
		elsif ($num_words[$j] =~ /(\d+\.\d+)\s+(\d+\.\d+)\s+(<[^<>]+>)(\S+)/) {
		    print $fp "$stack[$i]{$FILENAME_KEY} ";
		    print $fp "$stack[$i]{$NUM_CHAN_KEY} ";
		    printf $fp ("%.6f %.6f %s\n", $1, $2-$1, $3.$4);
		}
		elsif ($num_words[$j] =~ /(\d+\.\d+)\s+(\d+\.\d+)\s+(.+$)/) {
		    print $fp "$stack[$i]{$FILENAME_KEY} ";
		    print $fp "$stack[$i]{$NUM_CHAN_KEY} ";
		    printf $fp ("%.6f %.6f %s\n", $1, $2-$1, $3);
		}
	    }
	}
    }
}

sub MAX{
    local(@a) = @_;
    local($max) = -99e9;

    while ($#a >= 0){
	$max = $a[0] if ($max < $a[0]);
	splice(@a,0,1);
    }
    $max;
}


sub MIN{
    local(@a) = @_;
    local($min) = 99e9;

    while ($#a >= 0){
	# print "  $a[0]\n";
	$min = $a[0] if ($min > $a[0]);
	splice(@a,0,1);
    }
    $min;
}

#### tokenize a text stream by whitespace while respecting sgml tags
sub ie_split{
    local($text) = @_;
    local(@a) = ();
    local($i);
#    ALD FIX
#    @a = split(/(\s+|">|<\/)/,$text);
#    for ($i=0; $i<=$#a; $i++){
#        if ($a[$i] =~ /^[^<]*<[^>]*$/){
#            $a[$i] .= splice(@a,$i+1,1);
#            $i--;
#        }
#    }
    # "
    # tag regexp
    my $t = "<(?:(?:\\\"[^\\\"]*\\\")|[^<>])*>";
    # cdata regexp
    my $d = '[^ <>]+';
    @a = ($text =~ m(
		     $t
		     |
		     $d(?:$t$d)+
		     |
		     $d
		     )xiog
	  );
    @a;
}
