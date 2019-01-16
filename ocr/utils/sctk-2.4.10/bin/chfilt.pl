#!/usr/bin/env perl

### This changes the function of a runtime warning to dump a stack trace
#use Carp ();  local $SIG{__WARN__} = \&Carp::cluck;
### This dies on a warning after the stack dump
#use Carp ();  local $SIG{__WARN__} = sub { Carp::cluck(); die; };
#use Carp ();  local $SIG{__DIE__} = sub { Carp::cluck(); die; };

use strict;

# File:  chfilt
# History:  
#    version 1.0  Released 950921
#    version 1.1  Released 951006
#        - For Spanish, it now converts a set of predefined words to 
#          interjections.
#        - Corrected the order of deleting "--" from the transcripts
#        - Normalized improperly formatted words in (( )) markers
#    version 1.2  Released 951025
#        - The "add_spaces_for_japanese_punctuation" function was splitting
#          words that contained vowel elongation characters
#        - Japapese quotation characters are now removed
#    version 1.3 Released 960418
#        - changed the exit codes to be 0 for successful completion, or
#          1 otherwise,  (thanks Barb).
#        - added English support	 
#        - modified the main loop to first read in the entire file, then normalize
#          it, by deleting empty lines and concatenating continued lines together.
#        - added the -i option
#        - added the translation flag option "-t"
#        - added the option -I to not make alternations for interjection words.
#        - added the equivalence class option "-q"
#        - added the hyphenated word alternation flag "-a"
#    version 1.4 Released 970711
#        - added a the pem output format
#        - added the -u option
#        ????? is there an unresolved issue for Japanese ????
#    version 1.5 Release 971010
#        - Spanish: installed hesitation filter, removing interjections.
#        - German:  installed hesitation filter
#        - Arabic:  updated hesitation filter for *romanized script*
#        - removed the -i, -u and -I flags
#        - removed all code that made altenations with NULL
#        - added the '-d' optionally deletable flag
#    version 1.6 Released 
#        - For Arabic scoring, translated the arabic script for an apostrophy to a hyphen
#    version 1.7 Released April 7, 1998
#        - fixed a problem with Out-of-Language markers within an unintelligible markers
#        - fixed a problem with unintelligible markers within an OOL markers
#        - added a filter to remove empty optionally deletable flags.
#  Version 1.10
#       JGF updated to new version or perl 
#  Version 1.11 
#       JGF added support for Miss. State transcript
#       JGF modified the PEM output to pass through the duration times as strings
#  Version 1.12
#       JGF commented out part of the MState code that converted 'OH' and 'AW' to hesitations
#       JGF Correctly handle possessive ancronyms in MState transcripts
#  Version 1.13
#       JGF Added processing for spoken letters.  Previously they did not exist.
#       JGF Added code to remove '^' for proper names
#       JGF Corrected the line header extraction code to handle empty transcripts
#       JGF Added -i and -k options
#       JGF Added code to remove 'aside' sgml tags
#       JGF Added -p to pad segment times with silence and -s to smooth segmentations
#       JGF Added -E to exclude overlapping speech          
#  Version 1.14
#       JGF Added -c to read contractions
#       JGF disabled all old arabic character-specific processing
#       JGF Uses Getopt::Long now tp process the command line arguments
#       JGF Modified the output for stdout to be "-- -".  The new Getopt
#           is interpreting '-' as an ambiguous arguement.
#  Version 1.15
#       JGF added Arabic processing steps for 2004 arabic transcripts.  See the Eval
#           plan appendix for Arabic.
#       JGF add use strict pragma
#  Version 1.16
#       JGF added code to check for unannotated, non-posessive contractions IFF contractions
#           are to be processed.
#  Version 1.17
#       JGF Added -C
#  Version 1.18
#       JGF Added two new hesitations

my $Version="1.18";

my $usage = "Usage: chfilt <OPTIONS> infile outfile|-- -\n".
"Version: $Version\n".
"\n".
"Desc:  chfilt converts a Callhome transcription file, 'infile', to STM\n".
"       format, written to 'outfile'.  If '-- -' is used as output, the output goes to stdout.\n".
"\n".
"       OPTIONS :==\n".
"       -l <language>  Identifies the language on the transcript\n".
"                      Current chfilt can handle:  mandarin\n".
"                                                  spanish\n".
"                                                  japanese\n".
"                                                  english\n".
"                                                  arabic\n".
"                                                  german\n".
"       -m             Read Miss. State Transcripts\n".
"       -b b_time      Output segments After 'b_time'.\n".
"       -e e_time      Output segments ending before 'e_time'.\n".
"       -d             Output text with optionally deletable tags.\n".
"       -q Word1=Word2\n".
"                      Add alternations to the output such that Word1 and Word2\n".
"                      are equivalent\n".
"       -t Word1=Word2\n".
"                      Translate Word1 to Word2\n".
"       -a             Convert all hyphenated word to alternations whereby the \n".
"                      two alternations are the word with hyphens deleted, or\n".
"                      the word(s) with hyphens converted to spaces\n".
"       -i             Add inter-segment gaps to the stm records.  This option\n".
"                      requires -b and -e to be used\n".
"       -k             Keep lines with no lexemes in the transcript rather than\n".
"                      ignoring them\n".
"       -o fmt         Set the output format to: \n".
"                        stm  -> Output STM format (default)\n".
"                        pem  -> Output PEM format\n".
"       -s time        Smooth the segmentation (i.e., merge segments) that hava gap\n".
"                      intersegment gaps less than 'time'\n".
"       -p time        Pad segment boundaries with 'time' seconds of silence if\n".
"                      the time is available\n".
"       -c             Don't expand the contractions if they are annotated\n".
"       -C (numbers|letters)\n".
"                      If 'numbers', change channels A and B to 1 and 2 respectively \n".
"                      If 'letters', change channels 1 and 2 to A and B respectively \n".
"       -E             Exclude overlapping segments in the output STMs\n";;

##########3  Globals 
my $Language;
my $OptDel;
my ($NewBeginTime, $NewEndTime);
my $Format; 
my $KeepEmptyLines;
my @Translate;
my @Equivs;
my $AltHyphens;
my $ReadMSstate;
my $ExcludeOverlap;
my $PadSilence;
my $SmoothSegments;
my $ExpandContractions;
my $AddInterSegGap;
my $HES;
my $InFile;
my $OutFile;
my $id;
my %TranslateAArray;
my %EquivsAArray;
my @Trans;
my $ChannelsTo = "asis";

use Getopt::Long;
$Getopt::Long::ignorecase = 0;
my $ret = GetOptions ("s:f", "p:f", "l=s", "b:f", "e:f", "o=s", "t:s", "q:s", "a", "d", "m", "i", "k", "E", "c", "C=s");
die "\n$usage\nError: Failed to parse argements" if (! $ret);
if (defined($main::opt_o)) {
    $Format = $main::opt_o;
    if($Format !~ /^(stm|pem)$/){
        print "\n$usage\nError: Unrecognized format type --> $Format\n\n";
        exit 1;
    }
} else {
    $Format = "stm";
}
if (!defined($main::opt_l)){
    print "\n$usage\nError: Language option is required\n\n";
    exit 1;
} else {
    $Language = $main::opt_l;
    if ($Language !~ /japanese/ &&
	$Language !~ /mandarin/ &&
	$Language !~ /english/ &&
	$Language !~ /spanish/ &&
        $Language !~ /arabic/ &&
        $Language !~ /german/) {
	print "\n$usage\nError: Language not recognized\n\n";
	exit 1;
    }
}

if (defined($main::opt_b)) { $NewBeginTime = $main::opt_b;            } else { $NewBeginTime = 0.0}
if (defined($main::opt_e)) { $NewEndTime   = $main::opt_e;            } else { $NewEndTime   = 999999;  }
if (defined($main::opt_k)) { $KeepEmptyLines= $main::opt_k; } else { $KeepEmptyLines = 0; }
if (defined($main::opt_t)) { @Translate    = split(/:/,$main::opt_t); } else { @Translate = (); }
if (defined($main::opt_q)) { @Equivs       = split(/:/,$main::opt_q); } else { @Equivs = ();    }
if (defined($main::opt_a)) { $AltHyphens   = 1;  $main::opt_a = 1; } else{ $AltHyphens = 0;$main::opt_a=0;}
if (defined($main::opt_m)) { $ReadMSstate  = 1;  $main::opt_m = 1; } else{ $ReadMSstate = 0;$main::opt_m=0;}
if (defined($main::opt_E)) { $ExcludeOverlap = $main::opt_E } else{ $ExcludeOverlap = 0}
if (defined($main::opt_p)) { $PadSilence   = $main::opt_p } else{ $PadSilence = -1.0}
if (defined($main::opt_s)) { $SmoothSegments = $main::opt_s } else{ $SmoothSegments = -1.0}
if (defined($main::opt_c)) { $ExpandContractions = !(defined($main::opt_c))} else{ $ExpandContractions = 1}
if (defined($main::opt_C)) { 
    die "Error: Unknown channel conversion '$main::opt_C'" if ($main::opt_C !~ /^(numbers|letters)/);
    $ChannelsTo = $main::opt_C;
}
$AddInterSegGap = defined($main::opt_i);
die "$usage\nError: -i requires -b and -e to be used"
    if (defined($main::opt_i) && (! defined($main::opt_b) || !defined($main::opt_e)));	

if (defined($main::opt_d)) {
    $OptDel       = 1;  $main::opt_d = 1;
    $HES = " (%HESITATION) ";
} else {
    $OptDel = 0; $main::opt_d = 0;
    $HES = "%HESITATION";
}

#### The main functions arguements:
if ($#ARGV > 1) { print "\n$usage\nToo many arguements \n\n"; exit 1; } 
if ($#ARGV == 0) { print "\n$usage\nOutput Not Specified\n\n"; exit 1; } 
if ($#ARGV == -1) { print "\n$usage\nInput and Output Not Specified\n\n";
		    exit 1; } 

$InFile=$ARGV[0];
$OutFile=$ARGV[1];

if (! -r $InFile){
    print "\n$usage\nInput file $InFile does not exist\n\n";
    exit 1;
}

if ($OutFile eq "-"){
    $OutFile = "";
} else {
    $OutFile = " > ".$OutFile;
}

($id=$InFile) =~ s:^.*/::;
$id =~ s/\.[^\.]*$//;

# Check the translation classes.  If they exist, make an associative
# array to house them
if ($#Translate >= 0){
    foreach my $tlate (@Translate){
	$tlate =~ tr/a-z/A-Z/;
	if ($Language eq "german" || $Language eq "spanish") {
	    $tlate =~ tr/\340-\377/\300-\337/;     #Upper-Case accented letters
	}
	my @tl = split(/=/,$tlate);
	$TranslateAArray{$tl[0]} = $tl[1];
    }
}

# Check the equivalence classes.  If they exist, make an associative
# array to house them
if ($#Equivs >= 0){
    foreach my $equiv (@Equivs){
	$equiv =~ tr/a-z/A-Z/;
	if ($Language eq "german" || $Language eq "spanish") {
	    $equiv =~ tr/\340-\377/\300-\337/;     #Upper-Case accented letters
	}
	my $alt;
 	($alt = $equiv) =~ s/^/{ /;
	 $alt           =~ s/$/ }/;
	 $alt           =~ s:=: / :g;
	foreach my $eq (split(/=/,$equiv)){
	    $EquivsAArray{$eq} = $alt;
	}
    }
}

open(FILE,"< $InFile") || die("cannot open input Callhome file $InFile"); 
#binmode FILE;

my $hubscr_path = "../hubscr/hubscr.pl";
die "ERROR: 'hubscr.pl' cannot be found, please edit 'chfilt.pl line 267 and set hubscr.pl path."if(! -e $hubscr_path);

#open(OUTPUT,"| sort -t \\   +0  -1  +1  -2 +3nb -4 $OutFile") ||
open(OUTPUT,"| perl $hubscr_path sortSTM $OutFile") || 
    die("cannot open output file $OutFile"); 

## make a place holder for the LUR reports
print OUTPUT ';; CATEGORY "0" "" ""'."\n";
print OUTPUT ';; LABEL "O" "Overall" "All Segments in the test set"'."\n";

# read in the file, deleting blank lines:
while (<FILE>){
    s/\[\[[^]]*\]\]//g;            #remove comments              '[[words]]'

    if ($_ !~ /^\s*$/ && $_ !~ /^#/){
	if ($ReadMSstate){
	    s/^\S+\s+//;
	}
	push(@Trans,$_);
    }
}
close (FILE);

# normalize the transcript so that each turn is on a line by itself
for (my $n=0; $n<=$#Trans; $n++){
    if ($Trans[$n] =~ /^\s*\d+\.\d*\s+\d+\.\d*\s+[AB]\d*:/){
	my $orig = $Trans[$n];
	if ($ExpandContractions){
	    ### Scan the transcript looking for un-annotated non-possessive contractions
	    my $ct = $Trans[$n];	    
	    ### delete all tokens annotated as contractions
	    $ct =~ s/(<contraction\s+e_form=\"([^\"]+)\">([^ \t\n\r\f\)]+))//gi;
	    foreach my $ctt(split(/[\s\,\.\?]+/,$ct)){
		print STDERR "Warning: Possible non-annotated contraction \"$ctt\"\n" if ($ctt =~ /\w'\w+/ && $ctt !~ /\w's$/);
	    }
	}
	while ($Trans[$n] =~ /(<contraction\s+e_form=\"([^\"]+)\">([^ \t\n\r\f\)]+))/i){
	    my ($tag, $rep, $token) = (quotemeta($1), $2, $3);
	    my $exp;
	    ($exp = $rep) =~ s/\[[^\[=]+=>/ /g;
	    $exp =~ s/\]//g;
	    $exp =~ s/^\s+//g;

#	    print "Found $tag $rep $token '$exp' ------------ in $Trans[$n]";
	    if ($ExpandContractions){
		$Trans[$n] =~ s/$tag/$exp/;
	    } else {
		$Trans[$n] =~ s/$tag/$token/;
	    }	    
	    die "Error: unable to expand the contraction\n" if ($orig eq $Trans[$n]);
	    $orig = $Trans[$n];
	}
	die "Error: Contraction not handled $Trans[$n]" if ($Trans[$n] =~ /<contrac/);
    } else {
#	print "no match $Trans[$n]";
	if ($n == 0){
	    die("Illegal format, first line does not have a begin time, end time and channel");
	}
	chop($Trans[$n - 1]);
	$Trans[$n - 1] = $Trans[$n - 1]." ".$Trans[$n];
	splice(@Trans,$n,1);
	$n--;
    }
}   

#print @Trans;

my @STMS = ();

### Normalize the text 
foreach $_ (@Trans){
    #match the lines header
    chomp;
    die "Unable to strip the line header of '$_'" unless s/^(.*[AB][0-9]*):\s*//;
    my $line_header = $1;             #Store the header in a variable

    tr/\011/ /;                    #change all tabs to spaces
    s/ /  /g;                      #multiple spaces go to one
    s/^/ /;                        #add spaces at begining of line
    s/$/ /;                        #put a space at the end of each line
    s/\s\(\(/ (( /g;                 #add spaces around unintelligble markers
    s/\)\)\s/ )) /g;                 #add spaces around unintelligble markers
    if ($OptDel == 0){
        s:<foreign\s+language=[^>]+>([^><]+)</foreign>:$1:gi;
    } else {
        s/\(\(\s*<foreign\s+language=\"[^"]+\">([^><]+)<\/foreign>\s*\)\)/\(\($1\)\)/gi; 
        s/<foreign\s+language=\"[^"]+\">([^><]+)<\/foreign>/\(\($1\)\)/gi;
   } 
#print "$_ here\n";
    if ($Language ne "arabic") {
       tr/a-z/A-Z/;                #Upper-Case all letters
       if ($Language eq "german" || $Language eq "spanish") {
	   tr/\340-\377/\300-\337/     #Upper-Case accented letters
       }
    }
    s/\{[^{}]*\}//g;               #remove non-speech vocal noises         '{words}'

    ### LANGUAGE Specific Normalizations
    if ($Language eq "arabic") { 
	my $hes;
	if ($OptDel == 0){
	    $hes="%\330\252\330\261\330\257\331\221\330\257";
	} else {
	    $hes=" (%\330\252\330\261\330\257\331\221\330\257) ";
	}
        s/[% ]\330\243\331\207 /$hes /g;           ## %>h
        s/[% ]\330\245\331\212\331\207 /$hes /g;   ## %<yh
        s/[% ]\330\243\331\205 /$hes /g;           ## >m
        s/[% ]\330\243\331\210\331\210 /$hes /g;   ## >ww
        s/[% ]\331\207\330\247\331\212 /$hes /g;   ## hAy
        s/[% ]\331\205\331\207\331\205 /$hes /g;   ## mhm
     
        s/%*(\330\243\331\207\331\207) /$1 /g;         ## Back Channel
        s/%\330\243\330\254\331\206\330\250\331\212 / /g;     ##  Foreign lexeme
        s/%\330\252\330\257\330\247\330\256\331\204.*%\330\252\330\257\330\247\330\256\331\204\\ / /g;   ## End cross channel

        ### Remove non speech markers
        s/%\330\265\331\205\330\252\\* / /g;    # silence
        s/%\330\245\331\206\331\202\330\267\330\247\330\271\\* / /g; # pause symbol
        s/%\330\266\330\254\331\221\330\251\\* / /g;                 # noise symbol
        s/%\330\252\331\206\331\201\331\221\330\263\\* / /g;         # breath symbol
        s/%\330\263\330\271\330\247\331\204\\* / /g;                 # cough
        s/%\330\266\330\255\331\203\\* / /g;                         # laugh
        s/%\330\243\330\265\331\210\330\247\330\252\\* / /g;         # people talk

#	## The LDC Transcripts have a special case encoded in them for
#	## foreign words with an attached definite article.  The next few commands
#	## move the definite article into the foreign tag.
#	s/\s(\S+)(<\S+)\s+(\S+)/ $2 $1$3/g;
#	## translate the arabic signle apostrophies to hyphens
#	s/\254\s/- /g;
#	s/\254$/- /g;
#        s/\s\254/ -/g;
#	## Arabic Semi colon
#	s/\s\273/ /g;
    } elsif ($Language eq "english") { 
	;
    } elsif ($Language eq "japanese") {
	$_ = &add_spaces_for_japanese_punctuation($_);
	s/\241\242 /, /g;          # Change the Japanese coma
	s/\241\243 /. /g;          # Change the Japanese periods
	s/\241\244 /, /g;          # Change the Japanese periods
	s/\241\250 /; /g;          # Change the Japanese semi-colon
	s/\241\251 /? /g;          # Change the Japanese question marks
	s/\241\252 /! /g;          # Change the Japanese exclaimation marks
	s/\241\327 / /g;           # Change the Japanese quotation mark
	s/ \241\326/ /g;           # Change the Japanese quotation mark
	s/\241\343([\000-\177])/<$1/g;      # Change the Japanese '<'
	s/([\000-\177])\241\344/$1>/g;      # Change the Japanese '>'	 
	# Convert Japanese col and dia flags to alternations
	s/@([^@\[\]]+)\[\[([^,\[\]]+)\s*,[^\[\]]+\]\]/ { $1 \/ $2 } /g;
    } elsif ($Language eq "mandarin") {
	while (s/(<[^_>]*)_/$1 /) { # Removes underlines from OOL items
	    ;
	}
	s/@([^@]+)@/$1/g;           # Removes obviously the @'s !!!
	s/\*([^\*]+)\*/$1/g;           # Removes obviously the *'s !!!
    }

    ### Call Home filtering (General)

    s/\-\-//g;                     #whatever they are??
    s/[.,?!;]//g;                   #Remove normal punctuation
    s/"//g;                        #Remove quotation marks
    s/\#/ /g;                      #remove simultaneous speech markers     '#'
    s/\[\[[^]]*\]\]//g;            #remove comments              '[[words]]'
    s/\[[^]]*\]//g;                #remove other sounds          '[words]'
    s/\/\///g;                     #remove speech to someone else markers  '//'
    s/<\/*aside\/*>//g;            #remove aside sgml tags

    if ($ReadMSstate){
        s/\^(\S)/$1/g;              #remove proper name markers? '^'
        s/\*(\S)/$1/g;              #remove ??? name markers? '*'
        s/\@(\S)/$1/g;              #remove ??? name markers? '@'
        s:<as/>[^<>]+</as>::ig;   #remove asides
        while ($_ =~ /~/){          # Deal with acronyms
            s/~(\S) /$1 /;
            s/~(\S\'[sS]) /$1 /;       # Deal with a possessive acronym
            s/~(\S)(\S)/$1 ~$2/;
        } 
    }

    s/\(\(\s*\)\)//g;              #remove unintelligable speech with no words

    ## Changes from new LDC transcription practices (Jan 2003)_
    s/~([A-Z-a-z])/$1./g;          #Change the spoken letters to the Hub4 form
    s/\^//g;                        #Remove proper names


    if ($OptDel == 1){
        s/\(\s+/(/g;                 #Normalize the variablity in the LDC trans
        s/\s+\)/)/g;                 #Normalize the variablity in the LDC trans
        my $vb = 0;
        my $iteration = 1;
        print "Start $_\n" if ($vb);
        my $loop_end = "xxxxxxxxxxxxxxxxx";

        my $plainWord = "[^ \t\n\r\f\(\)\{\}]+";
        my $optDelWord = "\\([^ \t\n\r\f\(\)\{\}]+\\)";
#        my $wordSet = "($plainWord|$optDelWord)(\s+($plainWord|$optDelWord))*";

        while ($_ =~ /\(\(.*\)\)/) {
            # if the unintell is out side of a OOL marker, just delete them
            s/\(\((\s*<[^>]+>\s*)\)\)/$1/;
            print "      Step 1 $_\n" if ($vb);

            #  handle unintelligle markers within OOL markers
            s/\(\((\s*<[^>]+>\s*)([^\)]*)\)\)/$1 \(\($2\)\)/;
            print "      Step 2 $_\n" if ($vb);

	    #Change 1st unintelligable word, if its already optionally deletable
            s/\(\(\s*($optDelWord)\s+(($plainWord|$optDelWord)(\s+($plainWord|$optDelWord))*\s*\)\))/ $1 \(\($2/g;
            print "      Step 3 $_\n" if ($vb);

	    #Change 1st unintelligable word, if its an alternate
            s:\(\(\s*({[^{}]+})\s+(($plainWord|$optDelWord)(\s+($plainWord|$optDelWord))*\s*\)\)): \($1\) \(\($2:g;
            print "      Step 4 $_\n" if ($vb);

	    #Change 1st unintelligable word, if its a plain word
            s/\(\(\s*($plainWord)\s+(($plainWord|$optDelWord)(\s+($plainWord|$optDelWord))*\s*\)\))/ \($1\) \(\($2/g;
            print "      Step 5 $_\n" if ($vb);

            ## Change a single optionally deletable word
            s/\(\(\s*($optDelWord)\s*\)\)/ $1 /g;
            print "      Step 8 $_\n" if ($vb);

            #alternate single unintelligable with alternate
            s/\(\(\s*({[^{}]+})\s*\)\)/ \($1\) /g;
            print "      Step 9 $_\n" if ($vb);

            #alternate single unintelligable words
            s/\(\(\s*($plainWord)\s*\)\)/ \($1\) /g;
            print "      Step A $_\n" if ($vb);

	    # remove empty unintellegible markers
	    s/\(\(\s*\)\)//g;
            $iteration ++;
	
	    if ($_ eq $loop_end){
	         print STDERR "Warning: unable to fully alternate an unitelligible marker\n".
	                      "         $_\n";
	         last;
	    }
            if ($iteration > 100){
	         die "Unable to alternate an unitelligible marker '$_'\n";
            }
            # Remember what the last line was so that We don't get it an infinite loop
            $loop_end = $_;
            print "   End $_\n" if ($vb);
        }
    } else {
#        print "Delete (( $_\n";
        s/\s\(\(/ /g;                 #Strip the unintelligible markers
        s/\)\)\s/ /g;                 #Strip the unintelligible markers
#        print "          $_\n";
    }

    if (! $OptDel){
        s/<\s*\S+\s+([^<>]*)>/$1/g;  #remove out-of-language speech markers
        s/<[^<>]*>//g;                #remove out-of-language speech markers
    } else {
        while ($_ =~ /(<\s*\S+\s+)([^<>]*)>/) {
            my ($otxt, $head) = ($2, $1);
            my $txt = $otxt;
            $txt =~ s/^\s+//;;
            $txt =~ s/\s+$//;;
            my $new= "";
            foreach my $w(split(/\s+/,$txt)){
                if ($w !~ /^\(.+\)$/){
                    if ($new eq ""){
                        $new = "(".$w.")";
                    } else {
                        $new .= " (".$w.")";
                    }
                } else {
                    $new .= " $w";
                }
            }
            if ( ($_ =~ s/${head}[^>]*>/$new/) != 1){
                die("OptDel substitution of non-language marker failed\n".
                    "    text: $_    head: '$head'\n    otxt: '$otxt'\n    new: '$new'".
                    "\n    merge: '${head}${otxt}>'");
            }
        }
        ### delete noise tags.  These are new as of 1/20/2004
        s/<[^ \t\n\r\f<>]+>//g;
   }

    ### Language Specific Filters

    # SPANISH
    if ($Language eq "spanish") {
        y/\240-\277//d;	             #remove spanish punctuation
        s/\*\*//g;		     #words marked with "**" (as neologisms or unknown
                                     # 'words')
        s/\+//g;		     #words marked with "++" (mispronunciations given
                                     # in correct orthography)

        # map the list of hesitiation's
        s:%::g; # remove all hesitation markers (maybe illegal ones!)
        # insert all hesitation markers (maybe missed ones!)
        s/ AAA / $HES /g;
        s/ EEE / $HES /g;
        s/ III / $HES /g;
        s/ MMM / $HES /g;
        s/ EMM / $HES /g;
        s/ AMM / $HES /g;
        s/ IMM / $HES /g;
        s/ EH / $HES /g;
        s/ MMH / $HES /g;
        s/ OH / $HES /g;
        s/ UH / $HES /g;
        s/ EY / $HES /g;
        s/ UY / $HES /g;
        s/ PSS / $HES /g;
        s/ HM / $HES /g;
        s/ UF / $HES /g;
        s/ SHH / $HES /g;
        s/ PFF / $HES /g;
        s/ OY / $HES /g;
        s/ HA / $HES /g;
        s/ AH / $HES /g;
     }

     # GERMAN
     if ($Language eq "german") {

        s/\&//g;	 		 #remove proper name marker
        s/\+//g;		     #words marked with "++" (mispronunciations given
                                     # in correct orthography)
         s/\*\*//g;		     #words marked with "**" (as neologisms
                                     # or unknown 'words')
        
        s:%::g; # remove all hesitation markers (maybe illegal ones!)
        # insert all hesitation markers (maybe missed ones!)

        s/ ÄH / $HES /g;
        s/ MM / $HES /g;
        s/ ÄHM / $HES /g;
        s/ HM / $HES /g;
        s/ HA / $HES /g;
        s/ EI / $HES /g;
        s/ UH / $HES /g;
        s/ HÄ / $HES /g;
        s/ HO / $HES /g;
        s/ UFF / $HES /g;
        s/ OI / $HES /g;
        s/ HUH / $HES /g;
        s/ BAH / $HES /g;
        s/ UI / $HES /g;
     }


     # ARABIC
     if ($Language eq "arabic") {

        s/\&//g;	 		 #remove proper name the marker
        s/\s\+(\S+)\+\s/ $1 /;           #Remove the plus markers
#        s:%\S+\s::g; # remove all hesitation markers (maybe illegal ones!)
        s/\*\*//g;		     #words marked with "**" (as neologisms
                                     # or unknown 'words')
        s/$/ /;                      ## Add a space at the end of the line
        s/ /  /;                     ## Add double spaces

        # insert all hesitation markers (maybe missed ones!)
#        s/ ah / $HES /g;
#        s/ E / $HES /g;
#        s/ M / $HES /g;
#        s/ mhm / $HES /g;
#        s/ ha / $HES /g;
#        s/ uh / $HES /g;
#        s/ yA / $HES /g;
#        s/ aha / $HES /g;
#        s/ yaa / $HES /g;
#        s/ hm / $HES /g;
#        s/ hum / $HES /g;
#        s/ Ah / $HES /g;
#        s/ ayyO  / $HES /g;
#        s/ Eyy / $HES /g;
#        s/ yuu / $HES /g;
#        s/ ih / $HES /g;
#        s/ yO / $HES /g;
#        s/ O / $HES /g;
#        s/ wAw / $HES /g;
#        s/ hi / $HES /g;
#        s/ Hay / $HES /g;
#        s/ yOO / $HES /g;
#        s/ OhO / $HES /g;
#        s/ hE / $HES /g;
#        ### Script equivalences
#        s/ \302\347 / $HES /g;
#        s/ \305\352\347 / $HES /g;
#        s/ \305\352\352\347 / $HES /g;
#        s/ \315\351 / $HES /g;
#        s/ \345 / $HES /g;
#        s/ \307\350 / $HES /g;
#        s/ \307\350\347\350 / $HES /g;
#        s/ \307\347 / $HES /g;
#        s/ \307\347\307 / $HES /g;
#        s/ \307\352\350     / $HES /g;
#        s/ \347\351 / $HES /g;
#        s/ \347\307 / $HES /g;
#        s/ \347\351 / $HES /g;
#        s/ \347\345 / $HES /g;
#        s/ \347\345 / $HES /g;
#        s/ \305\347 / $HES /g;
#        s/ \345\345 / $HES /g;
#        s/ \307\347 / $HES /g;
#        s/ \350\307\350 / $HES /g;
#        s/ \352\307\347 / $HES /g;
#        s/ \352\350 / $HES /g;
#        s/ \352\350\350 / $HES /g;
#        s/ \352\307 / $HES /g;
#        s/ \352\347 / $HES /g;
     }

     # MANDARIN
     if ($Language eq "mandarin") {
         s/\+//g;	 		 #remove non-standard representation marker
         s/\&//g;	 		 #remove some the marker
         s/\243[\254\255] / /g;      # Remove Japanese punctuation

         s/\s%(\S)/ $1/g;          # Delete the all the percent signs at word head
         s/\s\(\%/ (/g;          # Delete the all the percent signs at word head of unclear
         s/%\s/ /g;                  # Delete the all the percent signs at word tail

         # NOTE: Cannot remove all hesitation markers because some "%" are used
         #       in the text (I think)
         #       So we will just filter the known hesitation markers.
         s/\sÚÀ\s/ $HES /g;
         s/\sßÀ\s/ $HES /g;
         s/\sàÅ\s/ $HES /g;
         s/\sßí\s/ $HES /g;
         s/\sºÇ\s/ $HES /g;
     }	
	
     # JAPANESE
     if ($Language eq "japanese") {
         s/\+//g;		    #remove non-standard representation marker
         s/\=/\-/g;                 #change frag= to frag-
         s/\241[\241-\256] / /g;      # Remove Japanese punctuation
     }

     if ($Language eq "english") {
         s/\&//g;	 		 #remove proper name the marker
         s/\*\*//g;		     #words marked with "**" (as neologisms
                                     # or unknown 'words')
         s/\*//g;		     #words marked with "*" idiosyncratic words
                                     # or unknown 'words')
         s/\+//g;		     #words marked with "++" (mispronun-
                                     # ciations given in correct orthography)
         s/@//g;                     # Spoken acronyms
         ### Mapp the hesitations
         s/[\s\(][%]*(UH|UM|EH|MM|AH|HM|HUH|HA|ER|OOF|HEE|ACH|EEE|EW)[\s\)]/ $HES /g;
         if ($ReadMSstate){
#### Commented out because this should be changed in SPINE
####         s/[\s\(][%]*(OH|AW)[\s\)]/ $HES /g;
             s/[\s\(][%]*(MHM)[\s\)]/ $1 /g;
         }
     }

     # This has to be done AFTER ALL THE PROCEEDING STEPS to alternate any fragment words
     if ($OptDel){
         s/(\S+)-(\S+)[-=]\s/ $1 \($2-\) /g;  #put in alternates for fragments    
         s/(\S*)[-=]\s/ \($1-\) /g;  #put in alternates for fragments    
         s/\s[-=](\S*)/ \(-$1\) /g;  #put in alternates for fragments    
     } else {
         s/(\S*)[-=]\s/ $1- /g;  #put in alternates for fragments    
         s/\s[-=](\S*)/ -$1 /g;  #put in alternates for fragments    
     }
     s/ +/ /g;
     if ($#Translate >= 0){	
	$_ = &Map_AArray($_,\%TranslateAArray);
     }	
     if ($#Equivs >= 0){	
	$_ = &Map_AArray($_,\%EquivsAArray);
     }
     if ($AltHyphens) {
        $_ = &AlternateHyphenWords($_);
     }

     ### delete any emtpy optionally deletable words
     if ($OptDel){         s/\(\)//g;  }

     my($bt, $et, $spkr) = split(/\s+/,$line_header);
     my $chan;
     ($chan = $spkr) =~ s/[0-9]$//;
     if (&in_range($bt, $et) && ($KeepEmptyLines || $_ !~ /^\s*$/)){
         if ($Format eq "stm"){
	     push @STMS, sprintf "%s %s %s %s %6.2f <O> %s\n",$id,makeChannel($chan),$id."_".$spkr,$bt,$et,$_; 
         } elsif ($Format eq "pem"){
	     printf OUTPUT "%s %s %s %s %s\n",$id,makeChannel($chan),"unknown_speaker",$bt,$et; 
         }
     }
}

sub makeChannel{
    my ($chan) = @_;
    if ($ChannelsTo eq "numbers") {
	$chan =~ s/a/1/i;
	$chan =~ s/b/2/i;
    } elsif ($ChannelsTo eq "letters") {
	$chan =~ s/1/A/i;
	$chan =~ s/2/B/i;
    } elsif ($ChannelsTo eq "asis") {
	;
    } else {
	die "Error: unknown Channel conversion $ChannelsTo";
    }
    $chan;
}

sub MIN{
    my(@a) = @_;
    my($min) = 99e9;

    while ($#a >= 0){
        # print "  $a[0]\n";
        $min = $a[0] if ($min > $a[0]);
        splice(@a,0,1);
    }
    $min;
}

if ($Format eq "stm"){
    my (%lastStmEnd, %lastStmEndId, %lastStmEndIndex, $nt);
    ### Pad the STMs segments with silence if you got it
    if ($PadSilence >= 0){
        for (my $i=0; $i<@STMS; $i++){
#            print "STM[$i] = $STMS[$i]";
            my ($id, $chan, $spkr, $bt, $et, $lur, $text) = split(/\s+/,$STMS[$i],7);
            if (! defined($lastStmEnd{$chan})){
#                print "Set \$lastStmEnd{$chan} = $NewBeginTime;\n";
                $lastStmEnd{$chan} = $NewBeginTime;
                $lastStmEndId{$chan} = $id;
		$lastStmEndIndex{$chan} = $i;
            }
            if ($lastStmEnd{$chan} < $bt){
                my $shift = MIN(($bt - $lastStmEnd{$chan}) / 2.0, $PadSilence);
                if ($lastStmEndIndex{$chan} < $i){
                    $nt = sprintf("%.2f", $lastStmEnd{$chan} + $shift);
#		    print "fix $nt $STMS[$lastStmEndIndex{$chan}] ";
                    die "Can't fix silence pad time of previous line" unless ($STMS[$lastStmEndIndex{$chan}] =~ s/^(\S+\s+\S+\s+\S+\s+\S+\s+)\S+/$1$nt/);
#		    print "->  $STMS[$lastStmEndIndex{$chan}]\n";
                } 
                $nt = sprintf("%.2f",$bt - $shift);
                die "Can't fix silence pad time" unless ($STMS[$i] =~ s/^(\S+\s+\S+\s+\S+\s+)\S+/$1$nt/);
            } 
            $lastStmEnd{$chan} = $et;
       	    $lastStmEndIndex{$chan} = $i;
#            print "     \$lastStmEnd{$chan} = $et\n";
        }
        foreach my $chan(sort(keys %lastStmEnd)){
            my ($id, $ccchan, $spkr, $bt, $et, $lur, $text) = split(/\s+/,$STMS[$lastStmEndIndex{$chan}],7);
#	    print "End $STMS[$lastStmEndIndex{$chan}]";
            if ($lastStmEnd{$chan} < $NewEndTime){
                my $shift = MIN(($NewEndTime - $lastStmEnd{$chan}) / 2.0, $PadSilence);
                $nt = $lastStmEnd{$chan} + $shift;
                die "Can't fix silence pad time" unless ($STMS[$lastStmEndIndex{$chan}] =~ s/^(\S+\s+\S+\s+\S+\s+\S+\s+)\S+/$1$nt/);
#		print "    end now $STMS[$lastStmEndIndex{$chan}]\n";
            }
        }
    }
    undef %lastStmEnd;
    undef %lastStmEndId;

    ## Smooth the segments 
    if ($SmoothSegments >= 0){
        for (my $i=0; $i<@STMS; $i++){
            my ($id, $chan, $spkr, $bt, $et, $lur, $text) = split(/\s+/,$STMS[$i],7);
            if (! defined($lastStmEnd{$chan})){
                $lastStmEnd{$chan} = $NewBeginTime;
                $lastStmEndId{$chan} = $id;
		$lastStmEndIndex{$chan} = $i;
            }
	    my $change = 0;
            if ($lastStmEnd{$chan} > $bt - $SmoothSegments && $lastStmEndIndex{$chan} < $i){
		my ($lid, $lchan, $lspkr, $lbt, $let, $llur, $ltext) = split(/\s+/,$STMS[$lastStmEndIndex{$chan}],7);
		if ($id eq $lid && $chan eq $lchan && $spkr eq $lspkr && $lur eq $llur){
		    chomp $ltext;
		    chomp $text;
		    $STMS[$lastStmEndIndex{$chan}] = 
			"$id $chan $spkr $lbt $et $lur $ltext $text\n";
		    splice (@STMS, $i, 1);
		    $i --;
		    $change = 1;
		}
            } 
	    $lastStmEnd{$chan} = $et;
	    if (! $change){
		$lastStmEndIndex{$chan} = $i;
	    }
        }
    }
    undef %lastStmEnd;
    undef %lastStmEndId;

    ## Smooth the segments 
    if ($ExcludeOverlap){
        for (my $i=0; $i<@STMS; $i++){
#            print "STM[$i] = $STMS[$i]";
            my ($id, $chan, $spkr, $bt, $et, $lur, $text) = split(/\s+/,$STMS[$i],7);
            if (! defined($lastStmEnd{$chan})){
                $lastStmEnd{$chan} = $NewBeginTime;
                $lastStmEndId{$chan} = $id;
		$lastStmEndIndex{$chan} = $i;
            }
	    my $change = 0;
            if ($lastStmEnd{$chan} > $bt && $lastStmEndIndex{$chan} < $i){
		my ($lid, $lchan, $lspkr, $lbt, $let, $llur, $ltext) = split(/\s+/,$STMS[$lastStmEndIndex{$chan}],7);
		if ($id eq $lid && $chan eq $lchan){
		    chomp $ltext;
		    $STMS[$lastStmEndIndex{$chan}] = 
			"$id $chan excluded_region $lbt $et $lur ignore_time_segment_in_scoring\n";
		    splice (@STMS, $i, 1);
		    $i --;
		    $change = 1;
		}
            } 
	    $lastStmEnd{$chan} = $et;
	    if (! $change){
		$lastStmEndIndex{$chan} = $i;
	    }
        }
    }
    undef %lastStmEnd;
    undef %lastStmEndId;

     ## Add the intersegment gaps
     if ($AddInterSegGap){
         for (my $i=0; $i<@STMS; $i++){
             my ($id, $chan, $spkr, $bt, $et, $lur, $text) = split(/\s+/,$STMS[$i],7);
             if (! defined($lastStmEnd{$chan})){
                 $lastStmEnd{$chan} = $NewBeginTime;
                 $lastStmEndId{$chan} = $id
             }
             if ($lastStmEnd{$chan} < $bt - 0.01){
#            if (0){
                 splice (@STMS, $i, 0, sprintf("%s %s %s %s %6.2f <O> %s\n",$id,$chan,
                                       "${id}_${chan}_".
                                       "inter_segment_gap",
                                       $lastStmEnd{$chan},$bt,""));
                 $i++;
             }
             $lastStmEnd{$chan} = $et;
         }
         foreach my $chan(sort(keys %lastStmEnd)){
             if ($lastStmEnd{$chan} < $NewEndTime - 0.01){
                 push(@STMS, sprintf("%s %s %s %s %6.2f <O> %s\n",$lastStmEndId{$chan},$chan,
                   "${id}_${chan}_inter_segment_gap",
                    $lastStmEnd{$chan},$NewEndTime,"")); 
             }
         }
    }
}
print OUTPUT @STMS;
close OUTPUT;
exit 0;

###############################################################################
####                                                                       ####
####                        END OF MAIN PROGRAM                            ####
####                                                                       ####
###############################################################################

 
####
####  Return true if the passed in begin and end times are within the 
####  specified range of output times.  If not return false.
####  Inputs:   - Argument 0: The begin time
####            - Argument 1: The end time
####  Outputs:  - Return true if:
####              * the global variables $NewBeginTime or $NewEndTime
####              * are set, and the begin and end times are with their
####              * respective ranges
####              OR
####              * neither global variables are defined.
sub in_range{
    my ($bt,$et) = @_;
    my $do_prnt = 0;
    if (defined($NewBeginTime) && defined($NewEndTime)) {
	if (($bt >= $NewBeginTime || &eqdelta($bt,$NewBeginTime,0.5)) &&
	    ($et <= $NewEndTime || &eqdelta($et,$NewEndTime,0.5))) {  
	    $do_prnt = 1;
	}
    } elsif (defined($NewBeginTime)) {
	if ($bt >= $NewBeginTime ||
	    &eqdelta($bt,$NewBeginTime,0.5)) {
	    $do_prnt = 1;
	} 
    } elsif (defined($NewEndTime)) {
	if ($et <= $NewEndTime || &eqdelta($et,$NewEndTime,0.5)) {
	    $do_prnt = 1;
	}
    } else {
	$do_prnt = 1;
    }
    $do_prnt;
}


####
####  Compare two real numbers, they're equal if the absolute value of the
####  difference is less than delta
####  Inputs:  - Argument 0: real number 1.
####           - Argument 1: real number 2.
####           - Argument 2: The delta.
####  Outputs: - True if they are equal, False if they are not
sub eqdelta {
    my($v1,$v2,$del) = @_;
    if ($v1 < $v2) {
        $v2 - $v1 < $del;
    } else {
        $v1 - $v2 < $del;
    }
}

sub add_spaces_for_japanese_punctuation {
    $_ = $_[0];
    chomp;
    my $first = 0;
    my $c1 = "";
    my $lc = "";
    my $out = "";
    
    while (length($_) > 0){
	s/^(.)//;
	$c1 = $1;
	if ($c1 =~ /^[\177-\377]/){
	    if (!$first) { $first = 1; } else { $first = 0; }
	}
	$out = $out.$c1;
	if (!$first && ($lc.$c1) =~ /^\241[\241-\270\272\273\275-\377]$/) {
	    $out = $out." ";
	}
	$lc = $c1;
    }
    $out."\n";
}

sub Map_AArray{
    $_ = $_[0];
    my $AArray = $_[1];
    my $out = "";
    my $word = "";
    foreach $word (split){
	if ($AArray->{$word} !~ /^$/){
	    $out .= " ".$AArray->{$word};
	} else {
	    $out .= " ".$word;
	}
    }
    $out .= "\n";
    $out;
}

sub AlternateHyphenWords{
    $_ = $_[0];
    my $out = "";
    my $word = "";
    my $x;
    my $y;
    my $z;
    foreach $word (split){
	if ($word =~ /[A-Za-z][_-][A-Za-z]/){
	    ($x=$word) =~ s/[_-]//g;
	    ($y=$word) =~ s/[_-]/ /g;
	    ($z=$word) =~ s/[_-]/-/g;
	    $word = "{ ".$y." / ".$x." / ".$z." }"; 
	}
	$out .= " ".$word;
    }
    $out .= "\n";
    $out;
}
