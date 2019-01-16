#!/usr/bin/perl -w

$Version="1.3";

#####
#  Version 1.0  Released September 9, 1997
#        - Initial Release
#  Version 1.1  Released October 14, 1997
#        - Added the ability to perform the operation on Arabic script
#          This requires the LDC Arabic dictionary
#  Version 1.2 
#        - Fixed a coding bug, there was an incorrect 'if ($Format = "Script");'
#  Version 1.3 
#        - Fixes to avoid undefined hash lookup errors

$Usage="Usage: def_art.sh [ -i fmt ] [ -s LDC_LEX ] -t Infile|- OutFile|-\n".
"Version: $Version\n".
"Desc: Detaches the arabic definite article.  This script was donated by BBN".
"Options:\n".
"      -i fmt   Set the input file formant to 'fmt'.  The possible choices are:\n".
"                  txt -> plain text, the default\n".
"                  ctm -> CTM format, ignores all but the 5th column, and if\n".
"                         a division occurs and a confidence score is present,\n".
"                         the confidence score is copied to all parts.\n".
"                  stm -> STM format, change only the text field of the stm record\n".
"      -s LDC_LEX\n".
"               Perform the operation on arabic script transcripts.  This requires\n".
"               the LDC Arabic Lexicon.  The script uses the lexicons morphological\n".
"               tags to decided when to seperate the definate article.\n".
"      -t       If -s is used, Translate any token that match entries a romanized\n".
"               orthography to it's script form\n".
"\n";

$Format="Romanized";
$Lexicon="";
$Translate = 0;
%DefArt_LUT = ();
%Script_LUT = ();

use Getopt::Std;

getopts("i:s:t");

if (defined($opt_i)) {
    die("$Usage\n\nError: Undefined input format '$opt_i'") 
	if ($opt_i !~ /^(txt|ctm|stm)$/);
    $InFmt = $opt_i;
}
if (defined($opt_s)){
    $Format="Script";
    $Lexicon=$opt_s;
    die("$Usage\n\nError: Unable to read lexicon file '$Lexicon'") 
	if (! -r $Lexicon);
}
if (defined($opt_t)){
    die("$Usage\n\nError: -t requires -s as well")
	if (!defined($opt_s));
    $Translate = 1;
    $opt_t = 1;
}

#### The main functions arguements:
die "$Usage\nToo many arguements" if ($#ARGV > 1);
die "$Usage\nOutput Not Specified" if ($#ARGV == 0);
die "$Usage\nInput and Output Not Specified" if ($#ARGV == -1);

$InFile=$ARGV[0];
$OutFile=$ARGV[1];
die("$Usage\nError: Input file $InFile does not exist\n")
    if ($InFile ne "-" && ! -r $InFile);

#### 
&Load_Lexicon($Lexicon) if ($Format eq "Script");

open(IN, "$InFile") || die "Unable to open trans-file $InFile";
open(OUT, ">$OutFile") || die "Unable to open new-trans-file $OutFile";

while (<IN>){
    chop;
    if ($InFmt eq "txt"){
	print OUT &String_LU($_)."\n";
    } elsif ($InFmt eq "ctm"){
	s/^\s+//;
	local(@ctm);
	local(@new_words);
	local($i);	    
	local($newt);
	local($conf);

	@ctm = split(/\s+/,$_);
	if ($#ctm <= 4) { $conf = "" ; } else { $conf = $ctm[5]; }

	$newt = &String_LU($ctm[4]);
	@new_words = split(/\s+/,$newt);
	for ($i=0; $i<=$#new_words; $i++){
	    printf OUT ("%s %s %.2f %.3f %s %s\n",$ctm[0],$ctm[1],
			$ctm[2] + ($i * ($ctm[3] / ($#new_words+1))),
			$ctm[3] / ($#new_words + 1), $new_words[$i], $conf);
	}
    } elsif ($InFmt eq "stm"){
	s/^\s+//;
	local($file, $chan, $spk, $bt, $et, $lab_txt) = split(/\s+/,$_." ENDOFLN",6);
	local($head);
	$lab_txt =~ s/\s*ENDOFLN$//;
	if ($lab_txt =~ /^$/){
	    $lab = "";
	    $txt = "";
	    $head = "$file $chan $spk $bt $et";
	} elsif ($lab_txt =~ /^(<[^<>]*>)(.*)$/){
	    $lab = $1;
	    $txt = $2;
	    $head = "$file $chan $spk $bt $et $lab";
	} else {
	    $lab = "";
	    $txt = $lab_txt;
	    $head = "$file $chan $spk $bt $et";
	}
	print OUT $head." ".&String_LU($txt)."\n";
    }
}

close IN; close OUT;
exit 0;

#########  SUBROUTINES  ###################

sub String_LU {
    local($v) = @_;

    unless (/^\s*$/) {
	if ($Format eq "Romanized"){
	    # replace "il+" using the solar/lunar rules which state
	    # that "l+" in "il+" is replaced by the first letter after
	    # the "+" sign if that letter is one of the following:
	    # t,g,d,r,z,s,$,S,D,T,Z,n,j,k and keep em seperate.
	    $v =~ s/il\+([&]{0,1})([tgdrzs\$SDTZnjk])/i$2 $1$2/g;
	    $v =~ s/(il)\+([&]{0,1})([a-zA-Z])/$1 $2$3/g;
	    $v =~ s/\s+\*\*i([ltgdrzs\$SDTZnjk])\s+/ i$1 /g;
	    $v =~ s/\s+([^\*\s]+)\*\*/ $1/g;
	    $v =~ s/\s+(il)\+(\*?)/ $1 $2/g;
	} else {
	    # Tokenize and lookup
	    local ($new) = "";
	    foreach $word(split(/\s+/,$v)){
		if ($Translate){
		    if ($Script_LUT{$word} !~ /^$/){
			$word = $Script_LUT{$word};
		    }
		}
		if (defined($DefArt_LUT{$word})){
		    if ($DefArt_LUT{$word} ne /^$/){	
			$new .= " $DefArt_LUT{$word}";
		    } else {
			$new .= " $word";
		    }
		} else {
		    warn "Word $word is not in lexicon\n";
		    $new .= " $word";
		}
	    }
	    ($v = $new) =~ s/^ //;
	}
    }
    $v;
}

sub Load_Lexicon{
    local ($orth_rom, $orth_scr, $pron, $stress, $morph, $tr_wf, $dt_wf, $et_wf);
    local ($det, $base);

    open(LEX,$Lexicon) || die("Error: Unable to open lexicon file $Lexicon");
    while (<LEX>){
	($orth_rom, $orth_scr, $pron, $stress, $morph, $tr_wf, $dt_wf, $et_wf) =
	    split;
	if ($Translate) {
	    $Script_LUT{$orth_rom} = $orth_scr;
	}

	if ($morph =~ /\+article/ && $orth_rom =~ /il\+/){
	    # extract the base form 
	    ($base = $morph) =~ s/:.*$//;
	    if (($orth_rom =~     /^&*il\+[^+]*$/) ||
		($orth_rom =~ /^&*bi\+il\+[^+]*$/) ||
		($orth_rom =~ /^&*fa\+il\+[^+]*$/)) {
		($det = $orth_scr) =~ s/(ה)/$1 /;
	    } elsif (($orth_rom =~     /^li\+il\+[^+]*$/) ||
		     ($orth_rom =~ /^li\+bi\+il\+[^+]*$/)) {
		($det = $orth_scr) =~ s/(הה)/$1 /;
	    } else {
		print "HELP  $base  $_";
	    }
	    $DefArt_LUT{$orth_scr} = $det;
	} else {
	    $DefArt_LUT{$orth_scr} = $orth_scr;
	}
    }
    close(LEX);
}

