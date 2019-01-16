#!/usr/bin/perl -w

my $Version="1.1";

#####
#  Version 1.0  Released September 9, 1997
#        - Initial Release
#  Version 1.1 Released October 24, 1997
#        - added the -f flag to interpret fragments


my $Usage="Usage: acompl.pl [ -t -s -f ] [ -i fmt ] -m min -l lex Infile|- OutFile|-\n".
    "Version: $Version\n".
    "Desc: The 'acomp' program expands compound words in German orthography into\n".
    "      the constituent parts of those compound words.  The program uses the\n".
    "      LDC German Lexicon as a basis for the expansions.  Input is read from\n".
    "      InFile or STDIN if infile is '-' and output is written to Outfile or\n".
    "      STDOUT if OutFile is '-'\n".
    "Options:\n".
    "      -m min   Set the minimum length on a compound costituent part to 'min'.\n".
    "               Default is 2.\n".
    "      -l lex   File name of the LDC German Dictionary\n".
    "      -t       Disable the processing of triplet consonants\n".
    "      -s       Disable the processing of -s and -en insertions\n".
    "      -f       Divide word fragments as best as you can\n".
    "      -i fmt   Set the input file formant to 'fmt'.  The possible choices are:\n".
    "                  txt -> plain text, the default\n".
    "                  ctm -> CTM format, ignores all but the 5th column, and if\n".
    "                         a division occurs and a confidence score is present,\n".
    "                         the confidence score is copied to all parts.\n".
    "                  stm -> STM format, change only the text field of the stm record\n".
    "\n";
$Lex = "";
$MaxLen = 0;
%LexHash = ();
%CompHash = ();
$MinCompLen = 2;
$DoTriplet = 1;
$DoInserts = 1;
$InFmt = "txt";
$InFile = "";
$OutFile = "";
$Frag = 0;

$| = 1; # This will cause stdout to be flushed whenever we print to it.

use Getopt::Long;
my $ret = GetOptions ("l=s",
		      "m=s",
		      "t",
		      "s",
		      "i:s",
		      "f");
die "\n$usage\nError: Failed to parse argements" if (! $ret);


if (defined($opt_l)) {  $Lex = $opt_l; } else { die("$Usage\n\nError: Lexicon required via -l.\n"); }
if (defined($opt_m)) {  $MinCompLen = $opt_m; }
if (defined($opt_t)) {  $DoTriplet = 0; $opt_t = 0;}
if (defined($opt_s)) {  $DoInserts = 0; $opt_s = 0;}
if (defined($opt_f)) {  $Frag = 1; $opt_f = 0;}
if (defined($opt_i)) {
    die("$Usage\n\nError: Undefined input format '$opt_i'") 
	if ($opt_i !~ /^(txt|ctm|stm)$/);
    $InFmt = $opt_i;
}
#### The main functions arguements:
if ($#ARGV > 1) { print "\n$Usage\nToo many arguements\n\n"; exit 1; } 
if ($#ARGV == 0) { print "\n$Usage\nOutput Not Specified\n\n"; exit 1; } 
if ($#ARGV == -1) { print "\n$Usage\nInput and Output Not Specified\n\n";
		    exit 1; } 

$InFile=$ARGV[0];
$OutFile=$ARGV[1];
die("$Usage\nError: Input file $InFile does not exist\n")
    if ($InFile ne "-" && ! -r $InFile);

##########################   MAIN   #####################D

&LoadLex();

open(IN,"$InFile") || die("Error: Unable to open input file '$InFile'");
open(OUT,">$OutFile") || die("Error: Unable to open output file '$OutFile'");

while (<IN>){
    chop;
    if ($_ =~ /^;;/){
	print OUT "$_\n";
    } elsif ($InFmt eq "txt"){
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
	    printf OUT ("%s %s %.2f %.2f %s %s\n",$ctm[0],$ctm[1],
			$ctm[2] + ($i * ($ctm[3] / ($#new_words+1))),
			$ctm[3] / ($#new_words + 1), $new_words[$i], $conf);
	}
    } elsif ($InFmt eq "stm"){
	s/^\s+//;
	local(@sl) = split(/\s+/,$_,7);
        local($txt);
	die "Error: Not enough fields in stm line '$_'\n" if ($#sl < 4);

	if ($#sl == 4){
	    print OUT "$sl[0] $sl[1] $sl[2] $sl[3] $sl[4]\n";
	} else {
	    print OUT "$sl[0] $sl[1] $sl[2] $sl[3] $sl[4]";
	    if ($sl[5] =~ /^<.*>$/){
		print OUT " $sl[5]";
		$txt = "";
		if ($#sl == 6){  $txt = $sl[6];  }
	    } else {
		$txt = $sl[5];
		if ($#sl == 6){  $txt .= " ".$sl[6];  }
	    }
	    if ($txt ne ""){
		print OUT " ".&String_LU($txt);
	    }
	    print OUT "\n";
	}
    }
}

close (IN); close(OUT);

#################   END OF MAIN   #########################
###########################################################

sub String_LU{
    local($s) = @_;
    local($new) = "";
    local($ne);
    $s =~ tr/a-z/A-Z/;
    $s =~ tr/\340-\377/\300-\337/; 
    foreach $word(split(/\s+/,$s)){
	if ($word =~ /^\(.*\)$/){
	    $word =~ s/^\((.*)\)$/$1/; 
	    local($lookup) = &Comp_LU($word);
	    if ($word !~ /-$/ || (! $Frag)){
		foreach $ne(split(/\s+/,$lookup)){
		    $new .= "(".$ne.") ";
		}
	    } else {
		$lookup =~ s/(\S+-)$/aq($1)/; 
		$new .= "$lookup ";
	    }
	} else {
	    $new .= &Comp_LU($word)." ";
	}
    }
    $new =~ s/ $//;
    $new;
}

sub Comp_LU{
    local($text) = @_;
    local(@Comps) = ();
    local($comp);
    local($db) = 0;

    if ($CompHash{$text} !~ /^$/){
	print "hashed $text = $CompHash{$text}\n" if ($db);
	$CompHash{$text};
    } else {
	if ($Frag && $text =~ /-$/) {  
	    print "Fragment Lookup '$text'\n" if ($db);
	    
	    local($len) = length($text);
	    local($htext) = "";
	    local($nocomp) = 1;
	    local($i, $j);
	    for ($i=$len-2; $i>=$MinCompLen && $nocomp; $i--){
		$htext = substr($text,0,$i);
		&is_Compound("",$htext,"",*Comps);
		if ($#Comps != -1){
		    print "Fragment match @Comps\n" if ($db);
		    foreach ($j=0; $j <= $#Comps; $j++){
			$Comps[$j] .= " ".substr($text,$i);
		    }
		    $nocomp = 0;
		}
	    }	
	} else {
	    print "compute $text  " if ($db);
	    &is_Compound("",$text,"",*Comps);
	}
	
	if ($#Comps == -1) {
	    $comp = $text;
	} elsif ($#Comps > 0) {
	    $comp = "{";
	    foreach $c (@Comps){
		$comp .= " $c /";
	    }
	    $comp =~ s:/$:}:;
        } else {
	    $comp = $Comps[0];
	}
        $CompHash{$text} = $comp;
        print "compute $text = $comp\n" if ($db);
        $comp;
    }
}

sub is_Compound{
    local($hist,$val,$ind,*comps) = @_;
    local($sval);
    local($resid);
    local($vallen) = length($val);
    local($srchlen) = $MaxLen;
    local($db) = 0;
    local($stop_recur) = 0;
    local($i);
    
    $srchlen = $vallen if ($vallen < $MaxLen);

    printf $ind."is_Compound hist: '$hist' val: '$val' slen: $srchlen\n" if ($db);
    if ($LexHash{$val} eq "1"){
	### The word is already in the hashed list, return it.
	local($co);
	($co = "$hist $val") =~ s/^ //;
	print "$ind*******  SUCCESS Search value pre-defined '$co'\n" if ($db);
	push(@comps,$co);
	1;
    } else {
	### loop through all of the characters in a 
	for ($i=$srchlen; $i>=$MinCompLen && ! $stop_recur; $i--){
	    $sval = substr($val,0,$i);
	    print $ind."   $sval\n" if ($db);
	    if ($LexHash{$sval} eq "1"){
		$resid = substr($val,$i);
		print $ind."      Possible subword '$sval' residual '$resid'\n" if ($db);
		if (length($resid) >= $MinCompLen){
		    if (&is_Compound($hist." $sval",$resid, $ind."   |",*comps) == 1){
			$stop_recur = 1;
			print "$ind**** Curtailing search for normal lookup\n" if ($db);
		    } else {
			print "$ind**** Failed to Curtail search\n" if ($db);
		    }
		} else {
		    print "$ind**** Residual length < MinCompLen $MinCompLen\n" if ($db);
		}
		#### handle the geminate constant conditional
		if (! $stop_recur && $DoTriplet && $sval =~ /([bcdfghjklmnpqrstvwxyz])\1$/){
		    $resid = substr($val,$i-1);
		    print $ind."      Possible geminate consonate subword residual $resid\n" if ($db);
		    if (length($resid) >= $MinCompLen){
			if (&is_Compound($hist." $sval",$resid, $ind."   |",*comps) == 1){
			    $stop_recur = 1;
			    print "$ind**** Curtailing search for triplet consonants\n" if ($db);
			} else {
			    print "$ind**** Failed to Curtail search\n" if ($db);
			}
		    }		    
		}
		#### handle the insertion of syllables
		if (! $stop_recur && $DoInserts && ( $resid =~ /^([sS]|[eE][nN])/ ) ){
		    local($gen);
		    if ($resid =~ /^[sS]/){  $resid = substr($val,$i+1); $gen = "s" } 
		    if ($resid =~ /^[eE][nN]/){  $resid = substr($val,$i+2); $gen = "en" } 
		    print $ind."      Possible additional '$gen' on residual '$gen'$resid\n" if ($db);
		    if (length($resid) >= $MinCompLen){
			if (&is_Compound($hist." $sval",$resid, $ind."   |",*comps) == 1){
			    $stop_recur = 1;
			    print "$ind**** Curtailing search for insertion of syllables\n" if ($db);
			} else {
			    print "$ind**** Failed to Curtail search\n" if ($db);
			}
		    }		    
		}
	    }
	}
	$stop_recur;
    }
}

sub LoadLex{
    local($len);
    local($db) = 0;
    open(LEX,$Lex) || die("Unable to open Lexicon '$Lex'");

    print "Reading Lexicon: '$Lex'\n   Each . is 10000 entries:   " if ($db);
    while (<LEX>){
	chop;
	($word) = split;
	$word =~ tr/a-z/A-Z/;
	$word =~ tr/\340-\377/\300-\337/; 

	if ($word =~ /[-_]/){
	    ($sep = $word) =~ s/[-_]+/ /g;
	    ($cmp = $word) =~ s/_//g;
	    foreach $sw(split(/\s+/,$sep)){
		$LexHash{$sw} = "1";
		$len = length($sw);
		$MaxLen = $len if ($MaxLen < $len);
	    }
	    $CompHash{$cmp} = $sep;
	    if ($cmp =~ /-/){
		$cmp =~ s/-//g;
		$CompHash{$cmp} = $sep;
	    }
	} else {
	    $LexHash{$word} = "1";
	    $CompHash{$word} = $word;
	}
	if (($. + 1) % 10000 == 0){
	    print "." if ($db);
	}
    }
    print "\n   $. entries loaded, Largest word is $MaxLen\n" if ($db);

    close(LEX);
    
#    print "LexHash:\n"; foreach $key(sort(keys %LexHash)) { print "   $key -> $LexHash{$key}\n"; }
#    print "CompHash:\n"; foreach $key(sort(keys %CompHash)) { print "   $key -> $CompHash{$key}\n"; }
}

