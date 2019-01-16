
#define MAIN
#include "sctk.h"

#define SCLITE_VERSION "2.10"

void do_exit(char *desc, char *prog, int ret);
void proc_args(int argc, char **argv, char *prog, char **rname, char **rfmt, char **hname, char **hfmt, int *nhyps,  enum id_types *id_type, int *case_sens, int *outputs, char **title, int *feedback, int *linewidth, int *use_diff, char **out_dir, char **out_name,int *char_align, int *pipeout, int *pipein, int *infered_wordseg, char **lexicon, int *frag_correct, int *opt_del, int *inf_flags, int *stm2ctm_reduce, int *time_align, int *conf_outputs, int *left_to_right, char **wwl_file, char **lm_file);

#define OUT_SUM            0x0001
#define OUT_RSUM           0x0002
#define OUT_PRALIGN        0x0004
#define OUT_LUR            0x0008
#define OUT_SGML           0x0010
#define OUT_STDOUT         0x0020
#define OUT_SENT           0x0040
#define OUT_SPKR           0x0080
#define OUT_DTL            0x0100
#define OUT_PRALIGN_FULL   0x0200
#define OUT_WWS            0x0400
#define OUT_NL_SGML        0x0800

#define CONF_OUT_NONE      0x0000
#define CONF_OUT_DET       0x0001
#define CONF_OUT_BHIST     0x0002
#define CONF_OUT_HIST      0x0004
#define CONF_OUT_SBHIST    0x0008

#define REDUCE_NOTHING        0x0000
#define REDUCE_REF_SEGMENTS   0x0001
#define REDUCE_HYP_WORDS      0x0002

char *usage = "%s: <OPTIONS>\n" 
"sclite Version: " SCLITE_VERSION ", SCTK Version: " TK_VERSION "\n"
"Input Options:\n"
"    -r reffile [ <rfmt> ]\n"
"                Define the reference file, and it's format\n"
"    -h hypfile [ <hfmt> <title> ]\n"
"                Define the hypothesis file, it's format, and a 'title' used\n"
"                for reports.  The default title is 'hypfile'.  This option\n"
"                may be used more than once.\n"
"    -i <ids>    Set the utterance id type.   (for transcript mode only)\n"
"    -P          Accept the piped input from another utility.\n"
"    -e gb|euc|utf-8 [ case-conversion-localization ]\n"
"                Interpret characters as GB, EUC, utf-8, or the default, 8-bit ASCII.\n"
"                Optionally, case conversion localization can be set to either 'generic',\n"
"                'babel_turkish', or 'babel_vietnamese'\n"  
"Alignment Options:\n"
"    -s          Do Case-sensitive alignments.\n"
"    -d          Use GNU diff for alignments.\n"
"    -c [ NOASCII DH ]\n"
"                Do the alignment on characters not on words as usual by split-\n"
"                ting words into chars. The optional argument NOASCII does not\n"
"                split ASCII words and the optional arg. DH deletes hyphens from\n"
"                both the ref and hyp before alingment.   Exclusive with -d.\n"
"    -L LM       CMU-Cambridge SLM Language model file to use in alignment and scoring.\n"
"    -S algo1 lexicon [ ASCIITOO ]\n"
"    -S algo2 lexicon [ ASCIITOO ]\n"
"                Instead of performing word alignments, infer the word segmenta-\n"
"                tion using algo1 or algo2.  See sclite(1) for algorithm details.\n"
"    -F          Score fragments as correct.  Options -F and -d are exclusive.\n"
"    -D          Score words marked optionally deletable as correct if deleted.\n"
"                Options -D and -d are exclusive.\n"
"    -T          Use time information, (if available), to calculated word-to-\n"
"                word distances based on times. Options -F and -d are exlc.\n"
"    -w wwl      Perform Word-Weight Mediated alignments, using the WWL file 'wwl'.\n"
"                IF wwl is 'unity' use weight 1.o for all words.\n"
"    -m [ ref | hyp ]\n"
"                Only used for scoring a hyp/ctm file, against a ref/stm file.\n"
"                When the 'ref' option is used, reduce the reference segments\n"
"                to time range of the hyp file's words.  When the 'hyp' option\n"
"                is used, reduce the hyp words to the time range of the ref\n"
"                segments.  The two may be used together.  The argument -m\n"
"                by itself defaults to '-m ref'.  Exclusive with -d.\n"
"Output Options:\n"
"    -O output_dir\n"
"                Writes all output files into output_dir. Defaults to the\n"
"                hypfile's directory.\n"
"    -f level    Defines feedback mode, default is 1\n"
"    -l width    Defines the line width.\n"
"    -p          Pipe the alignments to another sclite utility.  Sets -f to 0.\n"
"Scoring Report Options:\n"
"    -o [ sum | rsum | pralign | all | sgml | stdout | lur | snt | spk | \n"
"         dtl | prf | wws | nl.sgml | none ]\n"
"                Defines the output reports. Default: 'sum stdout'\n"
"    -C [ det | bhist | sbhist | hist | none ] \n"
"                Defines the output formats for analysis of confidence scores.\n"
"                Default: 'none'\n"
"    -n name     Writes all outputs using 'name' as a root filename instead of\n"
"                'hypfile'.  For multiple hypothesis files, the root filename\n"
"                is 'name'.'hypfile'\n";

#define MAX_HYPS 40

int main(int argc, char **argv){
    char *prog = "sclite";
    char *refname, *hypname[MAX_HYPS], *reffmt, *hypfmt[MAX_HYPS], 
         *title[MAX_HYPS];
    char *out_dir, *out_name;
    char *wwl_file;
    enum id_types id_type;
    int errors = 0;
    int num_hyps;
    int linewidth;
    int outputs;
    int conf_outputs;
    int feedback;
    int use_diff;
    SCORES *scor[MAX_HYPS];
    int case_sense, nsc;
    int char_align;
    int pipeout;
    int pipein;
    int num_piped;
    int nh;
    char *hroot;
    int hdirLen = FILENAME_MAX;
    int outrootLen = FILENAME_MAX;
    TEXT *hdir, *outroot;
    int infered_wordseg;
    char *lexicon;
    int frag_correct;
    int opt_del;
    int inf_flags;
    int stm2ctm_reduce;
    int time_align;
    int left_to_right;
    WWL *wwl = (WWL *)0;
    char *lm_file;

    alloc_singZ(hdir, hdirLen, TEXT, NULL_TEXT);
    alloc_singZ(outroot, outrootLen, TEXT, NULL_TEXT);

#ifdef LM_ALIGN
    /*    ng_t ng; */

#endif
    
    db = db_level = 0;
    proc_args(argc, argv, prog, &refname, &reffmt, hypname, hypfmt,
	      &num_hyps, &id_type, &case_sense,&outputs,title,&feedback,
	      &linewidth,&use_diff,&out_dir,&out_name,&char_align,&pipeout,
	      &pipein, &infered_wordseg, &lexicon, &frag_correct, &opt_del,
	      &inf_flags, &stm2ctm_reduce, &time_align, &conf_outputs,
	      &left_to_right, &wwl_file, &lm_file);

    if (feedback > 0) printf("sclite: " SCLITE_VERSION 
			     " TK Version " TK_VERSION"\n");

    if (pipein){
	if (feedback >= 1) printf("Loading Piped input. . . ");
	num_piped = 0;
	if (!load_SCORES_sgml(stdin,scor,&num_piped,MAX_HYPS)){
	    fprintf(scfp,"load_SCORES_sgml Failed\n");
	    exit(1);
	}
	if (feedback >= 1) printf("%d systems loaded.\n",num_piped);	
    } else 
	num_piped = 0;

    if (wwl_file != (char *)0){
      if (feedback >= 1)
	printf("Loading WWL file '%s'\n",wwl_file);
      if (load_WWL(&wwl,(TEXT *)wwl_file) != 0){
	fprintf(stderr,"Error: Unable to read WWL file '%s'\n",wwl_file);
	exit(1);
      }
      wwl->curw = 0;
    }

    for (nsc=0; nsc<num_hyps + num_piped; nsc++){
	if (nsc >= num_piped){
	    nh = nsc - num_piped;
	    if (feedback >= 1)
		printf("Begin alignment of Ref File: '%s' and %s: '%s'\n",
		       refname, "Hyp File", hypname[nh]);

	    /* Set up the output root name */
	    if ((hroot = strrchr(hypname[nh],'/')) != NULL){
	      if (hroot-hypname[nh] + 1 > hdirLen){
		expand_singarr_to_size(hdir,(hroot-hypname[nh]),hdirLen,hroot-hypname[nh]+1,TEXT);
	      }
	      TEXT_strBcpy(hdir,hypname[nh],hroot-hypname[nh]);
	      hroot++;
	    } else {
		hroot = hypname[nh];
		TEXT_strcpy(hdir,(TEXT *)".");
	    }
	    
	    { char *name;
	      if ((out_dir != (char *)0) ||
		  TEXT_strcmp(((out_dir != (char *)0) ? (TEXT *)out_dir : (TEXT *)hdir),(TEXT *)".") != 0)
                name = rsprintf("%s/%s",((out_dir != (char *)0) ? out_dir : (char *)hdir),
				(out_name != (char *)0) ? out_name : hroot);
	      else
	        name = rsprintf("%s",(out_name != (char *)0) ? out_name : hroot);
	      if (outrootLen < TEXT_strlen((TEXT *)name)+1){
		expand_singarr(outroot,TEXT_strlen((TEXT *)name),outrootLen,1.5,TEXT);
	      }
              TEXT_strcpy(outroot, name);	      
	    }
	    if (strcmp(reffmt,"trn") == 0 && strcmp(hypfmt[nh],"trn") == 0)
		if (!use_diff)
		    scor[nh]=align_trans_mode_dp(refname,hypname[nh],title[nh],
						 1,case_sense,feedback,
						 char_align,id_type,
						 infered_wordseg, lexicon,
						 frag_correct, opt_del, 
						 inf_flags, wwl, lm_file);
		else
		    scor[nh]=align_trans_mode_diff(refname,hypname[nh],
						   title[nh],1,case_sense,
						   feedback,id_type);
	    else if (strcmp(reffmt,"stm")== 0 && strcmp(hypfmt[nh],"ctm") == 0)
		if (!use_diff)
		    scor[nh]=align_ctm_to_stm_dp(refname,hypname[nh],title[nh],
						 1, case_sense,feedback,
						 char_align,id_type,
						 infered_wordseg, lexicon,
						 frag_correct, opt_del,
						 inf_flags, 
				  BF_isSET(stm2ctm_reduce,REDUCE_REF_SEGMENTS),
				  BF_isSET(stm2ctm_reduce,REDUCE_HYP_WORDS),
						 left_to_right, wwl, lm_file);
		else {
		  if (left_to_right)
		    scor[nh]=align_ctm_to_stm_diff(refname, hypname[nh],
						   title[nh],1,case_sense,
						   feedback,id_type);
		  else {
		    fprintf(scfp,"Error: can't perform diff alignments on "
			    "right-to-left data\n");
		    exit(1);
		  }
		}
	    else if (strcmp(reffmt,"ctm")== 0 && strcmp(hypfmt[nh],"ctm") == 0)
	      scor[nh]=align_ctm_to_ctm(hypname[nh], refname, title[nh],
					feedback, frag_correct, opt_del,
					case_sense, time_align, left_to_right, wwl,
					lm_file);
	    else if (strcmp(reffmt,"stm")== 0 && strcmp(hypfmt[nh],"txt") ==0){
#if DIFF_ENABLED
	      if (left_to_right)
		scor[nh] =align_text_to_stm(refname, hypname[nh],title[nh],
					    1,case_sense,feedback,id_type);
	      else {
		fprintf(scfp,"Error: can't perform diff alignments on "
			"right-to-left data\n");
		exit(1);
	      }
#else		
		do_exit("Alignments via diff have been disabled",prog,1);
#endif
	    } else{
		fprintf(stderr,"Error: Unable to score '%s' against '%s'\n",
			hypfmt[nh],reffmt);
		exit(1);
	    }
	    
	    if (scor[nh] == (SCORES *)0){
		fprintf(stderr,"%s: Alignment failed.  Exiting.\n",prog);
		exit(1);
	    }
	} else { /* input came from piped input */
	    /* Set up the output root name */
	    if ((hroot = strrchr(scor[nsc]->title,'/')) != NULL){
		strncpy(hdir,scor[nsc]->title,hroot-scor[nsc]->title);
		hdir[hroot-scor[nsc]->title] = '\0';
		hroot++;
	    } else {
		hroot = scor[nsc]->title;
		strcpy(hdir,".");
	    }
	    if ((out_dir != (char *)0) ||
		strcmp(((out_dir != (char *)0) ? out_dir : (char *)hdir),".") != 0)
	      sprintf(outroot,"%s/%s",((out_dir != (char *)0) ? out_dir : (char *)hdir),
		      (out_name != (char *)0) ? out_name : hroot);
	    else
	      sprintf(outroot,"%s",(out_name != (char *)0) ? out_name : hroot);
	}	    
	
	if (BF_isSET(outputs,OUT_SUM))
	  print_system_summary(scor[nsc],
			       BF_isSET(outputs,OUT_STDOUT) ? "-": (char *)outroot,
			       0, 0, 0, feedback);
	if (BF_isSET(outputs,OUT_WWS)){
	  if (scor[nsc]->weight_ali)
	    print_system_summary(scor[nsc],
				 BF_isSET(outputs,OUT_STDOUT) ? "-": (char *)outroot,
				 0, 0, 1, feedback);
	  else
	    printf("    Skipping WWS Report, no word weights supplied.\n");
	}
	if (BF_isSET(outputs,OUT_RSUM))
	    print_system_summary(scor[nsc], 
				 BF_isSET(outputs,OUT_STDOUT) ? "-":(char *)outroot,
				 0, 1, 0, feedback);
	if (BF_isSET(outputs,OUT_SENT))
	    score_dtl_sent(scor[nsc], BF_isSET(outputs,OUT_STDOUT)?"-":(char *)outroot,
			   feedback);
	if (BF_isSET(outputs,OUT_SPKR))
	    score_dtl_spkr(scor[nsc], BF_isSET(outputs,OUT_STDOUT)?"-":(char *)outroot,
			   feedback);
	if (BF_isSET(outputs,OUT_DTL))
	    score_dtl_overall(scor[nsc],
			      BF_isSET(outputs,OUT_STDOUT) ? "-":(char *)outroot,
			      feedback);
	if (BF_isSET(outputs,OUT_LUR))
	    print_lur(scor[nsc],
		      BF_isSET(outputs,OUT_STDOUT) ? "-":(char *)outroot, feedback);
	if (BF_isSET(outputs,OUT_PRALIGN)){
	    FILE *fp = stdout;
	    if (BF_notSET(outputs,OUT_STDOUT))
		fp = fopen(rsprintf("%s.pra",outroot),"w");
	    if (fp == (FILE *)0) fp = stdout;
	    if (feedback >= 1)
		printf("    Writing string alignments to '%s'\n",
		       (fp == stdout) ? "stdout" :
		       rsprintf("%s.pra",outroot));
	    dump_SCORES_alignments(scor[nsc],fp,linewidth,0);
	    if (fp != stdout) fclose(fp);
	}
	if (BF_isSET(outputs,OUT_PRALIGN_FULL)){
	    FILE *fp = stdout;
	    if (BF_notSET(outputs,OUT_STDOUT))
		fp = fopen(rsprintf("%s.prf",outroot),"w");
	    if (fp == (FILE *)0) fp = stdout;
	    if (feedback >= 1)
		printf("    Writing string alignments to '%s'\n",
		       (fp == stdout) ? "stdout" :
		       rsprintf("%s.prf",outroot));
	    dump_SCORES_alignments(scor[nsc],fp,linewidth,1);
	    if (fp != stdout) fclose(fp);
	}
	if (BF_isSET(outputs,OUT_SGML)){
	    FILE *fp = stdout;
	    if (BF_notSET(outputs,OUT_STDOUT))
		fp = fopen(rsprintf("%s.sgml",outroot),"w");
	    if (fp == (FILE *)0) fp = stdout;
	    if (fp != stdout && feedback >= 1)
		printf("    Writing SGML string alignments to '%s'\n",
		       rsprintf("%s.sgml",outroot));
	    dump_SCORES_sgml(scor[nsc],fp,(TEXT *)":",(TEXT *)",");
	    if (fp != stdout) fclose(fp);
	}
	if (BF_isSET(outputs,OUT_NL_SGML)){
	    FILE *fp = stdout;
	    if (BF_notSET(outputs,OUT_STDOUT))
		fp = fopen(rsprintf("%s.nl.sgml",outroot),"w");
	    if (fp == (FILE *)0) fp = stdout;
	    if (fp != stdout && feedback >= 1)
		printf("    Writing Newline Separated SGML string alignments to '%s'\n",
		       rsprintf("%s.nl.sgml",outroot));
	    dump_SCORES_sgml(scor[nsc],fp,(TEXT *)"\n",(TEXT *)"\n");
	    if (fp != stdout) fclose(fp);
	}
	if (BF_isSET(conf_outputs,CONF_OUT_DET))
	  if (make_SCORES_DET_curve(&(scor[nsc]),1,outroot,feedback,"") != 0)
	    errors++;
	if (BF_isSET(conf_outputs,CONF_OUT_BHIST))
	    if (make_binned_confidence(scor[nsc],outroot,feedback) != 0)
	        errors++;
	if (BF_isSET(conf_outputs,CONF_OUT_HIST))
  	    if (make_confidence_histogram(scor[nsc],outroot,feedback) != 0)
	        errors++;
	if (BF_isSET(conf_outputs,CONF_OUT_SBHIST))
	    if (make_scaled_binned_confidence(scor[nsc],outroot,20,feedback) != 0)
	        errors++;
	
    }

    if (pipeout)
	for (nsc=0; nsc < num_hyps + num_piped; nsc++)
	    dump_SCORES_sgml(scor[nsc],stdout,(TEXT *)"\n",(TEXT *)"\n");

    /* clean up the score structures */
    for (nsc=0; nsc < num_hyps + num_piped; nsc++)
        SCORES_free(scor[nsc]);
    if (wwl != (WWL *)0) free_WWL(&wwl);

    if (feedback >= 1) 
        if (errors == 0)
	    printf("\nSuccessful Completion\n");
	else
  	    printf("\nUnsuccessful Completion\n");
    return(0);
}

void proc_args(int argc, char **argv, char *prog, char **rname, char **rfmt, char **hname, char **hfmt, int *nhyps, enum id_types *id_type, int *case_sense, int *outputs, char **title, int *feedback, int *linewidth, int *use_diff, char **out_dir, char **out_name, int *char_align, int *pipeout, int *pipein, int *infered_wordseg, char **lexicon, int *frag_correct, int *opt_del, int *inf_flags, int *stm2ctm_reduce, int *time_align, int *conf_outputs, int *left_to_right, char **wwl_file, char **lm_file){
    int opt, fbset=0, outset=0, i, nh;
    char *id;
    struct stat fileinfo;

    if (argc <= 1) 
	do_exit("Arguments reguired",prog,1);

    *linewidth=1000;
    *feedback=1;
    *nhyps = 0;
    *rname=(char *)0;  *hname=(char *)0;
    *rfmt=(char *)0;   *hfmt=(char *)0;
    id=(char *)0;
    *outputs=0;
    *case_sense=0;
    *title=(char *)0;
    *out_dir = (char *)0;
    *out_name = (char *)0;
    *use_diff = 0;
    *char_align = 0;
    *pipeout = 0;
    *pipein = 0;
    *infered_wordseg = 0;
    *lexicon = (char *)0;
    *frag_correct = 0;
    *opt_del = 0;
    *inf_flags = 0;
    *stm2ctm_reduce = 0;
    *time_align = 0;
    *conf_outputs = CONF_OUT_NONE;
    *left_to_right = 1;
    *wwl_file = (char *)0;
    *lm_file = (char *)0;

    for (opt = 1; opt < argc && (*(argv[opt]) == '-'); opt++){
	/* printf("Current OP %s\n",argv[opt]); */
	if (strcmp(argv[opt],"-r") == 0){
	    if (argc < opt + 2) do_exit("Not enough Ref arguments",prog,1);
	    if (*(argv[opt+1]) == '-') do_exit("Req'd Ref File name",prog,1);
	    *rname = argv[++opt];
	    if (argc >= opt + 2 && *(argv[opt+1]) != '-') *rfmt  = argv[++opt];
	} else if (strcmp(argv[opt],"-h") == 0){
	    if (argc < opt + 2) do_exit("Not enough Hyp arguments",prog,1);
	    if (*(argv[opt+1]) == '-') do_exit("Req'd Hyp File name",prog,1);
	    hname[*nhyps] = argv[++opt];
	    if (argc >= opt + 2 && *(argv[opt+1]) != '-') 
		hfmt[*nhyps]  = argv[++opt];
	    else
		hfmt[*nhyps] = "trn";
	    if (argc >= opt + 2 && *(argv[opt+1]) != '-') 
		title[*nhyps] = argv[++opt];
	    else
		title[*nhyps] = hname[*nhyps];
	    (*nhyps)++;
	} else if (strcmp(argv[opt],"-i") == 0){
	    if (argc <= opt+1) do_exit("Not enough ID arguments",prog,1);
	    id = argv[++opt];
	} else if (strcmp(argv[opt],"-l") == 0){
	    if (argc <= opt+1)
		do_exit("Not enough Line Width arguments",prog,1);
	    *linewidth = atoi(argv[++opt]);
	} else if (strcmp(argv[opt],"-L") == 0){
#ifdef WITH_SLM
	    if (argc <= opt+1)
		do_exit("Not enough Language Model arguments",prog,1);
	    *lm_file = argv[++opt];
#else
	    do_exit("-L option used, but SLM toolkit not compiled into sctk\n",prog,1);
#endif
	} else if (strcmp(argv[opt],"-f") == 0){
	    if (argc <= opt+1) do_exit("Not enough Feedback arguments",prog,1);
	    *feedback = atoi(argv[++opt]);
	    fbset = 1;
	} else if (strcmp(argv[opt],"-o") == 0){
	    opt ++;
	    if (argc < opt + 1) do_exit("Not enough Report arguments",prog,1);
	    while (opt < argc && *(argv[opt]) != '-'){
		if (strcmp(argv[opt],"sum") == 0) 
		    BF_FLIP(*outputs,OUT_SUM);
		else if (strcmp(argv[opt],"wws") == 0) 
		    BF_FLIP(*outputs,OUT_WWS);
		else if (strcmp(argv[opt],"rsum") == 0) 
		    BF_FLIP(*outputs,OUT_RSUM);
		else if ((strcmp(argv[opt],"pralign") == 0) ||
			 (strcmp(argv[opt],"pra") == 0)) 
		    BF_FLIP(*outputs,OUT_PRALIGN);
		else if (strcmp(argv[opt],"prf") == 0) 
		    BF_FLIP(*outputs,OUT_PRALIGN_FULL);
		else if (strcmp(argv[opt],"lur") == 0) 
		    BF_FLIP(*outputs,OUT_LUR);
		else if (strcmp(argv[opt],"stdout") == 0) 
		    BF_FLIP(*outputs,OUT_STDOUT);
		else if (strcmp(argv[opt],"sgml") == 0) 
		    BF_FLIP(*outputs,OUT_SGML);
		else if (strcmp(argv[opt],"nl.sgml") == 0) 
		    BF_FLIP(*outputs,OUT_NL_SGML);
		else if (strcmp(argv[opt],"snt") == 0) 
		    BF_FLIP(*outputs,OUT_SENT);
		else if (strcmp(argv[opt],"spk") == 0) 
		    BF_FLIP(*outputs,OUT_SPKR);
		else if (strcmp(argv[opt],"dtl") == 0) 
		    BF_FLIP(*outputs,OUT_DTL);
		else if (strcmp(argv[opt],"all") == 0) 
		    BF_FLIP(*outputs,OUT_PRALIGN + OUT_SUM + OUT_RSUM);
		else if (strcmp(argv[opt],"none") == 0) 
		    *outputs = 0;
		else
		    fprintf(stderr,"Unknown report '%s'\n",argv[opt]);
		opt++;
	    }
	    /* backup if we've gone to far */
	    if (opt < argc && *(argv[opt]) == '-') opt--;
	    outset=1;
	} else if (strcmp(argv[opt],"-C") == 0){
	    opt ++;
	    if (argc < opt + 1)
	      do_exit("Not enough Confidence Report arguments",prog,1);
	    while (opt < argc && *(argv[opt]) != '-'){
		if (strcmp(argv[opt],"det") == 0) 
		    BF_FLIP(*conf_outputs,CONF_OUT_DET);
		else if (strcmp(argv[opt],"bhist") == 0) 
		    BF_FLIP(*conf_outputs,CONF_OUT_BHIST);
		else if (strcmp(argv[opt],"sbhist") == 0) 
		    BF_FLIP(*conf_outputs,CONF_OUT_SBHIST);
		else if (strcmp(argv[opt],"hist") == 0) 
		    BF_FLIP(*conf_outputs,CONF_OUT_HIST);
		else if (strcmp(argv[opt],"none") == 0) 
		    *conf_outputs = CONF_OUT_NONE;
		else
		    fprintf(stderr,"Unknown confidence report '%s'\n",
			    argv[opt]);
		opt++;
	    }
	    /* backup if we've gone to far */
	    if (opt < argc && *(argv[opt]) == '-') opt--;
	    outset=1;
	} else if (strcmp(argv[opt],"-O") == 0){
	    if (argc <= opt+1)do_exit("Output directory not specified",prog,1);
	    *out_dir = argv[++opt];
	} else if (strcmp(argv[opt],"-n") == 0){
	    if (argc <= opt+1)do_exit("Output name not specified",prog,1);
	    *out_name = argv[++opt];
	} else if (strcmp(argv[opt],"-d") == 0){
#if DIFF_ENABLED
	    *use_diff = 1;
#else
	    do_exit("Alignments via diff have been disabled",prog,1);
#endif
	} else if (strcmp(argv[opt],"-s") == 0){
	    *case_sense = 1;
	} else if (strcmp(argv[opt],"-m") == 0){
	    *stm2ctm_reduce = REDUCE_NOTHING;
	    while (opt+1 < argc && *(argv[opt+1]) != '-') {
		opt++;
		if (strcmp(argv[opt],"ref") == 0)
		    BF_SET(*stm2ctm_reduce,REDUCE_REF_SEGMENTS);
		else if (strcmp(argv[opt],"hyp") == 0)
		    BF_SET(*stm2ctm_reduce,REDUCE_HYP_WORDS);
		else
		    do_exit(rsprintf("Unrecognized -m option '%s'",
				     argv[opt]),prog,1);
	    }
	    if (*stm2ctm_reduce == REDUCE_NOTHING)
		*stm2ctm_reduce = REDUCE_REF_SEGMENTS;
	} else if (strcmp(argv[opt],"-F") == 0){
	    *frag_correct = 1;
	} else if (strcmp(argv[opt],"-D") == 0){
	    *opt_del = 1;
	} else if (strcmp(argv[opt],"-c") == 0){
            *char_align = CALI_ON;
	    while (opt+1 < argc && *(argv[opt+1]) != '-') {
		opt++;
		if (strcmp(argv[opt],"NOASCII") == 0)
		    BF_SET(*char_align,CALI_NOASCII);
		else if (strcmp(argv[opt],"DH") == 0)
		    BF_SET(*char_align,CALI_DELHYPHEN);
		else
		    do_exit(rsprintf("Unrecognized character alignment "
				     "option '%s'",argv[opt]),prog,1);
	    }
	} else if (strcmp(argv[opt],"-e") == 0){
	    if (opt+1 >= argc || (opt+1 < argc && *(argv[opt+1]) == '-'))
		do_exit("Argument required for character encoding\n",prog,1);
	    opt++;
	    if (!TEXT_set_encoding(argv[opt]))
		do_exit(rsprintf("Unrecognized character encoding "
				 "option '%s'",argv[opt]),prog,1);
            // Parse the optional localization
	    if (opt+1 < argc && *(argv[opt+1]) != '-'){
	        if (!TEXT_set_lang_prof(argv[opt+1]))
		    do_exit(rsprintf("Optional case conversion localization failed /%s/\n",
		            argv[opt+1]),prog,1);
  	        opt++;			 
  	     }
	} else if (strcmp(argv[opt],"-p") == 0){
	    *pipeout = 1;
	    outset = 1;
	} else if (strcmp(argv[opt],"-P") == 0){
	    *pipein = 1;
	} else if (strcmp(argv[opt],"-T") == 0){
	    *time_align = 1;
	} else if (strcmp(argv[opt],"-S") == 0){
	    opt ++;
	    if (argc < opt + 2) do_exit("Not enough Inferred Segmentation "
					"arguments",prog,1);
	    if (opt < argc && *(argv[opt]) != '-')
		if (strcmp(argv[opt],"algo1") == 0) 
		    *infered_wordseg ^= INF_SEG_ALGO1;
	        else if (strcmp(argv[opt],"algo2") == 0) 
		    *infered_wordseg ^= INF_SEG_ALGO2;
		else 
		    do_exit(rsprintf("Unrecognize inferred segmentation "
				     "algorithm '%s'",argv[opt]),prog,1);
	    else
		do_exit("No Inferred Segmentation algorithm",prog,1);
	    opt++;
	    if (opt < argc && *(argv[opt]) != '-'){
		if (stat(argv[opt],&fileinfo) != 0) 
		    do_exit(rsprintf("Inferred Segmentation lexicon '%s'"
				     " does not exist",argv[opt]),prog,1);
		*lexicon = argv[opt];
	    } else
		do_exit("No Inferred Segmentation lexicon",prog,1);
	    if (opt+1 < argc && *(argv[opt+1]) != '-'){
		opt++;
		if (strcmp(argv[opt],"ASCIITOO") == 0)
		    BF_SET(*inf_flags,INF_ASCII_TOO);
		else
		    do_exit(rsprintf("Unrecognized Inferred Segmentation"
				     " flag '%s'",argv[opt]),prog,1);
	    }
	} else if (strcmp(argv[opt],"-w") == 0){
	    if (argc <= opt+1)do_exit("Word Weight list file not specified",prog,1);
	    *wwl_file = argv[++opt];
	} else {
            (void) fprintf(stderr,usage,prog);
            printf("Illegal argument: %s\n",argv[opt]);
            exit(1);
        }
    }

    if (*pipeout){
	*feedback = 0;
	if (BF_isSET(*outputs,OUT_STDOUT))
	    do_exit("Error: output can not be both to stdout and a pipe",
		    prog,1);
    }

    /* set the default outputs */
    if (*outputs == 0 && !outset)
	BF_SET(*outputs,OUT_SUM + OUT_STDOUT);

    /* reset feedback to 0 if STDOUT is used */
    if ((BF_isSET(*outputs,OUT_STDOUT) && outset) &&
	(*feedback > 0 && !fbset)){
	/* fprintf(stderr,"Feedback level changed to 0\n");*/
	*feedback = 0;
    }

    /* Check the input arguments */
    if (! *pipein && ! (*rname != (char *)0 && *nhyps > 0))
	do_exit("Input not specified, use transcription input or "
		"piped input",prog,1);

    if (*rname == (char *)0 && *nhyps > 0)
	do_exit("Hypothesis input(s) specified, but no reference file",
		prog,1);
    if (*rname != (char *)0 && *nhyps == 0)
	do_exit("Reference input specified, but no Hypothesis input(s).",
		prog,1);

    if (*rname != (char *)0 && *nhyps > 0){
	if (stat(*rname,&fileinfo) != 0) 
	    do_exit(rsprintf("Reference file '%s' does not exist",*rname),
		    prog,1);

	for (nh=0; nh < *nhyps; nh++)
	    if (stat(hname[nh],&fileinfo) != 0) 
		do_exit(rsprintf("Hypothesis file '%s' does not exist",
				 hname[nh]),prog,1);

	/* check the ref and hyp format arguments */
	if (*rfmt == (char *)0) *rfmt = "trn";
	if (! (strcmp(*rfmt,"trn") == 0 || strcmp(*rfmt,"ctm") == 0 || 
	       strcmp(*rfmt,"tmk") == 0 || strcmp(*rfmt,"stm") == 0))
	    do_exit(rsprintf("Reference file format '%s' not acceptable",*rfmt),
		    prog,1);
	
	if (*hfmt == (char *)0) *hfmt = "trn";
	for (i=0; i<*nhyps; i++)
	    if (! (strcmp(hfmt[i],"trn") == 0 || strcmp(hfmt[i],"ctm") == 0 || 
		   strcmp(hfmt[i],"tmk") == 0  || strcmp(hfmt[i],"txt") == 0)) 
		do_exit(rsprintf("Hypothesis file format '%s' not acceptable",
				 hfmt[i]),prog,1);
	
	/* check the id arguements */
	if (id == (char *)0){
	    if (strcmp(*rfmt,"trn") == 0)
		do_exit("Required utterance id (option -i) for"
			" transcript mode", prog,1);
	    for (i=0; i<*nhyps; i++){
		if (strcmp(hfmt[i],"trn") == 0)
		    do_exit("Required utterance id (option -i) for"
			    " transcript mode",prog,1);
	    }
	}

	/* only do time alignment on ctm to ctm */
	if (*time_align){
	    if (strcmp(*rfmt,"ctm") != 0)
		do_exit("Time-mediated alignments require the reference"
			" file to be in CTM format",prog,1);
	    for (i=0; i<*nhyps; i++)
		if (strcmp(hfmt[i],"ctm") != 0)
		    do_exit("Time-mediated alignments require the hypothesis"
			    " file to be in CTM format",prog,1);
	}
    }
    
    if (id != (char *)0){
	if (strcmp(id,"swb") == 0)         *id_type=SWB;
	else if (strcmp(id,"rm") == 0)     *id_type=RM;
	else if (strcmp(id,"wsj") == 0)    *id_type=WSJ;
	else if (strcmp(id,"spu_id") == 0) *id_type=SPUID;
        else if (strcmp(id,"sp") == 0)     *id_type=SP;
	else
	    do_exit(rsprintf("Utterance id type '%s' not acceptable",id),
		    prog,1);
    }

    if (*stm2ctm_reduce){
	if (strcmp(*rfmt,"stm") != 0)
	    do_exit("-m flag requires stm format reference file",prog,1);
	for (i=0; i<*nhyps; i++)
	    if (strcmp(hfmt[i],"ctm") != 0)
		do_exit("-m flag requires ctm format hypothesis file",prog,1);
	if (*use_diff)
	    do_exit("-m flag can not be used with -d flag",prog,1);
    }

    if (*char_align && (*use_diff ||
		       (strcmp(*rfmt,"stm") == 0 && strcmp(*hfmt,"txt")==0)))
	do_exit("Unable to do character alignments using diff\n",prog,1);
    
    if (*wwl_file != (char *)0 && *lm_file != (char *)0)
      do_exit("Flags '-L' and '-w' are mutually exclusive\n",prog,1);

    if (*wwl_file != (char *)0){
      if (*time_align) do_exit("Flags '-T' and '-w' are mutually exclusive\n",prog,1);
      if (*char_align) do_exit("Flags '-c' and '-w' are mutually exclusive\n",prog,1);
      if (*infered_wordseg) do_exit("Flags '-S' and '-w' are mutually exclusive\n",prog,1);
      if (*use_diff) do_exit("Flags '-d' and '-w' are mutually exclusive\n",prog,1);
    }
    if (*lm_file != (char *)0){
      if (*time_align) do_exit("Flags '-T' and '-L' are mutually exclusive\n",prog,1);
      if (*char_align) do_exit("Flags '-c' and '-L' are mutually exclusive\n",prog,1);
      if (*infered_wordseg) do_exit("Flags '-S' and '-L' are mutually exclusive\n",prog,1);
      if (*use_diff) do_exit("Flags '-d' and '-L' are mutually exclusive\n",prog,1);
    }

    if (BF_isSET(*outputs,OUT_WWS))
      if ((*wwl_file == (char *)0) && (*lm_file == (char *)0))
	do_exit("-o wws option specified without a word weight file (via -w) or "
		"LM file (via -L) specified\n",
		prog,1)	;
}

void do_exit(char *desc, char *prog, int ret){
    fprintf(stderr,usage,prog);
    fprintf(stderr,"\n%s: Error, %s\n\n",prog,desc);
    exit(ret);
}

