#define MAIN
#include "sctk.h"

#define SCSTATS_VERSION "1.3"

int load_pralign_file(FILE *fp, SCORES **scor, int *nscor, int maxn);


void do_exit(char *desc, char *prog, int ret);
void proc_arcs(int argc, char **argv, char *prog, char **pralign, int *npralign, int *pipein, int *feedback, char **out_dir, int *stats, int *stats_verbose, int *stats_unified, int *stats_graphs, char **stats_out_name, char **stats_formula, char **stats_test_name, int *reports, int *linewidth);
void run_multi_sys(SCORES *scor[],int nscor,char *out_dir,int stats,int stats_verbose,int stats_unified, int stats_graphs, char *stats_out_name, char *stats_formula, char *stats_test_name, int reports, int feedback, int linewidth);
 
#define STAT_MCN	0x0001
#define STAT_MAPSSWE	0x0002
#define STAT_SIGN	0x0004
#define STAT_WILC	0x0008
#define STAT_ANOVAR	0x0010
#define STAT_STDOUT	0x0020
#define STAT_MAPSSWE_SEG	0x0040

#define GRAPH_RANGE	0x0001
#define GRAPH_GRANGE	0x0002
#define GRAPH_GRANGE2	0x0008
#define GRAPH_STDOUT	0x0004
#define GRAPH_DET	0x0010

#define REPORT_NONE	0x0000
#define REPORT_SUM	0x0001
#define REPORT_RSUM	0x0002
#define REPORT_LUR	0x0004
#define REPORT_ESUM	0x0010
#define REPORT_RESUM	0x0020
#define REPORT_PRN	0x0040

char *usage = "%s: <OPTIONS>\n" 
"sc_stats Version: " SCSTATS_VERSION ", SCTK Version: " TK_VERSION "\n"
"Input Options:\n"
"    -p                             Read from stdin, the piped output\n"
"                                   alignments from another sctk program.\n"
"Output Options:\n"
"    -O output_dir                  Writes all output files into output_dir.\n"
"                                   Defaults to the hypfile's directory\n"
"    -n name                        Writes all multiple hypothesis file\n"
"                                   reports to files beginning with 'name'.\n"
"                                   Using '-' writes to stdout. Default:\n"
"                                   'Ensemble'\n"
"    -e 'ensemble description'      Description of the ensemble of hyp files.\n"
"    -l width                       Set the report line width to 'width'\n"
"Report Generation Options:\n"
"    -r [ lur | sum | rsum | es | res | prn | npr | none ]\n"
"                                   Defines the output reports.  Default 'none'.\n"
"Statistical Test Options:\n"
"    -t [ mcn | mapsswe | sign | wilc | anovar | std4 ]\n"
"                                   Create comparison matrices for each\n"
"                                   if the specified tests.\n"
"    -v                             For each test performed on a pair of\n"
"                                   hyp files, output a detailed analysis.\n"
"    -u                             Rather than creating a comparison matrix\n"
"                                   for each test, unify statistical test\n"
"                                   results into a single comparision matrix\n"
"    -f [ E | R | W ]               Percentage formula used in the sign,\n"
"                                   wilcoxon and anovar tests. Def, 'E' \n"
"                                   for Word E.\n"
"    -g [ range | grange | grange2 | det ]\n"
"                                   Create range graphs based on the % \n"
"                                   formula defined by '-f'\n";

#define MAX_HYPS 40

int main(int argc, char **argv){
    char *prog = "sc_stats";
    char *pralign[MAX_HYPS];
    int npralign;
    int pipein;
    char *out_dir;
    int stats;
    int stats_verbose;
    int stats_unified;
    int stats_graphs;
    char *stats_formula;
    char *stats_out_name = "STATS";
    char *stats_test_name;
    int feedback;
    int reports;
    SCORES *scor[MAX_HYPS];
    int nscor=0, nh;
    int linewidth;

    db = db_level = 0;
    proc_arcs(argc, argv, prog, pralign, &npralign, &pipein,
	      &feedback,&out_dir,&stats,&stats_verbose,&stats_unified,
	      &stats_graphs,&stats_out_name,&stats_formula, &stats_test_name,
	      &reports, &linewidth);

    if (feedback > 0) printf("sc_stats: " SCSTATS_VERSION "\n");

    if (pipein){
	if (!load_SCORES_sgml(stdin,scor,&nscor,MAX_HYPS)){
	    fprintf(scfp,"Error: load_SCORES_sgml Failed\n");
	    exit(1);
	}
    }

    if (stats > 0 || stats_graphs > 0 || reports > 0)
	run_multi_sys(scor,nscor,out_dir,stats,stats_verbose,stats_unified,
		      stats_graphs,stats_out_name,stats_formula,
		      stats_test_name,reports,feedback, linewidth);

    /* clean up the score structures */
    for (nh=0; nh<nscor; nh++)
	SCORES_free(scor[nh]);

    if (feedback >= 1) printf("\nSuccessful Completion\n");
    return(0);
}

void run_multi_sys(SCORES *scor[],int nscor,char *out_dir,int stats,int stats_verbose,int stats_unified, int stats_graphs, char *stats_out_name, char *stats_formula, char *stats_test_name, int reports, int feedback, int linewidth)
{
    RANK rank;
    int nwin=0, **winner[15];
    double **confidence[15];
    char *win_desc[15], *win_id[15], *stat_form_str=formula_str(stats_formula);
    char outroot[200], *sign_str=(char *)0, *wilc_str=(char *)0;
    char outname[200];

    if (feedback >= 1)
	printf("Beginning Multi-System comparisons and reports\n");

    if (strcmp(stats_out_name,"-") == 0)
	strcpy(outroot,stats_out_name);
    else
	sprintf(outroot,"%s%s%s",
		((out_dir != (char *)0) ? out_dir : ""),
		((out_dir != (char *)0) ? "/" : ""),
		stats_out_name);
    /* printf("Out Root = %s\n",outroot); */
    if (strcmp("-",outroot) == 0)
	strcpy(outname,outroot);
    else
	sprintf(outname,"%s.%s",outroot,"stats");


    if (stats > 0 || stats_graphs > 0)
	init_RANK_struct_from_SCORES(&rank, scor, nscor, stats_formula);

    /*    dump_full_RANK_struct(&rank, "Systems", "Speakers", "", "",stats_formula,
			  "test name", "Speaker ranks for the system",
			  "System ranks for the speaker");  */

    if ((stats & STAT_MAPSSWE) > 0){
	win_desc[nwin] = "Matched Pair Sentence Segment (Word Error)";
	win_id[nwin] = "MP";
	if (feedback >= 1) 
	    printf("    Performing the %s Test\n",win_desc[nwin]);
	do_mtch_pairs(scor, nscor, "2", stats_test_name, 
		      !stats_unified, stats_verbose, &winner[nwin],
		      outname, feedback, &confidence[nwin]);
	nwin++;
    }

    if ((stats & STAT_MAPSSWE_SEG) > 0){
        do_mtch_pairs_seg_analysis(scor, nscor, stats_test_name, 1, 1);
    }
    
    if ((stats & STAT_SIGN) > 0){
	win_desc[nwin] = (char *)TEXT_strdup((TEXT*)rsprintf("%s (%s)",
					     "Signed Paired Comparison",
						      stat_form_str));
	sign_str = win_desc[nwin];
	win_id[nwin] = "SI";
	if (feedback >= 1) 
	    printf("    Performing the %s Test\n",win_desc[nwin]);
	perform_signtest(&rank, stats_verbose, !stats_unified, 
			 stat_form_str, *stats_formula, &winner[nwin],
			 outname, feedback, &confidence[nwin]);
	nwin++;
    }

    if ((stats & STAT_WILC) > 0){
	win_desc[nwin] = (char*)TEXT_strdup((TEXT *)rsprintf("%s (%s)",
					 "Wilcoxon Signed Rank",
					 stat_form_str));
	wilc_str = win_desc[nwin];
	win_id[nwin] = "WI";
	if (feedback >= 1) 
	    printf("    Performing the %s Test\n",win_desc[nwin]);
	perform_wilcoxon(&rank, stats_verbose, !stats_unified,
			 stat_form_str, *stats_formula, &winner[nwin],
			 outname, feedback, &confidence[nwin]);
	nwin++;
    }

    if ((stats & STAT_MCN) > 0){
	win_desc[nwin] = "McNemar (Sentence Error)"; win_id[nwin] = "MN";
	if (feedback >= 1) 
	    printf("    Performing the %s Test\n",win_desc[nwin]);
	McNemar_sent(scor, nscor, &winner[nwin], stats_test_name,
		     !stats_unified, stats_verbose,outname,
		     feedback, &confidence[nwin]);
	nwin++;
    }

    if ((stats & STAT_ANOVAR) > 0){
	win_desc[nwin] = "Analysis of Variance by Ranks";
	win_id[nwin] = "AN";
	if (feedback >= 1) 
	    printf("    Performing the %s Test\n",win_desc[nwin]);
	compute_anovar(&rank, stats_verbose, !stats_unified, 
		       &winner[nwin],outname,feedback,&confidence[nwin]);
	nwin++;
    }

    if (stats_unified){
	if (feedback >= 1) 
	    printf("    Printing Unified Statistical Test Reports\n");
#define new
#ifdef new
	print_composite_significance2(&rank, 80, nwin, winner, confidence,
				      win_desc, win_id,
				      1 /*matrix*/, 1 /*int report*/,
				      stats_test_name,outname, feedback,
				      (out_dir != (char *)0) ? out_dir : "");
#endif
#ifdef old
	print_composite_significance(&rank, 80, nwin, winner,
				      win_desc, win_id,
				      1 /*matrix*/, 1 /*int report*/,
				      stats_test_name,outname,feedback,
				      (out_dir != (char *)0) ? out_dir : "");
#endif
    }

    
    if ((stats_graphs & GRAPH_RANGE) > 0){
	if (strcmp("-",outroot) == 0)
	    strcpy(outname,outroot);
	else
	    sprintf(outname,"%s.%s",outroot,"range");
	if (feedback >= 1) printf("    Printing Range Graphs\n");
	print_rank_ranges(&rank, stat_form_str,stats_test_name, outname, feedback);
    }

    if ((stats_graphs & GRAPH_GRANGE) > 0){
	if (strcmp("-",outroot) == 0)
	    strcpy(outname,outroot);
	else
	    sprintf(outname,"%s.%s",outroot,"grange");
	if (feedback >= 1) printf("    Printing GNUPLOT Range Graphs\n");
	print_gnu_rank_ranges(&rank, stat_form_str,stats_test_name, outname, feedback);
    }
    if ((stats_graphs & GRAPH_GRANGE2) > 0){
	if (strcmp("-",outroot) == 0)
	    strcpy(outname,outroot);
	else
	    sprintf(outname,"%s.%s",outroot,"grange2");
	if (feedback >= 1) printf("    Printing GNUPLOT Range Graphs V2\n");
	print_gnu_rank_ranges2(&rank, stat_form_str,stats_test_name, outname, feedback);
    }

    if ((stats_graphs & GRAPH_DET) > 0){
      if (strcmp("-",outroot) != 0){
	if (make_SCORES_DET_curve(scor, nscor,outroot,feedback,stats_test_name)
	    != 0)
	  exit(1);
      } else {
	fprintf(scfp,"Warning: DET plot can not go to STDOUT\n");
      }
    }

    strcpy(outname,outroot);

    if ((reports & REPORT_SUM) > 0)
	print_N_system_summary(scor, nscor, outname, stats_test_name, 0, feedback);

    if ((reports & REPORT_RSUM) > 0)
	print_N_system_summary(scor, nscor, outname, stats_test_name, 1,
			       feedback);
    if ((reports & REPORT_ESUM) > 0)
	print_N_system_executive_summary(scor, nscor, outname,
					 stats_test_name, 0, feedback);
    if ((reports & REPORT_RESUM) > 0)
	print_N_system_executive_summary(scor, nscor, outname,
					 stats_test_name, 1, feedback);
    if ((reports & REPORT_PRN) > 0)
	print_N_SCORE(scor, nscor, outname, linewidth, feedback, 0);

    if ((reports & REPORT_LUR) > 0)
	print_N_lur(scor, nscor, outname, stats_test_name, feedback);

    if (stats > 0 || stats_graphs > 0){
	int i;
	for (i=0; i<nwin; i++)
	    free_2dimarr(winner[i],rank.n_trt,int);
	free_RANK(&rank);
    }
    if (sign_str != (char *)0) free(sign_str);
    if (wilc_str != (char *)0) free(wilc_str);

}

void proc_arcs(int argc, char **argv, char *prog, char **pralign, int *npralign, int *pipein, int *feedback, char **out_dir, int *stats, int *stats_verbose, int *stats_unified, int *stats_graphs, char **stats_out_name, char **stats_formula, char **stats_test_name, int *reports, int *linewidth){
    int opt, fbset=0, statset=0;

    if (argc <= 1) 
	do_exit("Arguments reguired",prog,1);

    *feedback=1;
    *npralign = 0;
    *pralign =(char *)0;
    *out_dir = (char *)0;
    *stats = *stats_verbose = *stats_unified = *stats_graphs = 0;    
    *stats_formula = "E";
    *stats_test_name = "";
    *stats_out_name = "Ensemble";
    *reports = REPORT_NONE;
    *linewidth = 100;

    for (opt = 1; opt < argc && (*(argv[opt]) == '-'); opt++){
	/* printf("Current OP %s\n",argv[opt]); */
	if (strcmp(argv[opt],"-a") == 0){
	    if (argc < opt + 2) do_exit("Not enough alignment files",prog,1);
	    if (*(argv[opt+1]) == '-') do_exit("Req'd alignment File name",prog,1);
	    while (argc > opt+1 && (*(argv[opt+1]) != '-'))
		pralign[(*npralign)++] = argv[++opt];
	} else if (strcmp(argv[opt],"-p") == 0){
	    *pipein = 1;
	} else if (strcmp(argv[opt],"-f") == 0){
	    if (argc <= opt+1) do_exit("Not enough Feedback arguments",prog,1);
	    *feedback = atoi(argv[++opt]);
	    fbset = 1;
	} else if (strcmp(argv[opt],"-O") == 0){
	    if (argc <= opt+1)do_exit("Output directory not specified",prog,1);
	    *out_dir = argv[++opt];
	} else if (strcmp(argv[opt],"-n") == 0){
	    if (argc <= opt+1)do_exit("Test output name not specified",prog,1);
	    *stats_out_name = argv[++opt];
	} else if (strcmp(argv[opt],"-e") == 0){
	    if (argc <= opt+1)do_exit("Test ensemble desc. not specified",
				      prog,1);
	    *stats_test_name = argv[++opt];
	} else if (strcmp(argv[opt],"-l") == 0){
	    if (argc <= opt+1)
		do_exit("Not enough Line Width arguments",prog,1);
	    *linewidth = atoi(argv[++opt]);
	} else if (strcmp(argv[opt],"-v") == 0){
	    *stats_verbose = 1;
	} else if (strcmp(argv[opt],"-u") == 0){
	    *stats_unified = 1;
	} else if (strcmp(argv[opt],"-t") == 0){
	    opt ++;
	    if (argc < opt + 1) do_exit("Not enough Test arguments",prog,1);
	    while (opt < argc && *(argv[opt]) != '-'){
		if (strcmp(argv[opt],"mcn") == 0) 
		    *stats ^= STAT_MCN;
		else if (strcmp(argv[opt],"mapsswe") == 0) 
		    *stats ^= STAT_MAPSSWE;
		else if (strcmp(argv[opt],"mapsswe_seg") == 0) 
		    *stats ^= STAT_MAPSSWE_SEG;
		else if (strcmp(argv[opt],"sign") == 0) 
		    *stats ^= STAT_SIGN;
		else if (strcmp(argv[opt],"wilc") == 0) 
		    *stats ^= STAT_WILC;
		else if (strcmp(argv[opt],"anovar") == 0) 
		    *stats ^= STAT_ANOVAR;
		else if (strcmp(argv[opt],"std4") == 0) 
		    *stats = STAT_MCN + STAT_MAPSSWE + STAT_SIGN + STAT_WILC;
		else
		    fprintf(stderr,"Unknown test '%s'\n",argv[opt]);
		opt++;
	    }
	    /* backup if we've gone to far */
	    if (opt < argc && *(argv[opt]) == '-') opt--;
	    statset=1;
	} else if (strcmp(argv[opt],"-r") == 0){
	    opt ++;
	    if (argc < opt + 1) do_exit("Not enough Report arguments",prog,1);
	    while (opt < argc && *(argv[opt]) != '-'){
		if (strcmp(argv[opt],"sum") == 0) 
		    *reports ^= REPORT_SUM;
		else if (strcmp(argv[opt],"rsum") == 0) 
		    *reports ^= REPORT_RSUM;
		else if (strcmp(argv[opt],"es") == 0) 
		    *reports ^= REPORT_ESUM;
		else if (strcmp(argv[opt],"res") == 0) 
		    *reports ^= REPORT_RESUM;
		else if (strcmp(argv[opt],"prn") == 0) 
		    *reports ^= REPORT_PRN;
		else if (strcmp(argv[opt],"lur") == 0) 
		    *reports ^= REPORT_LUR;
		else if (strcmp(argv[opt],"none") == 0) 
		    *reports = 0;
		else
		    fprintf(stderr,"Unknown report '%s'\n",argv[opt]);
		opt++;
	    }
	    /* backup if we've gone to far */
	    if (opt < argc && *(argv[opt]) == '-') opt--;
	} else if (strcmp(argv[opt],"-g") == 0){
	    opt ++;
	    if (argc < opt + 1) do_exit("Not enough Graph arguments",prog,1);
	    while (opt < argc && *(argv[opt]) != '-'){
		if (strcmp(argv[opt],"range") == 0) 
		    *stats_graphs ^= GRAPH_RANGE;
		else if (strcmp(argv[opt],"grange") == 0) 
		    *stats_graphs ^= GRAPH_GRANGE;
		else if (strcmp(argv[opt],"grange2") == 0) 
		    *stats_graphs ^= GRAPH_GRANGE2;
		else if (strcmp(argv[opt],"det") == 0) 
		    *stats_graphs ^= GRAPH_DET;
		else
		    fprintf(stderr,"Unknown graph option '%s'\n",argv[opt]);
		opt++;
	    }
	    /* backup if we've gone to far */
	    if (opt < argc && *(argv[opt]) == '-') opt--;
	    statset=1;
	} else if (strcmp(argv[opt],"-f") == 0){
	    char *sf;
	    if (argc <= opt+1) do_exit("Percentage Formula Not defined",
				       prog,1);
	    sf = *stats_formula = argv[++opt];
	    if ((strcmp(sf,"E") != 0) && (strcmp(sf,"W") != 0) &&
		(strcmp(sf,"R") != 0))
		if ((strcmp(sf,"e") == 0) || (strcmp(sf,"w") == 0) ||
		    (strcmp(sf,"r") == 0))
		    *sf = toupper(*sf);
		else
		    do_exit(rsprintf("Unrecognized Statistics percent, '%s'\n",
				     sf),prog,1);
	} else if (strcmp(argv[opt],"-w") == 0){
	    /* Do the works */
	    *stats_graphs = GRAPH_RANGE;
	    *stats = STAT_MCN + STAT_MAPSSWE + STAT_SIGN + STAT_WILC;
	    *stats_unified = 1;
	} else {
            (void) fprintf(stderr,usage,prog);
            printf("Illegal argument: %s\n",argv[opt]);
            exit(1);
        }
    }
}

void do_exit(char *desc, char *prog, int ret){
    fprintf(stderr,usage,prog);
    fprintf(stderr,"%s: %s\n\n",prog,desc);
    exit(ret);
}

int load_pralign_file(FILE *fp, SCORES **scor, int *nscor, int maxn){
    char *proc = "load_pralign_file";
    TEXT *buf, *buf2;
    SCORES *tscor;
    char *msg;
    int buf_len=100, buf2_len=100;

    alloc_singZ(buf,buf_len,TEXT,'\0');
    alloc_singZ(buf2,buf2_len,TEXT,'\0');

    while (!feof(fp) && TEXT_ensure_fgets(&buf,&buf_len,fp) != NULL){
	TEXT_xnewline(buf);

	if (TEXT_strCcmp(buf,(TEXT *)"System name:",12) == 0){
	    printf("PARSE: System hit:  %s",buf);

	    if (maxn <= *nscor){
		msg = rsprintf("SCORE array too small, increase size\n");
		goto FAIL;
	    }
	    if (TEXT_nth_field(&buf2,&buf2_len,buf,3) == 0){
		msg = rsprintf("Unable to extract system name from '%s'\n",buf);
		goto FAIL;
	    }
	    tscor = SCORES_init((char *)buf2,10);
	    scor[(*nscor)++] = tscor;
	} else if (TEXT_strCcmp(buf,(TEXT *)"Speaker sentences",17) == 0){
	    printf("PARSE: Speaker hit: %s",buf);
	} else if (TEXT_strCcmp(buf,(TEXT *)"id: ",4) == 0){
	    printf("PARSE: utterance hit: %s",buf);
	}


    }

    free_singarr(buf,TEXT);
    free_singarr(buf2,TEXT);
    return 1;

  FAIL:
    fprintf(scfp,"Error: %s %s\n",proc,msg);
    return(0);
}
