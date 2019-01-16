#define MAIN
#include "sctk.h"

#define ROVER_VERSION "0.1"

char *usage = "%s: <OPTIONS>\n" 
"rover Version: " ROVER_VERSION ", SCTK Version: " TK_VERSION "\n"
"Description: rover takes N input files and does an N-way DP alignment\n"
"             on those files.  The output is either a set of minimal cost\n"
"             alignments, or a Voted output depending the -m option.\n"
"Input Options:\n"
"    -h hypfile ctm\n"
"                Define the hypothesis file and it's format. This option\n"
"                must be used more than once.\n"
"Output Options:\n"
"    -o outfile  Define the output file.  (Will be same format as hyps)\n"
"    -f level    Defines feedback mode, default is 1\n"
"    -l width    Defines the line width.\n"
"Alignment Options:\n"
"    -m meth     Execute method:\n"
"                oracle   -> output the fully alternated transcript\n"
"                meth1    -> alpha = -a , conf = -c, choose highest avg\n"
"                maxconf  -> Same as meth1, but use the maximum conf\n"
"                            score for a CS set as the metric\n"
"                avgconf  -> alpha*Sum(WO) + (1-alpha)Sum(Conf(W))\n"
"                maxconfa -> Same as maxconf, but conf=N(@)/S\n"
"                putat    -> Output the putative hit format\n"
"    -s          Do Case-sensitive alignments.\n"
"    -T          Use time information, (if available), to calculated word-to-\n"
"                word distances based on times.\n"
"Processing Options:\n"
"    -a alpha    Default: 1.0\n"
"    -c Null_conf\n"
"                Default: 0.0\n"
;             

#define METH_NULL   0
#define METH_METH1  1
#define METH_ORACLE 2
#define METH_MAXCONF  3
#define METH_METH1A 4
#define METH_MAXCONFA 5
#define METH_PUTAT  6
#define METH_AVGCONF  7

void do_exit(char *desc, char *prog, int ret);
void proc_args(int argc, char **argv, char *prog, char **hname, char **hfmt, int *nhyps, int *feedback, int *linewidth, int *time_align, int *case_sense, char **out_name, double *alpha, double *null_conf, int *method);
void mfalign_ctm_files_v1(char **hypname, int nhyps, int time_align, int case_sense, char *outfile, int feedback, char *out_name, double alpha, double null_conf, int method);
void print_linear(NODE *node, void *p);
NETWORK *perform_mfalign_v1(WTOKE_STR1 **ctms, int nctm, int *sil_end, int time_align, double null_conf);
void set_tag1(ARC *arc, void *p);

#define MAX_HYPS 50
extern int mfdb;
extern int glob_ties;

struct mfalign_best_seq_struct {
    WORD *null_alt;    /* a NULL word for comparisons in the function */
    double alpha;      /* the trade off between selection by word count or
			  confidence scores */
    double null_conf;  /* Confidence associated with NULL words */
    FILE *fpout;       /* output file to write the resulting selection */
    int write_to_fp;   /* boolean to write the selection to fpout */
    int ctm_format;    /* boolean to write the records in ctm format */
    int putat_format;  /* boolean to the putative hit format */
    char *file;        /* File name for ctm format, utt id for trn format */
    char *chan;        /* Channed for the ctm format */
    int method;        /* id of the selection method performed */
};

int main(int argc, char **argv){
    char *prog = "rover";
    char *hypname[MAX_HYPS], *hypfmt[MAX_HYPS];
    int case_sense;
    int feedback;
    int linewidth;
    int time_align;
    int nhyps;
    int method;
    char *out_name;
    double alpha, null_conf;

    proc_args(argc, argv, prog, hypname, hypfmt, &nhyps, &feedback,
	      &linewidth, &time_align, &case_sense, &out_name, &alpha,
	      &null_conf,&method);

    if (strcmp(hypfmt[0],"ctm") == 0){
        mfalign_ctm_files_v1(hypname,nhyps,time_align,case_sense,"-",feedback,
			  out_name, alpha, null_conf, method);
    }
    return(0);
}

void mfalign_ctm_files_v1(char **hypname, int nhyps, int time_align, int case_sense, char *outfile, int feedback, char *out_name, double alpha, double null_conf, int method){
    char *proc = "mfalign_ctm_files";
    WTOKE_STR1 **ctms;
    NETWORK *mfnet;
    FILE **files;
    int *eofs, *conv_end, *sil_end;
    int in;
    int done;
    struct mfalign_best_seq_struct mf_sel_str;

    {if (getenv("MFALIGN_DBG") != NULL) {
	mfdb=(int)atof(getenv("MFALIGN_DBG")); }}

    if (feedback > 1) printf("Beginning: %s\n",proc);
    /* ALLOCATE MEMORY */
    alloc_singZ(ctms,nhyps,WTOKE_STR1 *,(WTOKE_STR1 *)0);
    alloc_singZ(files,nhyps,FILE *,(FILE *)0);
    alloc_singZ(eofs,nhyps,int,0);
    alloc_singZ(conv_end,nhyps,int,0);
    alloc_singZ(sil_end,nhyps,int,0);
    /* initial the best_seq_struct */
    mf_sel_str.method = method;
    mf_sel_str.alpha = alpha;
    mf_sel_str.null_conf = null_conf;
    mf_sel_str.write_to_fp = 1;
    if (out_name != (char *)0)
	if ((mf_sel_str.fpout = fopen(out_name,"w")) == (FILE *)0){
	    fprintf(scfp,"Error: unable to open output file '%s'\n",out_name);
	    exit (1);
	}
    mf_sel_str.ctm_format = 1;
    mf_sel_str.putat_format = 0;
    if (mf_sel_str.method == METH_PUTAT) {
	mf_sel_str.ctm_format = 0;
	mf_sel_str.putat_format = 1;
	/* right the filenames into the putative file :} */
	fprintf(mf_sel_str.fpout,"<input names=\"");
	
	for (in=0; in < nhyps; in++){
	    if (in != 0) fprintf(mf_sel_str.fpout," ");
	    fprintf(mf_sel_str.fpout,"%s",hypname[in]);
	}
	fprintf(mf_sel_str.fpout,"\">\n");
    }
    mf_sel_str.null_alt = new_WORD((TEXT *)"@",-1,0.0,0.0,null_conf,(TEXT *)0,(TEXT *)0,0,0,0.0);
  
    /* OPEN FILES, INIT WTOKE'S, FILE WTOKE'S */
    for (in=0; in < nhyps; in++){
	if ((files[in] = fopen(hypname[in],"r")) == NULL){
	    fprintf(stderr,"Can't open input hypothesis file %s\n",hypname[in]);
	    exit(1);
	}
	ctms[in] = WTOKE_STR1_init(hypname[in]);
    }
    done = 0;
    while (!done){
        locate_next_file_channel(ctms, nhyps, files, hypname, eofs, conv_end,
				 case_sense, feedback);

	mf_sel_str.file = ctms[0]->word[ctms[0]->s].conv;	
	mf_sel_str.chan = ctms[0]->word[ctms[0]->s].turn;	/* channel */
	
	/* loop though the file/channel section, looking for time breaks */
	while(find_common_silence(ctms, nhyps, conv_end, sil_end, 1.0) == 1){
	    if (mfdb >= 5){
		printf("--------- Aligning this chunk -----------\n");

		for (in=0; in<nhyps; in++){
		    double beg, end;
		    beg = (ctms[in]->s > conv_end[in]) ? -1.0 : 
			ctms[in]->word[sil_end[in]].t1 +
			ctms[in]->word[sil_end[in]].dur;
		    end = (sil_end[in] < conv_end[in]) ? 
		    ctms[in]->word[sil_end[in]+1].t1 : 9999999.9;
		    printf("   CTM %d: start: %d end_word: %d start: %.2f"
			   " end: %.2f\n",in,ctms[in]->s,sil_end[in],beg,end);
		}
	    }

	    mfnet = perform_mfalign_v1(ctms,nhyps,sil_end,time_align, null_conf);

	    /* 	    Network_sgml_dump(mfnet, stdout); */

	    Network_traverse(mfnet,print_linear,(void *)&mf_sel_str,0,0,0);

	    Network_destroy(mfnet);

	    /* skip over this chunk because we're done with it. */
	    done = 1;
	    for (in=0; in < nhyps; in++){
		ctms[in]->s = sil_end[in] + 
		    ((sil_end[in] <= conv_end[in]) ? 1 : 0);
		if (!eofs[in] || (ctms[in]->s < ctms[in]->n)) done = 0;
	    }
	}
    }
#ifdef mmm
    if (glob_ties > 0)
	fprintf(stderr,"Warning: %d ties were arbitrarily broken\n",glob_ties);
#endif

    /* FREE THE MEMORY */
    free_singarr(eofs, int);
    free_singarr(conv_end, int);
    free_singarr(sil_end, int);
    for (in=0; in<nhyps; in++) {
	free_mark_file(ctms[in]);
	fclose(files[in]);
    }
    release_WORD(mf_sel_str.null_alt);
    free_singarr(files, FILE *);
    free_singarr(ctms,WTOKE_STR1 *);
    cleanup_NET_ALIGN();
    if (mf_sel_str.fpout != (FILE *)0 && mf_sel_str.fpout != stdout )
	fclose(mf_sel_str.fpout);
}
 
void proc_args(int argc, char **argv, char *prog, char **hname, char **hfmt, int *nhyps, int *feedback, int *linewidth, int *time_align, int *case_sense, char **out_name, double *alpha, double *null_conf, int *method){
    int opt, fbset=0;

    *hname=(char *)0;
    *hfmt=(char *)0;
    *nhyps = 0;
    *feedback = 1;
    *linewidth = 1000;
    *time_align = 0;
    *case_sense=0;
    *out_name = (char *)0;
    *alpha = 1.0;
    *null_conf = 0.0;
    *method = METH_NULL;

    for (opt = 1; opt < argc && (*(argv[opt]) == '-'); opt++){
	/* printf("Current OP %s\n",argv[opt]); */
	if (strcmp(argv[opt],"-o") == 0){
	    if (argc < opt + 1) do_exit("Not enough Output arguments",prog,1);
	    if (*(argv[opt+1]) == '-') do_exit("Req'd Output Filename",prog,1);
	    *out_name = argv[++opt];
	} else if (strcmp(argv[opt],"-h") == 0){
	    if (argc < opt + 2) do_exit("Not enough Hyp arguments",prog,1);
	    if (*(argv[opt+1]) == '-') do_exit("Req'd Hyp File name",prog,1);
	    hname[*nhyps] = argv[++opt];
	    if (argc >= opt + 2 && *(argv[opt+1]) != '-') 
		hfmt[*nhyps]  = argv[++opt];
	    else
		hfmt[*nhyps] = "trn";
	    (*nhyps)++;
	    if (*nhyps > 1 && 
		strcmp(hfmt[0],hfmt[(*nhyps)-1]) != 0)
	        do_exit("Hyp formats must all be identical\n",prog,1);
	} else if (strcmp(argv[opt],"-a") == 0){
	    if (argc <= opt+1)
		do_exit("Not enough alpha arguments",prog,1);
	    *alpha = atof(argv[++opt]);	
	} else if (strcmp(argv[opt],"-c") == 0){
	    if (argc <= opt+1)
		do_exit("Not enough null_conf arguments",prog,1);
	    *null_conf = atof(argv[++opt]);	
	} else if (strcmp(argv[opt],"-l") == 0){
	    if (argc <= opt+1)
		do_exit("Not enough Line Width arguments",prog,1);
	    *linewidth = atoi(argv[++opt]);
	} else if (strcmp(argv[opt],"-f") == 0){
	    if (argc <= opt+1) do_exit("Not enough Feedback arguments",prog,1);
	    *feedback = atoi(argv[++opt]);
	    fbset = 1;
	} else if (strcmp(argv[opt],"-m") == 0){
	    if (argc <= opt+1) do_exit("Not enough Method arguments",prog,1);
	    if (strcmp(argv[opt+1],"meth1") == 0)
		*method = METH_METH1;
	    else if (strcmp(argv[opt+1],"oracle") == 0)
		*method = METH_ORACLE;
	    else if (strcmp(argv[opt+1],"putat") == 0)
		*method = METH_PUTAT;
	    else if (strcmp(argv[opt+1],"maxconf") == 0)
		*method = METH_MAXCONF;
	    else if (strcmp(argv[opt+1],"avgconf") == 0)
		*method = METH_AVGCONF;
	    else if (strcmp(argv[opt+1],"meth1a") == 0)
		*method = METH_METH1A;
	    else if (strcmp(argv[opt+1],"maxconfa") == 0)
		*method = METH_MAXCONFA;
	    else
		do_exit("Illegal method",prog,1);
	    opt++;
	} else if (strcmp(argv[opt],"-s") == 0){
	    *case_sense = 1;
	} else if (strcmp(argv[opt],"-T") == 0){
	    *time_align = 1;
	} else {
            (void) fprintf(stderr,usage,prog);
            printf("Illegal argument: %s\n",argv[opt]);
            exit(1);
        }
    }

    if (*nhyps <= 1)
        do_exit("Req'd Hyp File names, 2 or more",prog,1);
    if (*out_name == (char *)0)
        do_exit("Req'd Output File name",prog,1);
    if (*method == METH_NULL)
        do_exit("Req'd method",prog,1);
}


void do_exit(char *desc, char *prog, int ret){
    fprintf(stderr,usage,prog);
    fprintf(stderr,"%s: %s\n\n",prog,desc);
    exit(ret);
}

void print_linear(NODE *node, void *p){
    char *proc = "print_linear";
    ARC_LIST_ATOM *parc;
    struct mfalign_best_seq_struct *mf_sel_str;
    int nword, max_word=50, i;
    WORD *wlist[50];
    double word_conf_sum[50], word_t1[50], word_t2[50], word_score[50];
    int word_occ[50],  best_w, Narc, ties;
    double max_score, word_max_conf[50];
    double total_conf;

    mf_sel_str = (struct mfalign_best_seq_struct *)p;
    /* clear the wlist and it's values */
    nword = 0;
    for (i=0; i<max_word; i++){
	wlist[i] = NULL_WORD;
	word_conf_sum[i] = word_t1[i] = word_t2[i] = word_score[i] = 0.0;
	word_max_conf[i] = 0.0;
	word_occ[i] = 0;
    }

    if (node != NULL && strcmp(node->name,"STOP") != 0){
	for (Narc=0, parc=node->out_arcs; parc != (ARC_LIST_ATOM *)0;
	     parc = parc->next)
	    Narc++;
	if (mfdb) {
	    printf("<putative_tag >\n"); 
	    for (parc=node->out_arcs; parc != (ARC_LIST_ATOM *)0;
		 parc = parc->next){
		if (node->out_arcs->arc->to_node != parc->arc->to_node){
		    fprintf(scfp,"Error: Node Violates linear topology\n");
		    exit(1);
		}
		printf("   <attrib tag=\"%s\" t1=\"%f\" t2=\"%f\" "
		       "conf=\"%f\" tag1=\"%s\">\n",
		       ((WORD*)(parc->arc->data))->value,
		       ((WORD*)(parc->arc->data))->T1,
		       ((WORD*)(parc->arc->data))->T2,
		       ((WORD*)(parc->arc->data))->conf,
		       ( ( ((WORD*)(parc->arc->data))->tag1 != (TEXT *)0) ?
			 ((WORD*)(parc->arc->data))->tag1 : (TEXT *)""));
	    }
	    printf("</putative_tag >\n");
	}


	/* Compute the winner based on the formula:
	 */
	total_conf = 0.0;
	for (parc=node->out_arcs; parc != (ARC_LIST_ATOM *)0;
	     parc = parc->next){
	    if (node->out_arcs->arc->to_node != parc->arc->to_node){
		fprintf(scfp,"Error: Node Violates linear topology\n");
		exit(1);
	    }
	    /* Search the wlist for this word */
	    for (i=0;
		 i<nword && parc->arc->net->arc_func.equal(wlist[i],
					     (WORD*)(parc->arc->data)) != 0;
		 i++) ;
	    if (i==nword) {
		if (i == max_word-1){
		    fprintf(scfp,"Error: Too many words in %s\n",proc);
		    exit(1);
		}
		wlist[i] = (WORD*)(parc->arc->data);
		nword++;
	    }
	    /* wlist[i] is the current words data */
	    /* adding info for the word */
	    word_occ[i] ++;
	    word_t1[i] += ((WORD*)(parc->arc->data))->T1;
	    word_t2[i] += ((WORD*)(parc->arc->data))->T2;
#ifdef mmm
	    if (parc->arc->net->arc_func.equal(wlist[i],
                                               mf_sel_str->null_alt)==0){
		if (mf_sel_str->method == METH_METH1 ||
		    mf_sel_str->method == METH_MAXCONF ||
		    mf_sel_str->method == METH_AVGCONF) {
		    word_conf_sum[i] += mf_sel_str->null_conf;
		    total_conf +=  mf_sel_str->null_conf;
		    word_max_conf[i] = MAX(word_max_conf[i],
					   mf_sel_str->null_conf);
		} else if (mf_sel_str->method == METH_METH1A ||
			   mf_sel_str->method == METH_MAXCONFA) {
		    word_conf_sum[i] = word_occ[i];
		    word_max_conf[i] = MAX(word_max_conf[i],
					   (mf_sel_str->null_conf*word_occ[i]/
					    (double)Narc));
		} else if (mf_sel_str->method != METH_ORACLE &&
			   mf_sel_str->method != METH_PUTAT){
		    fprintf(scfp,"Error: undefined method PL1 %d\n",mf_sel_str->method);
		    exit(1);
		}
	    } else {
#endif
                word_conf_sum[i] += ((WORD*)(parc->arc->data))->conf;
		word_max_conf[i] = MAX(word_max_conf[i],
				       ((WORD*)(parc->arc->data))->conf);
		total_conf +=  ((WORD*)(parc->arc->data))->conf;
#ifdef mmm
	    }
#endif
	}	
	/* compute the actual score */
	max_score = -1;
	best_w = -1;
	ties = 0;
	for (i=0; i<nword; i++){
	    if (mf_sel_str->method == METH_METH1 ||	
		mf_sel_str->method == METH_METH1A || 
		mf_sel_str->method == METH_ORACLE) {
		word_score[i] = ( mf_sel_str->alpha * ((double)word_occ[i]/(double)Narc) + 
				  ((1.0-mf_sel_str->alpha) * 
				   (word_conf_sum[i]/word_occ[i])));
		if (mfdb > 5) printf("(%.3f * %.3f) + (%.3f * %.3f)\n",
				     mf_sel_str->alpha,((double)word_occ[i]/(double)Narc),
				     (1.0-mf_sel_str->alpha),
				     (word_conf_sum[i]/word_occ[i]));
	    } else if (mf_sel_str->method == METH_MAXCONF ||
		       mf_sel_str->method == METH_MAXCONFA)
		word_score[i] = ( mf_sel_str->alpha * ((double)word_occ[i]/(double)Narc) +
				  ((1.0-mf_sel_str->alpha) * 
				   word_max_conf[i]));
	    else if (mf_sel_str->method == METH_AVGCONF){
		word_score[i] = ( mf_sel_str->alpha * ((double)word_occ[i]/(double)Narc) +
				  ((1.0-mf_sel_str->alpha) * 
				   word_conf_sum[i] / total_conf));
		if (mfdb > 5) {
		    printf("avgconf word_score[%d] = %f\n",i,word_score[i]);
		    printf("      %.3f * (%.3f / %.3f) + (1 - %.3f) * (%.3f / %.3f)\n",
			   mf_sel_str->alpha,(double)word_occ[i],(double)Narc,
			   mf_sel_str->alpha, word_conf_sum[i],total_conf);
		}
	    } else if (mf_sel_str->method != METH_ORACLE &&
		       mf_sel_str->method != METH_PUTAT){
		fprintf(scfp,"Error: undefined method PL2 %d\n",mf_sel_str->method);
		exit(1);
	    }
	    if (word_score[i] > max_score){
		max_score = word_score[i];
		best_w = i;
		ties = 1;
	    } else if (word_score[i] == max_score)
		ties++;
	}
	if (ties > 1)
	    glob_ties ++;

	/* sanity check */
	if (best_w < 0){
	    fprintf(scfp,"Error: No best word found\n"); exit(1);
	}	    
	/* dump the Wlist */
	if (mfdb > 5){
	    printf("    Word analysis\n");
	    for (i=0; i<nword; i++){
		printf("       %1s word:'%s'  Occ:%d  Conf_sum:%.3f  "
		       "Max_Conf:%.3f  Sum_T1:%.2f  Sum_T2:%.2f  Score:%.2f\n",
		       (best_w == i) ? "*" : " ", wlist[i]->value,
		       word_occ[i], word_conf_sum[i], word_max_conf[i],
		       word_t1[i],word_t2[i],word_score[i]);
	    }
	}
	/* Write the output */
	if (mf_sel_str->putat_format){
	    fprintf(mf_sel_str->fpout,"<putative_tag file=%s chan=%s >\n",
		    mf_sel_str->file,mf_sel_str->chan); 
	    for (parc=node->out_arcs; parc != (ARC_LIST_ATOM *)0;
		 parc = parc->next){
		if (node->out_arcs->arc->to_node != parc->arc->to_node){
		    fprintf(scfp,"Error: Node Violates linear topology\n");
		    exit(1);
		}
		fprintf(mf_sel_str->fpout,
			"   <attrib tag=\"%s\" t1=\"%f\" t2=\"%f\" "
			"conf=\"%f\" tag1=\"%s\">\n",
			((WORD*)(parc->arc->data))->value,
			((WORD*)(parc->arc->data))->T1,
			((WORD*)(parc->arc->data))->T2,
			((WORD*)(parc->arc->data))->conf,
			( ( ((WORD*)(parc->arc->data))->tag1 != (TEXT *)0) ?
			  ((WORD*)(parc->arc->data))->tag1 : (TEXT *)""));
	    }
	    fprintf(mf_sel_str->fpout,"</putative_tag >\n");
	} else if (mf_sel_str->ctm_format){
	    if (mf_sel_str->method == METH_METH1 ||
		mf_sel_str->method == METH_MAXCONF ||
		mf_sel_str->method == METH_METH1A ||
		mf_sel_str->method == METH_MAXCONFA ||
		mf_sel_str->method == METH_AVGCONF){
		if (node->out_arcs->arc->net->arc_func.equal(wlist[best_w],
                        mf_sel_str->null_alt)!=0)
		    fprintf(mf_sel_str->fpout,
			"%s %s %.3f %.3f %s %.6f\n",mf_sel_str->file,
			mf_sel_str->chan,word_t1[best_w]/word_occ[best_w],
			(word_t2[best_w] - word_t1[best_w]) / word_occ[best_w],
			wlist[best_w]->value, 
			word_conf_sum[best_w] / word_occ[best_w]);
	    } else if (mf_sel_str->method == METH_ORACLE){
		if (nword > 1) fprintf(mf_sel_str->fpout,
				       "%s %s * * <ALT_BEGIN>\n",
				       mf_sel_str->file,mf_sel_str->chan);
		for (i=0; i<nword; i++){
		    fprintf(mf_sel_str->fpout,
			    "%s %s %.3f %.3f %s %.6f\n",mf_sel_str->file,
			    mf_sel_str->chan,word_t1[i]/word_occ[i],
			    (word_t2[i] - word_t1[i]) / word_occ[i],
			    wlist[i]->value, 
			    word_conf_sum[i] / word_occ[i]);
		    if (nword > 1)
			fprintf(mf_sel_str->fpout,
			    "%s %s * * %s\n",mf_sel_str->file,
			    mf_sel_str->chan,(i<nword-1)?"<ALT>":"<ALT_END>");

		}
	    } else {
		fprintf(scfp,"Error: undefined output format\n");
		exit(1);
	    }
	}
    }    
}


NETWORK *perform_mfalign_v1(WTOKE_STR1 **ctms, int nctm, int *end, int time_align, double null_conf){
    char *proc = "perform_mfalign";
    NETWORK **nets, *out_net;
    WORD *null_alt;
    int in;

    alloc_singZ(nets,nctm,NETWORK *,(NETWORK *)0);
    null_alt = new_WORD((TEXT *)"@",-1,0.0,0.0,null_conf,(TEXT *)0,(TEXT *)0,0,0,0);
     
    /* create the networks */
    for (in=0; in < nctm; in++){
	if ((nets[in]=
	     Network_create_from_WTOKE(ctms[in],ctms[in]->s,end[in],
				       rsprintf("mfalign net %d",in),
				       print_WORD_wt,
				       equal_WORD2,
				       release_WORD, null_alt_WORD,
				       opt_del_WORD,
				       copy_WORD, make_empty_WORD,
				       use_count_WORD, 1))
	    == NULL_NETWORK){ 
	    fprintf(stderr,"%s: Network_create_from_WTOKE failed\n",proc);
	    exit(1);
	}
	Network_traverse(nets[in],0,0,set_tag1,(void *)ctms[in]->id,0);
    }
    
    /* align the networks */
    Network_dpalign_n_networks(nets,nctm,(!time_align)?wwd_WORD_rover:wwd_time_WORD,
			       &out_net,(void *)null_alt);
    
    /* delete the networks */
    for (in=0; in < nctm; in++)
	Network_destroy(nets[in]);

    free_singarr(nets,NETWORK *);
    release_WORD(null_alt);

    return(out_net);
}


void set_tag1(ARC *arc, void *p)          
{
    if (arc != NULL){
	set_WORD_tag1((WORD *)(arc->data), (TEXT *)p);
    }
}

