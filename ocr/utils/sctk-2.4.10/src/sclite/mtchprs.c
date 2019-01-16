/**********************************************************************/
/*                                                                    */
/*             FILENAME:  matched_pairs.c                             */
/*             BY:  Jonathan G. Fiscus                                */
/*             DATE: April 14 1989                                    */
/*                   NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY   */
/*                   SPEECH RECOGNITION GROUP                         */
/*                                                                    */
/*             DESCRIPTION:  These programs perform the Matched Pairs */
/*                           tests. which include:                    */
/*                                  SEGMENTATION ANALYSIS             */
/*                                  MATCHED PAIRS TEST FOR SYSTEMS    */
/*                                                                    */
/**********************************************************************/


#include "sctk.h"

/**********************************************************************/
/*   structures and defines for the sentence segment list             */
#define MIN_NUM_GOOD		2
#define MAX_SEG_ANAL		10

#ifdef __cplusplus
extern "C" {
#endif

typedef struct segment_count_struct{
    int number_ref;
    int number_errors_for_hyp1;
    int number_errors_for_hyp2;
} SEG;

#ifdef __cplusplus
}
#endif

/* alloc a segment list */
#define alloc_SEG_LIST(_seg,_num) \
   alloc_2dimarr(_seg,_num,1,SEG);

#define expand_SEG_LIST(_seg_l,_num,_max) \
{   int _l;\
    expand_1dim(_seg_l,_num,_max,2.0,SEG *,1); \
    for (_l=_num; _l < _max; _l++) \
      alloc_singarr((_seg_l)[_l],1,SEG); \
}

#define free_SEG_LIST(_seg,_num)   free_2dimarr(_seg,_num,SEG)


/*  macros to access the segment list */
#define seg_n(_sg,_s)		_sg[_s]
#define seg_ref(_sg,_s)		_sg[_s]->number_ref
#define seg_hyp1(_sg,_s)	_sg[_s]->number_errors_for_hyp1
#define seg_hyp2(_sg,_s)	_sg[_s]->number_errors_for_hyp2


void evaluate_SEG(char *sys1_str, char *sys2_str, SEG **seg_l, int *num_seg, int sent_cnt, int min_good, double *s1_pct, double *s2_pct, double *seg_per_sent, int verbose, FILE *fp);
int analyze_Z_score(SCORES *sys1, SCORES *sys2, double mean, double variance, double std_dev, double Z_stat, int verbose);
void test_comp_sents(PATH *snt1, PATH *snt2, SEG ***seg_l, int *num_seg, int *max_seg, int min_good);
void count_seg(PATH *snt1, PATH *snt2, int beg1, int beg2, int end1, int end2, SEG ***seg_l, int *num_seg, int *max_seg, int dbg);
void seg_count(PATH *snt, int from, int to, int *err, int *ref);
void print_sent_seg_averaged_analysis(SCORES *scor[], int nscor, double ***seg_per_sent, char *tname);
void print_sent_seg_long_analysis(SCORES *scor[], int nscor, double ***seg_per_sent, char *tname);
void print_compare_matrix_for_sys(SCORES *scor[], int nscor, int **winner, char *matrix_name, char *tname, char *v_desc, double **sys1_pct_arr, double **sys2_pct_arr, int **num_seg, double **Z_stat, int min_num_good, FILE *fp);

/**********************************************************************/
/*   this procedure is written to analyze the segmentation for        */
/*   varying numbers on the minimum number of good words to bound     */
/*   the beginning and the ends of the segments                       */
/**********************************************************************/
void do_mtch_pairs_seg_analysis(SCORES *scor[], int nscor, char *t_name, int seg_ave, int seg_long)
{
    int sys1, sys2, min_g;
    int result;
    double conf;
    int **num_seg, **max_seg;
    double **sys1_pct_arr, **sys2_pct_arr, **Z_stat_arr, ***seg_per_sent;

    /* do all allocations */
    fprintf(stderr,"Doing segmentation analysis. This is slow so . . . \n\n");
    alloc_2dimarr(num_seg,nscor,nscor,int);
    alloc_2dimZ(max_seg,nscor,nscor,int,100);
    alloc_2dimarr(sys1_pct_arr,nscor,nscor,double);
    alloc_2dimarr(sys2_pct_arr,nscor,nscor,double);
    alloc_2dimarr(Z_stat_arr,nscor,nscor,double);
    alloc_3dimZ(seg_per_sent,MAX_SEG_ANAL,nscor,nscor,double,0.0);

    /* loop for min_good, and compared systems */
    for (min_g=0; min_g < MAX_SEG_ANAL; min_g++){
        fprintf(stderr,"   calculating buffer of  %d\n",min_g);
        for (sys1=0; sys1 <(nscor-1); sys1++)
            for (sys2=sys1+1; sys2<(nscor); sys2++){
                result = do_mtch_pairs_on_sys(scor, nscor,sys1,sys2,
                                              &(sys1_pct_arr[sys1][sys2]),
                                              &(sys2_pct_arr[sys1][sys2]),
                                              &(num_seg[sys1][sys2]),
                                              &(max_seg[sys1][sys2]),
                                              &(Z_stat_arr[sys1][sys2]),
                                           &(seg_per_sent[min_g][sys1][sys2]),
					      min_g+1,FALSE,stdout,&conf);
        }
    }

    /* if the flag, print the requested analysis */
    if (seg_long)
        print_sent_seg_long_analysis(scor,nscor,seg_per_sent,t_name);
    if (seg_ave)
        print_sent_seg_averaged_analysis(scor,nscor,seg_per_sent,t_name);

    free_2dimarr(num_seg,nscor,int);
    free_2dimarr(max_seg,nscor,int);
    free_2dimarr(sys1_pct_arr,nscor,double);
    free_2dimarr(sys2_pct_arr,nscor,double);
    free_2dimarr(Z_stat_arr,nscor,double);
    free_3dimarr(seg_per_sent,MAX_SEG_ANAL,nscor,double);
}


/**********************************************************************/
/*   this procedure performs a single matched paired analysis for the */
/*   one level of the min_good and returns the results in the winner  */
/*   array                                                            */
/**********************************************************************/
void do_mtch_pairs(SCORES *scor[], int nscor, char *min_num_good_str, char *test_name, int print_report, int verbose, int ***out_winner, char *outroot, int feedback, double ***out_conf)
{
    int sys1, sys2;
    int result, **winner;
    double **conf;
    int *sum_winner, **num_seg, min_num_good, **max_seg;
    double **sys1_pct_arr, **sys2_pct_arr, **Z_stat_arr, **seg_per_sent;
    char *matrix_name = "COMPARISON MATRIX: FOR THE MATCHED PAIRS TEST";
    char *mean_desc = "PECENTAGES ARE MEAN PCT ERROR/SEGMENT.  FIRST # IS LEFT SYSTEM";
    FILE *fp = stdout;
    
    if (print_report || verbose){
	char *f = rsprintf("%s.mapsswe",outroot);
	if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(f,"w")) ==
	    (FILE *)0){
	    fprintf(stderr,"Warning: Open of %s for write failed.  "
		           "Using stdout instead.\n",f);
	    fp = stdout;
	} else
	    if (feedback >= 1) printf("        Output written to '%s'\n",f);
    }
    if ((min_num_good = atoi(min_num_good_str)) <= 0){
        fprintf(stderr,"Warning: Minimum seperation by correct words is \n");
        fprintf(stderr,"         too low, setting to %d\n",
                       DEFAULT_MIN_NUM_GOOD);
        min_num_good = DEFAULT_MIN_NUM_GOOD;
    }
    if (!print_report) verbose = FALSE;

    /* all allocations */
    alloc_2dimarr(num_seg,nscor,nscor,int);
    alloc_2dimZ(max_seg,nscor,nscor,int,100);
    alloc_2dimarr(sys1_pct_arr,nscor,nscor,double);
    alloc_2dimarr(sys2_pct_arr,nscor,nscor,double);
    alloc_2dimarr(Z_stat_arr,nscor,nscor,double);
    alloc_2dimZ(seg_per_sent,nscor,nscor,double,0.0);
    alloc_singZ(sum_winner,nscor,int,NO_DIFF);
    alloc_2dimZ(winner,nscor,nscor,int,NO_DIFF);
    *out_winner = winner;
    alloc_2dimZ(conf,nscor,nscor,double,0.0);
    *out_conf = conf;

    /* loop each pair of compared systems */
    for (sys1=0; sys1 <(nscor-1); sys1++)
        for (sys2=sys1+1; sys2<(nscor); sys2++){
            result = do_mtch_pairs_on_sys(scor,nscor,sys1,sys2,
                                          &(sys1_pct_arr[sys1][sys2]),
                                          &(sys2_pct_arr[sys1][sys2]),
                                          &(num_seg[sys1][sys2]),
                                          &(max_seg[sys1][sys2]),
                                          &(Z_stat_arr[sys1][sys2]),
                                          &(seg_per_sent[sys1][sys2]),
                                          min_num_good,verbose,fp,
					  &(conf[sys1][sys2]));
            winner[sys1][sys2] = result;
            sum_winner[sys2] += result * (-1);
            sum_winner[sys1] += result;
	  }

    /* if print_report print the requested analysis */
    if (print_report){
	if (verbose) form_feed(fp);
	print_compare_matrix_for_sys(scor,nscor,winner,matrix_name,test_name,
				     mean_desc,sys1_pct_arr,sys2_pct_arr,
				     num_seg,Z_stat_arr,min_num_good,fp);
	if (fp == stdout) form_feed(fp);
    }

    free_2dimarr(sys1_pct_arr,nscor,double);
    free_2dimarr(sys2_pct_arr,nscor,double);
    free_2dimarr(Z_stat_arr,nscor,double);
    free_2dimarr(seg_per_sent,nscor,double);
    free_singarr(sum_winner,int);
    free_2dimarr(num_seg,nscor,int);
    free_2dimarr(max_seg,nscor,int);
    /* Big Error     free_2dimarr(conf,nscor,double); */

    if (fp != stdout) fclose(fp);
}

/**********************************************************************/
/*   the guts of the Matched Pairs program,  this procedure compares  */
/*   2 systems in a SYS_ALIGN_LIST with indexes sys1_ind and sys2_ind */
/*   then goes through each matching sentence then calls another      */
/*   procedure to find the segments and store them in the segment     */
/*   list. after all sentences are compared, the segment list is      */
/*   analyzed                                                         */
/**********************************************************************/
int do_mtch_pairs_on_sys(SCORES *scor[], int nscor, int sys1_ind, int sys2_ind, double *sys1_pct, double *sys2_pct, int *num_seg, int *max_seg, double *Z_stat, double *seg_per_sent, int min_num_good, int verbose,FILE *fp, double *conf)
{
    int spk1, spk2, snt1, snt2, i, sent_cnt=0, *err_diff;
    double mean, std_dev, variance, median;
    SCORES *sys1, *sys2;
    SEG **seg_l;

    *num_seg = 0;
    sys1 = scor[sys1_ind];
    sys2 = scor[sys2_ind];

    alloc_SEG_LIST(seg_l,*max_seg);

    for (spk1=0;spk1 < sys1->num_grp; spk1++){ /* for all speaker sys1 */
        /**** find the matching speaker */
        for (spk2=0;spk2 < sys2->num_grp; spk2++)
            if (strcmp(sys1->grp[spk1].name, sys2->grp[spk2].name) == 0)
                break;
        /**** the the speakers match, start on the sentences */
        if (spk2 != sys2->num_grp){
            /**** for all sents in sys1,spkr1 */
            for (snt1 = 0; snt1 < sys1->grp[spk1].num_path; snt1++){
                /**** for all sents in sys2,spkr2 */
                for (snt2 = 0; snt2 < sys2->grp[spk2].num_path; snt2++) 
                   /**** if the sentences are the same, compare them */
                   if(strcmp(sys1->grp[spk1].path[snt1]->id,
                             sys2->grp[spk2].path[snt2]->id) == 0){
                        test_comp_sents(sys1->grp[spk1].path[snt1],
					sys2->grp[spk2].path[snt2],
					&seg_l,num_seg, max_seg, min_num_good);
                        sent_cnt++;
                        break;
		   }
	    }
	} else {
            fprintf(stderr,"Warning: Speaker %s is in system %s but not system %s\n",
		    sys1->grp[spk1].name,sys1->title,sys2->title);
        }
    }
    /* calculate the percentages of errors for both systems and print */
    /* a report if verbose == TRUE                                    */
    evaluate_SEG(sys1->title, sys2->title,seg_l,num_seg,sent_cnt,min_num_good,
		 sys1_pct,sys2_pct,seg_per_sent,verbose,fp);

    alloc_singZ(err_diff,*max_seg,int,0);

    /* calc an arr with the difference in the errors for each segment */
    for (i=0;i<*num_seg;i++)
        err_diff[i] = seg_hyp1(seg_l,i) - seg_hyp2(seg_l,i);

    calc_mean_var_std_dev_Zstat(err_diff,*num_seg, 
                                &mean,&variance,&std_dev,&median,Z_stat);
    /* Special output routine */
    if (verbose){  
        fprintf(fp,"MTCH_PR_RESULTS (systems: %s %s) (# segs: %d) "
		"(Seg per Sent: %5.3f) ",
	       sys1->title,sys2->title,*num_seg,
	       *seg_per_sent);
        fprintf(fp,"(%% Error: %5.2f %5.2f) (mean: %5.3f) (std dev: %5.3f) "
		"(Z Stat: %5.3f) (Stat Diff: %s)\n",
	       *sys1_pct,*sys2_pct,mean,std_dev,*Z_stat,
	       (analyze_Z_score(sys1,sys2,mean,variance,std_dev,
				*Z_stat,SILENT) == NO_DIFF) ? "No" : "Yes"); 
    }

    free_singarr(err_diff,int);
    free_SEG_LIST(seg_l,*max_seg);
    *conf = 1.0 - (2.0 * (normprob((double) (fabs(*Z_stat))) - 0.50));

    return(analyze_Z_score(sys1,sys2,mean,variance,std_dev,*Z_stat,SILENT));
}

/**********************************************************************/
/*   this procedure calculates the percentages of errors for each     */
/*   systems segments,  if verbose == TRUE then a report is printed   */
/*   to stdout for looking at segment counts                          */
/**********************************************************************/
void evaluate_SEG(char *sys1_str, char *sys2_str, SEG **seg_l, int *num_seg, int sent_cnt, int min_good, double *s1_pct, double *s2_pct, double *seg_per_sent, int verbose,FILE *fp)
{
    int i, tot_ref=0, tot_err_h1=0, tot_err_h2=0;

    /* calc the pcts */   
    for (i=0;i<*num_seg;i++){
        tot_ref += seg_ref(seg_l,i);
        tot_err_h1 += seg_hyp1(seg_l,i);
        tot_err_h2 += seg_hyp2(seg_l,i);
    }
    (*s1_pct) = pct(tot_err_h1,tot_ref);
    (*s2_pct) = pct(tot_err_h2,tot_ref);
    (*seg_per_sent) = (double)(*num_seg) / (double)sent_cnt;

    /* if needed, print the systems segmentation reports */
    if (verbose){
      fprintf(fp,"      SEGMENTATION REPORT FOR SYSTEMS\n");
      fprintf(fp,"              %4s and %4s\n",sys1_str, sys2_str);
      fprintf(fp,"Minimum Number of Correct Boundary words %1d\n\n",min_good);
      fprintf(fp,"  Number of Segments %2d,     %5.3f per sentence\n",*num_seg,
                              (*seg_per_sent));
      fprintf(fp,"  Number of Sentences %2d\n\n",sent_cnt);
      fprintf(fp,"    # Ref wrds      Err %4s   Err %4s\n",sys1_str, sys2_str);
      fprintf(fp,"---------------------------------------\n");
      if (FALSE){
          for (i=0;i<*num_seg;i++)
              fprintf(fp,"       %4d           %4d        %4d\n",
                  seg_ref(seg_l,i),seg_hyp1(seg_l,i),seg_hyp2(seg_l,i));
          fprintf(fp,"---------------------------------------\n");
      }
      fprintf(fp,"Totals %4d           %4d        %4d\n\n",
                 tot_ref,tot_err_h1,tot_err_h2);
      fprintf(fp,"Pct err              %5.1f%%      %5.1f%%\n\n",
                 *s1_pct,*s2_pct);
      fprintf(fp,"\n\n\n");

      /* do a McNermar-style anaylsys */ 
      /* first calc the matrix */
      { 
	int **table=NULL, ans; double mcn_conf=0.0;
	alloc_2dimZ(table,2,2,int,0);
	for (i=0;i<*num_seg;i++){
	  if (seg_hyp1(seg_l,i) == 0 && seg_hyp2(seg_l,i) == 0 &&
	      seg_ref(seg_l,i) == 0){
	    table[0][0] ++;  /* THIS WILL NEVER HAPPEN */
	    fprintf(stderr,"Warning: MPESM test identified a correct/correct segment- %d\n",i);
	  }
	  else if (seg_hyp1(seg_l,i) > 0 && seg_hyp2(seg_l,i) == 0)
	    table[1][0] ++;
	  else if (seg_hyp1(seg_l,i) == 0 && seg_hyp2(seg_l,i) > 0)
	    table[0][1] ++;
	  else 
	    table[1][1] ++;
	}
	ans = do_McNemar(table,sys1_str,sys2_str,verbose,fp,&mcn_conf);
	
	free_2dimarr(table,2,int);
      }
    }
    if (0) { int i;
      fprintf(fp,"Segment size Report\n");
      for (i=0;i<*num_seg;i++){
	fprintf(fp, "   Segment:  #ref=%d    sys1Err=%d   sys2Err=%d errDiff=%d\n",
		seg_ref(seg_l,i), seg_hyp1(seg_l,i), seg_hyp2(seg_l,i),abs(seg_hyp1(seg_l,i) - seg_hyp2(seg_l,i)));
      }
    }
}


/**********************************************************************/
/*   this function returns whether or not the Z_stat is outside the   */
/*   range of significance, if needed, a verbose report is printed    */
/**********************************************************************/
int analyze_Z_score(SCORES *sys1, SCORES *sys2, double mean, double variance, double std_dev, double Z_stat, int verbose)
{
    int eval;
    char pad = '\0';

    if (verbose){
      printf("\n%s\t             Analysis of matched pairs comparison\n\n",&pad);
      printf("%s\t\t\t   Comparing %s and %s\n\n",&pad,
	     sys1->title, sys2->title);
      printf("%s\t       H : means of sentence segment errors are equal.\n",&pad);
      printf("%s\t        0\n\n",&pad);
      eval = print_Z_analysis(Z_stat);
    }
   else
      eval = Z_pass(Z_stat);
   /* if the mean is less then we want the sys1 index to be the winner */
   if (mean < 0.0)
       eval *= (-1);
   return(eval);
}

/**********************************************************************/
/*   this procedure tries to break up sentences into segments and then*/
/*   counts and stores the segment errors and size for later          */
/*   processing.  The algorithm is a modified state transition graph  */
/*   that looks like this:                                            */
/*                                                                    */
/*      the beginning state is b:                                     */
/*      corr means both sentences have the current word correct:      */
/*      error means at least one sentence had an error:               */
/*                              _                                     */
/*                             / \ corr                               */
/*                             v /                                    */
/*                              b <--------------.                    */
/*                              |                |                    */
/*                              |  error         |                    */
/*                              |                |                    */
/*                              V                |                    */
/*        ,-------------------> e<-. error       |                    */
/*        |                     |\_/             |                    */
/*        |                     |                |                    */
/*        |         ,---------> |  good          |                    */
/*        |         |           V                |                    */
/*        |         |  if #good == min_good      |                    */
/*   error|     corr|  then store the segment ---'                    */
/*        |         |           |                                     */
/*        |         |           V                                     */
/*        |         `---------- g                                     */
/*        |                     |                                     */
/*        |                     |                                     */
/*        `---------------------'                                     */
/* Modified: Nov 10, 1994                                             */
/*   The synchronization between reference strings is lost if altern- */
/*   ation is use in the reference strings.  To correct the problem,  */
/*   code was added to re-sync the reference strings if needed.       */
/*   if the sync passes, the state is modified to dependent in the    */
/*   words skipped to for re-synchronization.                         */
/**********************************************************************/
void test_comp_sents(PATH *snt1, PATH *snt2, SEG ***seg_l, int *num_seg, int *max_seg, int min_good)
{
    int ind1=0, ind2=0, eog1, eog2, state, num_good=0, dbg = 0;

    eog1 = eog2 = 0;
    state = 'b';

    if (dbg){
	printf("--------------------------------------------------\n");
	printf("   number of adjacent good words %d\n\n",min_good);
	PATH_print(snt1,stdout,100);
	PATH_print(snt2,stdout,100);
    }
    while (ind1 < snt1->num){
	if (dbg) printf("top of loop: state: %c  ind1=%d ind2=%d\n",
			state,ind1,ind2);
        /* if either sent has an insertion mark it and goto state 'e' */
        while ((ind1 < snt1->num) && snt1->pset[ind1].eval == P_INS)
            ind1++, state = 'e';
        while ((ind2 < snt2->num) && snt2->pset[ind2].eval == P_INS)
            ind2++, state = 'e';
	if (dbg) printf("  rm insert: state: %c  ind1=%d ind2=%d\n",
			state,ind1,ind2);
	/*  Check to see if there is a synchony error, and fix it ONLY */
	/*  if the evaluation of the words are different               */
	if (((ind1 < snt1->num && snt1->pset[ind1].a_ptr == NULL) ^
	     (ind2 < snt2->num && snt2->pset[ind2].a_ptr == NULL)) ||
	    ((ind1 < snt1->num && ind2 < snt2->num) &&
	     (strcmp(snt1->pset[ind1].a_ptr,snt2->pset[ind2].a_ptr) != 0) &&
	     (snt1->pset[ind1].eval != snt2->pset[ind2].eval))){

	    int i1=ind1, i2=ind2, synched=FALSE;
	    if (dbg) printf("  Warning: Reference words out of alignment\n");
	    if (dbg) printf("           Attempting to correct\n");
	    /* the algorithm: trace the following network in a depth first  */
	    /* manner until the first synchronization occurs:               */
	    /*   Where A1 is the ref word s_ref_wrd(snt1,i1) . . .          */
	    /*      A1  A2  A3  A4  ||  A1  A2  A3  A4  ||  A1  A2  A3  A4  */
	    /*      | \             ||     /| \         ||    ____//| \     */
            /*      |  \            ||   /  |  \        ||   /    / |  \    */
	    /*      B1  B2  B3  B4  ||  B1  B2  B3  B4  ||  B1  B2  B3  B4  */
	    while (i1 < snt1->num && !synched){
		if (dbg) printf("    Starting at %d - %d\n",i1,i2);
		while (i2 < snt2->num &&
		       (i2 <= (ind2+1+(i1-ind1))) && !synched){
		    if (dbg) printf("        compare %d to %d  ",i1,i2);
		    if (strcmp(snt1->pset[i1].a_ptr,snt2->pset[i2].a_ptr) != 0)
			synched = TRUE;
		    if (!synched) {
			if (dbg) printf("Not synched\n");
			i2++;
		    } else
			if (dbg) printf("Synched\n");
		}
		if (!synched)
		    i1++;
	    }
	    if (!synched) {
		if (dbg) printf("   I FAILED to synchronize, I'm sorry\n");
	    } else {
		char tstate = state;
		int ii1, ii2;
		for (ii1 = ind1; ii1 <= i1; ii1++)
		    if (snt1->pset[ii1].eval != P_CORR) state = 'e';
		for (ii2 = ind2; ii2 <= i2; ii2++)
		    if (snt2->pset[ii2].eval != P_CORR) state = 'e';
		ind1 = i1; ind2 = i2;
		if (dbg) {
		    printf("   Synch WORKED, at %d - %d\n",ind1,ind2);
		    printf("       State was '%c' now '%c'\n",tstate,state);
		}
	    }
	}
        switch (state){
	  case 'b':
	    /* this state means we haven't found any errors yet */
               /* if both corr, mark the end of the good(eog) for each sent*/
	    if (ind1 < snt1->num && snt1->pset[ind1].eval == P_CORR &&
		ind2 < snt2->num && snt2->pset[ind2].eval == P_CORR)
		eog1 = ind1, eog2 = ind2;
	    else
		state = 'e';
	    break;
	  case 'e':
	    /*  if both are correct, then check to see if the num_good */
	    /*  equals the minimum if it is mark the seg and goto 'b'  */
	    if (ind1 < snt1->num && snt1->pset[ind1].eval == P_CORR &&
		ind2 < snt2->num && snt2->pset[ind2].eval == P_CORR) {
		num_good = 1;
		if (num_good == min_good){
		    if (dbg) printf("    segment: snt1:%d-%d  snt2:%d-%d\n",
				    eog1-(min_good-1),ind1,eog2-(min_good-1),ind2);
		    count_seg(snt1,snt2,eog1-(min_good-1),
			      eog2-(min_good-1),ind1,ind2,
			      seg_l, num_seg, max_seg, dbg);
		    eog1 = ind1, eog2 = ind2, state = 'b';
		}
		else
		    state = 'g';
	    }
	    break;
	  case 'g':
	    /* count the good words until the minimum number is found*/
	    if (ind1 < snt1->num && snt1->pset[ind1].eval == P_CORR &&
		ind2 < snt2->num && snt2->pset[ind2].eval == P_CORR){
		num_good++;
		if (num_good == min_good){
		    if (dbg) {
			printf("    segment: snt1:%d-%d  snt2:%d-%d\n",
			       eog1-(min_good-1),ind1,eog2-(min_good-1),ind2);
		    }
		    count_seg(snt1,snt2,eog1-(min_good-1),
			      eog2-(min_good-1),ind1,ind2,seg_l,num_seg,
			      max_seg,dbg);
		    eog1 = ind1, eog2 = ind2, state = 'b';
		}
		else
		    state = 'g';
	    }
	    /* an error was found before the num_good == min_good */
	    else
		state = 'e';
	    break;
	}
    
	if (ind1 < snt1->num)       ind1++;
	if (ind2 < snt2->num)	    ind2++;
	if (dbg) printf("end of loop: state: %c  ind1=%d ind2=%d\n\n",state,ind1,ind2);
    }
    if (dbg) printf(    "outsid loop: state: %c  ind1=%d ind2=%d\n\n",state,ind1,ind2);
    /*  if the state is anything but 'b' then the end of the sentence ends */
    /*  the segment */
    if (state != 'b' || (state == 'b' && snt2->num > ind2)){
	if (dbg) {
	    printf("    segment: snt1:%d-%d  snt2:%d-%d\n",
		   eog1-(min_good-1),snt1->num-1,eog2-(min_good-1),snt2->num-1);
	}
        count_seg(snt1,snt2,eog1-(min_good-1), eog2-(min_good-1),snt1->num-1,
		  snt2->num-1,seg_l,num_seg,max_seg,dbg);
    }
}

/**********************************************************************/
/*   this procedure only stores the errors and reference words in     */
/*   the segment then it stores the info in a list for segments       */
/**********************************************************************/
void count_seg(PATH *snt1, PATH *snt2, int beg1, int beg2, int end1, int end2, SEG ***seg_l, int *num_seg, int *max_seg, int dbg)
{
/*     static char *ref_str=NULL, **hyp_strs;*/
    int err1, err2, ref1, ref2;

    /* do bounds checking, if indexes are < 0 */
    if (beg1 < 0) beg1 = 0;
    if (beg2 < 0) beg2 = 0;

    if (*num_seg >= *max_seg)
	expand_SEG_LIST(*seg_l,*num_seg,*max_seg);

    seg_count(snt1,beg1,end1,&err1,&ref1);
    seg_count(snt2,beg2,end2,&err2,&ref2);

    seg_ref((*seg_l),*num_seg) = (ref1 + ref2)/2;
    seg_hyp1((*seg_l),*num_seg) = err1;
    seg_hyp2((*seg_l),*num_seg) = err2;

    if (dbg)
      printf("Count Segment: segment# %3d:  sys1=[beg=%d,end=%d,ref=%d,err=%d] sys2=[beg=%d,end=%d,ref=%d,err=%d]\n",
	     *num_seg,beg1,end1,ref1,err1,beg2,end2,ref2,1,err2);

    if (err1 == 0 && err2 == 0){
      fprintf(scfp,"Warning: MAPSSWE segmentation produced a segment with"
	      " errors\n         ignoring segment\n");
    }

    /* print the sentence segments to stdout */
    if (dbg){
	if (ref1 != ref2){
	      printf("***** Reference Counts Not Equal\n");
	      printf("Segment# %3d:  #ref %1d  (%d+%d)/2\n",
		   *num_seg,(ref1+ref2)/2,ref1,ref2);
	} else
	    printf("Segment# %3d:  #ref %1d\n",*num_seg,ref1);

	printf("  #Err.  %2d\n",err1);
	PATH_n_print(snt1, stdout, beg1, end1+1, 100);
	
	printf("  #Err.  %2d\n",err2);
	PATH_n_print(snt2, stdout, beg2, end2+1, 100);	
	printf("\n\n");
    }
    if (! (err1 == 0 && err2 == 0))
      (*num_seg)++;
}

/**********************************************************************/
/* this routine counts the errors and ref words in a segment          */
/**********************************************************************/
void seg_count(PATH *snt, int from, int to, int *err, int *ref)
{
    int i;

    *err = *ref = 0;
    for (i=from;i<= to;i++){
        if (snt->pset[i].eval != P_CORR)
           (*err)++;
        if (snt->pset[i].eval != P_INS)
           (*ref)++;
    }
}

/**********************************************************************/
/*   when the average segment analysis is ran, this procedure prints  */
/*   the report, for each number of minimum good words.  this report  */
/*   prints the average number of segments per sentence for each      */
/*   number of buffer words (min_good)                                */
/**********************************************************************/
void print_sent_seg_averaged_analysis(SCORES *scor[], int nscor, double ***seg_per_sent, char *tname)
{
    int sys1,sys2,num_g,num_comp;
    double tot_sps;
    char pad[FULL_SCREEN];
    char *sub_title="Average Number of Segments Per Sentence Report for";
    char *sub_title1="Changing Numbers of Buffer Words";

    set_pad_cent_n(pad,strlen(sub_title), FULL_SCREEN);
    printf("%s%s\n",pad,sub_title);
    set_pad_cent_n(pad,strlen(sub_title1), FULL_SCREEN);
    printf("%s%s\n",pad,sub_title1);
    if (*tname != '\0'){
        set_pad(pad,tname, FULL_SCREEN);
        printf("%s%s\n",pad,tname);
    }
    printf("\n\n");

    set_pad(pad,"--------------------------------", FULL_SCREEN);
    printf("%s--------------------------------\n",pad);
    printf("%s|   Num Buf   |  Ave Seg/Sent  |\n",pad);
    printf("%s|-------------+----------------|\n",pad);
    for (num_g=0; num_g<MAX_SEG_ANAL; num_g++){
        num_comp = tot_sps = 0;
        for (sys1=0;sys1<nscor;sys1++)
            for (sys2=sys1+1;sys2<nscor;sys2++)
                num_comp++, tot_sps+= seg_per_sent[num_g][sys1][sys2];
        printf("%s|      %2d     |     %5.3f      |\n",
                      pad,num_g+1,(tot_sps/num_comp));
    }
    printf("%s--------------------------------\n",pad);
    form_feed(stdout);
}

/**********************************************************************/
/*   when the long segment analysis is ran, this procedure prints     */
/*   the  report, for each number of minimum good words.  this report */
/*   prints the number of segments per sentence for each system       */
/*   by system comparison                                             */
/**********************************************************************/
void print_sent_seg_long_analysis(SCORES *scor[], int nscor, double ***seg_per_sent, char *tname)
{
    int i,j,k;
    char pad[FULL_SCREEN];
    char *sub_title="Number of Segments Per Sentence Report for Changing";
    char *sub_title1="Numbers of Buffer Words";

    set_pad_cent_n(pad,strlen(sub_title), FULL_SCREEN);
    printf("%s%s\n",pad,sub_title);
    set_pad_cent_n(pad,strlen(sub_title1), FULL_SCREEN);
    printf("%s%s\n",pad,sub_title1);
    if (*tname != '\0'){
        set_pad(pad,tname, FULL_SCREEN);
        printf("%s%s\n",pad,tname);
    }
    printf("\n");

    set_pad_cent_n(pad,(nscor+1) * 15, FULL_SCREEN);
    printf("\n%s|-------------",pad);
    for (k=0;k<nscor;k++)
        printf("%13s","--------------");
    printf("|\n");

    printf("%s|             |",pad);
    for (i=0;i<nscor;i++)
        printf("     %4s    |",scor[i]->title);
    printf("\n");

    for (i=0;i<nscor;i++){
        printf("%s|-------------",pad);
        for (k=0;k<nscor;k++)
            printf("+-------------");
        printf("|\n");
        printf("%s|  %4s    0  |",pad,scor[i]->title);
        for (j=0;j<nscor;j++)
            if (j > i)
                printf("    %5.3f    |",seg_per_sent[0][i][j]);
            else
                printf("             |");
        printf("\n");
        if (MAX_SEG_ANAL >=2)
            for (k=1; k<MAX_SEG_ANAL; k++){
                printf("%s|         %2d  |",pad,k);
                for (j=0;j<nscor;j++)
                    if (j > i)
                        printf("    %5.3f    |",seg_per_sent[k][i][j]);
                    else
                        printf("             |");
                printf("\n");
	    }
    }
    printf("%s|-------------",pad);
    for (k=0;k<nscor;k++)
        printf("%11s","--------------");
    printf("|\n");
    form_feed(stdout);
}

/**********************************************************************/
/*   this procedure prints the Matched pairs report for a single      */
/*   min_num_good length.                                             */
/**********************************************************************/
void print_compare_matrix_for_sys(SCORES *scor[], int nscor, int **winner, char *matrix_name, char *tname, char *v_desc, double **sys1_pct_arr, double **sys2_pct_arr, int **num_seg, double **Z_stat, int min_num_good, FILE *fp)
{
    int i,j,k,block_size,sys,max_name_len=0;
    int hyphen_len=49, space_len=49;
    char pad[FULL_SCREEN], hyphens[50], spaces[50], sysname_fmt[40];
    char *min_good_title="Minimum Number of Correct Boundary words";

/* init the hyphens and spaces array */
    for (i=0; i<hyphen_len; i++){
         hyphens[i]='-'; 
         spaces[i]=' ';
    }
    hyphens[hyphen_len]='\0'; spaces[space_len]='\0';
    /* find the largest system length */
    for (sys=0;sys<nscor;sys++)
      if ((i=strlen(scor[sys]->title)) > max_name_len)
         max_name_len = i;
    block_size = (3+max_name_len);
    sprintf(sysname_fmt," %%%ds |",max_name_len);

    set_pad(pad,matrix_name, FULL_SCREEN);
    fprintf(fp,"\n\n\n%s%s\n",pad,matrix_name);
    if (*v_desc != '\0'){
        set_pad(pad,v_desc, FULL_SCREEN);
        fprintf(fp,"%s%s\n",pad,v_desc);
    }
    if (*tname != '\0'){


        set_pad(pad,tname, FULL_SCREEN);
        fprintf(fp,"%s%s\n",pad,tname);
    }
    set_pad_cent_n(pad,strlen(min_good_title)+2, FULL_SCREEN);
    fprintf(fp,"%s%s %1d\n",pad,min_good_title,min_num_good);
    fprintf(fp,"\n");

    set_pad_cent_n(pad,(nscor+1) * block_size, FULL_SCREEN);
    fprintf(fp,"\n%s|%s",pad,hyphens+hyphen_len-MIN(hyphen_len,(block_size-1)));
    for (k=0;k<nscor;k++)
        fprintf(fp,"%s",hyphens+hyphen_len-MIN(hyphen_len,block_size));
    fprintf(fp,"|\n");

    fprintf(fp,"%s|%s|",pad,spaces+space_len-MIN(space_len,(block_size-1)));
    for (i=0;i<nscor;i++)
        fprintf(fp,sysname_fmt,scor[i]->title);
    fprintf(fp,"\n");

    for (i=0;i<nscor;i++){
        fprintf(fp,"%s|%s",pad,hyphens+hyphen_len-MIN(hyphen_len,(block_size-1)));
        for (k=0;k<nscor;k++)
            fprintf(fp,"+%s",hyphens+hyphen_len-MIN(hyphen_len,(block_size-1)));
        fprintf(fp,"|\n");
        fprintf(fp,"%s|",pad);
        fprintf(fp,sysname_fmt,scor[i]->title);
        for (j=0;j<nscor;j++){
            char *name="";
            int t=0;
            if (j > i){
                if (winner[i][j] == TEST_DIFF)
                    name=scor[j]->title;
                else if (winner[i][j] == NO_DIFF)
                    name="same";
                else
                    name=scor[i]->title;
                t = (block_size-1-strlen(name))/2;
            }
            fprintf(fp,"%s%s%s|",spaces+space_len-MIN(space_len,t),name,
                             spaces+space_len-MIN(space_len,(block_size-1-strlen(name)-t)));
        }
        fprintf(fp,"\n");
#ifdef you_want_more_output
        fprintf(fp,"%s|             |",pad);
        for (j=0;j<nscor;j++)
            if (j > i)
                    fprintf(fp," %3d  %6.2f |",num_seg[i][j],Z_stat[i][j]);
            else
                fprintf(fp,"             |");
        fprintf(fp,"\n");
        fprintf(fp,"%s|             |",pad);
        for (j=0;j<nscor;j++)
            if (j > i)
                fprintf(fp," %4.1f%% %4.1f%% |",sys1_pct_arr[i][j],
                                             sys2_pct_arr[i][j]);
            else
                fprintf(fp,"             |");
        fprintf(fp,"\n");
#endif
    }
    fprintf(fp,"%s|%s",pad,hyphens+hyphen_len-MIN(hyphen_len,(block_size-1)));
    for (k=0;k<nscor;k++)
        fprintf(fp,"%s",hyphens+hyphen_len-MIN(hyphen_len,block_size));
    fprintf(fp,"|\n");
}

