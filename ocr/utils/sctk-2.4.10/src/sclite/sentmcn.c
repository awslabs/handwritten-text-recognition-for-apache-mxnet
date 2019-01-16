/********************************************************************/
/*                                                                  */
/*           FILE: McNemar_sent.c                                   */
/*           WRITTEN BY: Jonathan G. Fiscus                         */
/*           DATE: May 31 1989                                      */
/*                 NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY   */
/*                 SPEECH RECOGNITION GROUP                         */
/*                                                                  */
/*           USAGE: This program, performs a McNemar test on the    */
/*                  list of SYS_ALIGN structures                    */
/*                  The test statistics models the CHI SQUARE dist  */
/*                  and is calculated from the McNemar matrix:      */
/*                                                                  */
/*                McNemar matrix:                                   */
/*                          sys1            C: denotes a corr sent  */
/*                         C     E          E: denotes a error sent */
/*                      -------------       TC: sum of C            */
/*                    C |  a  |  b  | TC    TE: sum of E            */
/*               sys2   |-----+-----|                               */
/*                    E |  c  |  d  | TE    element definitions:    */
/*                      -------------         a: num corr by both   */
/*                        TC    TE               systems            */
/*                                            c: num corr by sys1   */
/*                                               but not by sys2    */
/*                                            b: num corr by sys2   */
/*                                               but not by sys1    */
/*                                            d: num errored on by  */
/*                                               both systems       */
/*           Test Statistic formula:                                */
/*                                                                  */
/*                                                                  */
/*                   TS = Binomial(MIN(c,b),c+b,0.5)                */
/*                                                                  */
/*                                                                  */
/********************************************************************/

#include "sctk.h"

static int compute_McNemar(int **table, char *treat1_str, char *treat2_str, int verbosely, FILE *fp, double *conf, double alpha);
static void print_compare_matrix_for_sent_M(SCORES *scor[], int nscor, int **winner, double **conf, char *tname, char *matrix_name, FILE *fp);

/********************************************************************/
/*   this procedure does all the system comparisons then it         */
/*   prints a report to stdout                                      */
/********************************************************************/
void McNemar_sent(SCORES *scor[], int nscor, int ***out_winner, char *testname, int print_results, int verbose, char *outroot, int feedback, double ***out_conf)
{
    int comp1, comp2, **winner, result;
    double **conf;
    FILE *fp = stdout;
    
    if (print_results || verbose){
	char *f = rsprintf("%s.mcn",outroot);
	if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(f,"w")) ==
	    (FILE *)0){
	    fprintf(stderr,"Warning: Open of %s for write failed.  "
		           "Using stdout instead.\n",f);
	    fp = stdout;
	}
	if (feedback >= 1) printf("        Output written to '%s'\n",f);
    }

    alloc_2dimZ(winner,nscor,nscor,int,NO_DIFF);
    *out_winner = winner;
    alloc_2dimZ(conf,nscor,nscor,double,0.0);
    *out_conf = conf;

    for (comp1=0; comp1 <(nscor-1); comp1++)
        for (comp2=comp1+1; comp2<nscor; comp2++){
            result = do_McNemar_by_sent(scor[comp1],scor[comp2],verbose,fp,
					&(conf[comp1][comp2]));
            winner[comp1][comp2] = result;
        }
    if (print_results){
	if (verbose) form_feed(fp);
	print_compare_matrix_for_sent_M(scor, nscor, winner, conf, testname,
            "COMPARISON MATRIX: McNEMAR\'S TEST ON CORRECT SENTENCES FOR THE TEST:", fp);
	if (fp == stdout) form_feed(fp);
    }
    if (fp != stdout) fclose(fp);
}

/********************************************************************/
/*   using the COUNT structure, calculate the matrix of the McNemar */
/*   test then perform the test                                     */
/********************************************************************/

int do_McNemar_by_sent(SCORES *sys1, SCORES *sys2, int verbose, FILE *fp, double *conf)
{
    int ans, spk1, spk2, snt1, snt2, e1, e2;
    PATH *cp;
    int **table=NULL, nw;
    int foundMatchSent;
    alloc_2dimZ(table,2,2,int,0);

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
	        foundMatchSent = 0;
                for (snt2 = 0; snt2 < sys2->grp[spk2].num_path; snt2++){ 
                   /**** if the sentences are the same, compare them */
                   if(strcmp(sys1->grp[spk1].path[snt1]->id,
                             sys2->grp[spk2].path[snt2]->id) == 0){
		     e1 = e2 = 0;
		     cp = sys1->grp[spk1].path[snt1];
		     for (nw=0; nw < cp->num; nw++)
		       if (cp->pset[nw].eval != P_CORR) {
			 e1 = 1;
			 break;
		       }
		     cp = sys2->grp[spk2].path[snt2];
		     for (nw=0; nw < cp->num; nw++)
		       if (cp->pset[nw].eval != P_CORR) {
			 e2 = 1;
			 break;
		       }
		     table[e1][e2] ++;
		     foundMatchSent = 1;
		   }
		}
		if (! foundMatchSent)
		     fprintf(stderr,"Warning: Speaker's '%s' path '%s' in system '%s' is not in system '%s'\n",
			     sys1->grp[spk1].name,sys1->grp[spk1].path[snt1]->id,sys1->title,sys2->title   );
	    }
	} else {
            fprintf(stderr,"Warning: Speaker %s is in system %s but not system %s\n",
		    sys1->grp[spk1].name,sys1->title,sys2->title);
        }
    }

    ans = do_McNemar(table,sys1->title,sys2->title,verbose,fp,conf);
    free_2dimarr(table,2,int);
    return(ans);
}

/********************************************************************/
/*   given the McNemar matrix, come up with an answer.  verbose if  */
/*   desired                                                        */
/********************************************************************/
int do_McNemar(int **table, char *name1, char *name2, int verbose, FILE *fp, double *conf)
{
    
  if (verbose){
   fprintf(fp,"\n\n");
   fprintf(fp,"                            McNemar test results\n");
   fprintf(fp,"                            ====================\n\n");
   fprintf(fp,"                                     %s\n\n",name2);
   fprintf(fp,"                                  corr     incorr\n");
   fprintf(fp,"      %15s      corr   %3d        %3d\n",name1,
                                      table[0][0],table[0][1]);
   fprintf(fp,"\t                 incorr   %3d        %3d\n",
                                      table[1][0],table[1][1]);
 }
 if (((table[0][1] == 0) && (table[1][0] == 0)) || 
     (table[0][1] == table[1][0])){
   /* if both of these feilds are 0, or equal, then there is no */
   /* difference so say so                                      */
   if (verbose){
     fprintf(fp,"\n\n\t\tSUMMARY:\n\t\t-------\n\n");
     fprintf(fp,"\n\n\tThe two totals for utterances missed by either test results\n");
     fprintf(fp,"\tare both zero, therfore there is no significant difference\n");
     fprintf(fp,"\tbetween the two tests!\n");
   }
   *conf = 1.00;
     return(NO_DIFF);
 }
 else
   return(compute_McNemar(table,name1,name2,verbose,fp,conf,0.05));
}

#ifdef UNUSED
/********************************************************************/
/*   this program calculates the test statistic corresponding to a  */
/*   CHI SQUARED distribution. the formula is at the top of this    */
/*   file.                                                          */
/*   references to the 'Peregoy method' are adapted from            */
/*   Appendix B "The Significance of Bench Mark Test Results" by    */
/*   Peter Peregoy in American National Standard IAI 1-1987.        */
/*   Other distributions might be assumed.                          */
/********************************************************************/
static int perform_peregoy_method(int **table, int verbosely,FILE *fp, double *conf)
{
    int i;
    double ts_per; 
    extern double sqrt(double);

    /*  Perogoy method */
    ts_per = ((fabs((double)(table[0][1]-table[1][0]))-1)*
              (fabs((double)(table[0][1]-table[1][0]))-1)) / 
             (table[0][1]+table[1][0]);

    /* compute the confidence measure */
    *conf = X2.per[MIN_X2_PER] / 100.0;    
    for (i=MIN_X2_PER;i<MAX_X2_PER;i++)
	if (X2.df[DF1].level[i]   < fabs(ts_per) && 
	    X2.df[DF1].level[i+1] > fabs(ts_per)){
	    *conf = X2.per[i] / 100.0;    
	    break;
	}

    
    if (X2.df[DF1].level[MAX_X2_PER] < fabs(ts_per))
	*conf = X2.per[MAX_X2_PER] / 100.0;    

    if (verbosely){
        fprintf(fp,"\n\n"); 
        fprintf(fp,"%30s     Reject if\n","");
        fprintf(fp,"%27sX > X2 of %s %s (%2.3f)\n", "",
                          X2.per_str[GEN_X2_PER],
                          X2.df[DF1].str,
                          X2.df[DF1].level[GEN_X2_PER]);
        fprintf(fp,"\n"); 
        fprintf(fp,"%30s   X = %2.3f\n","",ts_per);
        fprintf(fp,"\n\n\t\tSUMMARY:\n\t\t-------\n\n");
    }
    if (fabs(ts_per) > X2.df[DF1].level[GEN_X2_PER]){
        if (verbosely){
            fprintf(fp,"\tPeregoy's method shows that, with %s confidence, the\n",
                                 X2.neg_per_str[GEN_X2_PER]),
            fprintf(fp,"\t2 recognition systems are significantly different.\n");
            fprintf(fp,"\n");
            fprintf(fp,"\tFurther, the probablity of there being a difference is\n");
            for (i=GEN_X2_PER;i<MAX_X2_PER;i++)
                if (fabs(ts_per) < X2.df[DF1].level[i+1])
                    break;
            if (i==MAX_X2_PER)
		fprintf(fp,"\tgreater that %s.\n",X2.neg_per_str[i]);
            else
                fprintf(fp,"\tbetween %s to %s.\n",X2.neg_per_str[i],
                                               X2.neg_per_str[i+1]);
            fprintf(fp,"\n\n");
	}
        if (table[1][0] < table[0][1])
            return(TEST_DIFF * (-1)); /* invert because sys1 was better */
        else
            return(TEST_DIFF);
    }
    else{
        if (verbosely){ 
            fprintf(fp,"\tPeregoy's method shows that at the %s confidence\n",
                                  X2.neg_per_str[GEN_X2_PER]),
            fprintf(fp,"\tinterval, the 2 recognition systems are not significantly\n");
            fprintf(fp,"\tdifferent.\n\n");
            fprintf(fp,"\tFurther, the probablity of there being a difference is\n");
            for (i=GEN_X2_PER;i>MIN_X2_PER;i--)
                if (fabs(ts_per) > X2.df[DF1].level[i-1])
                    break;
            if (i==MIN_X2_PER)
                fprintf(fp,"\tless than %s.\n",X2.neg_per_str[i]);
            else
                fprintf(fp,"\tbetween %s to %s.\n",X2.neg_per_str[i-1],
                                         X2.neg_per_str[i]);
            fprintf(fp,"\n\n");
	}
        return(NO_DIFF);
    }
}
#endif


/********************************************************************/
/*   this procedure is called to print out a comparison matrix for  */
/*   any comparison function.                                       */
/********************************************************************/
static void print_compare_matrix_for_sent_M(SCORES *scor[], int nscor, int **winner, double **conf, char *tname, char *matrix_name,FILE *fp)
{
    char fmt[50];
    int i,j,sys,spkr,snt, *corr_arr;

    alloc_singarr(corr_arr,nscor,int);

    /* calc the number of correct for each system */
    for (sys=0;sys<nscor;sys++){
        corr_arr[sys] = 0;
        for (spkr=0; spkr<scor[sys]->num_grp; spkr++)
	    for (snt=0; snt<scor[sys]->grp[spkr].num_path; snt++)
		corr_arr[sys] += scor[sys]->grp[spkr].num_path - 
		                 scor[sys]->grp[spkr].serr;
    }

    strcpy(fmt,"c");
    for (i=0; i<nscor; i++) strcat(fmt,"|c");

    Desc_erase();
    Desc_set_page_center(SCREEN_WIDTH);
    Desc_add_row_values("c",matrix_name);
    Desc_add_row_separation(' ',AFTER_ROW);
    Desc_add_row_separation('-',AFTER_ROW);
    Desc_add_row_values("c",tname);

    Desc_set_iterated_format(fmt);
    Desc_set_iterated_value("");
    for (i=0; i<nscor; i++)
	Desc_set_iterated_value(rsprintf("%s",scor[i]->title));
    Desc_flush_iterated_row();

    for (i=0; i<nscor; i++) {
	Desc_add_row_separation('-',BEFORE_ROW);
	Desc_set_iterated_format(fmt);
	Desc_set_iterated_value(rsprintf("%s",scor[i]->title));
	for (j=0; j<nscor; j++)
  	    if (j > i)
		Desc_set_iterated_value(rsprintf("Conf=(%.3f)",(conf[i][j] < 0.001) ? 0.001 : conf[i][j]));
            else
		Desc_set_iterated_value("");
	Desc_flush_iterated_row();

	Desc_set_iterated_format(fmt);
	Desc_set_iterated_value("");
	for (j=0; j<nscor; j++){
            char *name="";
            if (j > i){
                if (winner[i][j] == TEST_DIFF)
                    name=scor[j]->title;
                else if (winner[i][j] == NO_DIFF)
                    name="same";
                else
                    name=scor[i]->title;
            }
	    Desc_set_iterated_value(name);
	}
	Desc_flush_iterated_row();
    }
    Desc_dump_report(1,fp);
    free_singarr(corr_arr,int);
}


static int compute_McNemar(int **table, char *treat1_str, char *treat2_str, int verbose, FILE *fp, double *confidence, double alpha)
{ 
    double test_stat, p=0.5;
    int decision_cutoff=(-1), i, num_a, num_b;

    num_a = table[1][0];
    num_b = table[0][1];

    /* multiplication by 2 means it's a two-tailed test */
    test_stat = 2.0 * compute_acc_binomial(MIN(num_a,num_b),num_a+num_b,p);
    *confidence = test_stat;

    for (i=0; i< (num_a + num_b); i++)
        if (2.0*compute_acc_binomial(i,num_a+num_b,p) < alpha)
            decision_cutoff=i;

    if (verbose){
        fprintf(fp,"The NULL Hypothesis:\n\n");
        fprintf(fp,"     The number unique utterance errors are equal for both systems.\n");
	fprintf(fp,"\n");
        fprintf(fp,"Alternate Hypothesis:\n\n");
        fprintf(fp,"     The number of unique utterance errors for both systems are NOT equal.\n");

        fprintf(fp,"Decision Analysis:\n\n");
        fprintf(fp,"     Assumptions:\n");
        fprintf(fp,"        A1: The distibution of unique utterance errors\n");
        fprintf(fp,"            follows the binomial distribution for N fair coin tosses.\n");
        fprintf(fp,"\n");
        fprintf(fp,"     Rejection criterion:\n");
        fprintf(fp,"        Reject the null hypothesis at the 95%% confidence level based\n");
        fprintf(fp,"        on the following critical values table.  N is the sum of the\n");
        fprintf(fp,"        unique utterance errors for both systems being compared and\n");
        fprintf(fp,"        MIN(uue) is the minimum number of unique utterance\n");
        fprintf(fp,"        foe either system.\n\n");
        /* print a table of critical values */
        fprintf(fp,"          MIN(uue)      P(MIN(uue) | N=%3d)\n",num_a+num_b);

        fprintf(fp,"          --------      -------------------\n");
        for (i=0; 2.0*compute_acc_binomial(i-1,num_a+num_b,p) < (MAX(0.25,test_stat)); i++){
	    double val = 2.0*compute_acc_binomial(i,num_a+num_b,p);
	    if (val >= 0.0005)
	      fprintf(fp,"             %3d               %5.3f", i,val);
	    else
	      fprintf(fp,"             %3d                 -  ", i);
            if ((val < alpha) && (2.0*compute_acc_binomial(i+1,num_a+num_b,p) > alpha))
  	        fprintf(fp,"  <---  Null Hypothesis rejection threshold\n");
            else
                fprintf(fp,"\n");
        }
        fprintf(fp,"\n");
        fprintf(fp,"     Decision:\n");

        fprintf(fp,"        There were MIN(uue)=%d unique utterance errors, the probability of\n",
		MIN(num_a,num_b));
        fprintf(fp,"        it occuring is %5.3f, therefore the null hypothesis ",test_stat);        
        if (test_stat < alpha){            
            fprintf(fp,"is REJECTED\n");
            fprintf(fp,"        in favor of the Alternate Hypothesis.  Further, %s is the\n",
                   (num_a > num_b) ? treat2_str : treat1_str);
            fprintf(fp,"        better System.\n");
        } else{
            fprintf(fp,"is ACCEPTED\n");
            fprintf(fp,"        There is no statistical difference between %s and %s\n",treat1_str,treat2_str);
        }
        form_feed(fp);
    }
    if (test_stat < alpha){
        if (0) fprintf(fp,"Returning Result %d\n",TEST_DIFF * ((num_a > num_b) ? 1 : -1));
        return(TEST_DIFF * ((num_a > num_b) ? 1 : -1));
      }
    return(NO_DIFF);
}
