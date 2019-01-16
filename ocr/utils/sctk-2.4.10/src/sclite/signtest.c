/**********************************************************************/
/*                                                                    */
/*           FILE: signtest.c                                         */
/*           WRITTEN BY: Jonathan G. Fiscus                           */
/*           DATE: April 14 1989                                      */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*           USAGE: This uses the rank structure to perform           */
/*                  the non-parametric Sign Test and generates        */
/*                  a report.                                         */
/*                                                                    */
/*           SOURCE:Statistics: Basic Techniques for Solving          */
/*                  Applied Problems, by Stephen A. Book, 1977        */
/*                                                                    */
/*           See Also: The documentation file "signtest.doc"          */
/*                                                                    */
/**********************************************************************/

#include "sctk.h"

/****************************************************************/
/*   main procedure to perform the sign test on the RANK        */
/*   structure.                                                 */
/****************************************************************/
void perform_signtest(RANK *rank, int verbose, int report, char *formula_str, char formula_id, int ***out_winner, char *outroot, int feedback, double ***out_confidence)
{
    int comp1, comp2, **winner;
    double **confidence;
    FILE *fp = stdout;
    
    if (report || verbose){
	char *f = rsprintf("%s.sign",outroot);
	if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(f,"w")) ==
	    (FILE *)0){
	    fprintf(stderr,"Warning: Open of %s for write failed.  "
		           "Using stdout instead.\n",f);
	    fp = stdout;
	} else
	    if (feedback >= 1) printf("        Output written to '%s'\n",f);
    }

    alloc_2dimZ(winner,rank->n_trt,rank->n_trt,int,NO_DIFF);
    alloc_2dimZ(confidence,rank->n_trt,rank->n_trt,double,0.0);
    *out_confidence = confidence;
    *out_winner = winner;

    for (comp1=0; comp1 < rank->n_trt  -1; comp1++)
        for (comp2=comp1+1; comp2< rank->n_trt ; comp2++){
            winner[comp1][comp2] = 
               compute_signtest_for_treatment(rank,comp1,comp2,"Spkr",
					      formula_str,verbose,
					      formula_id=='E',fp,
					      &(confidence[comp1][comp2]));
        }

    if (report) print_trt_comp_matrix_for_RANK_one_winner(winner,rank,
                 "Comparison Matrix for the Sign Test",formula_str,
                 "Speaker",fp);
    if (fp != stdout) fclose(fp);
}

/****************************************************************/
/*   Given the two indexes of treatments to compare (in the     */
/*   RANK Struct) compute the Rank Sum statistics               */
/*   vars:                                                      */
/*       zero_is_best : This identifies the "ideal" value for   */
/*                      the value computed in the rank struct   */
/*                      percentages.                            */
/****************************************************************/
int compute_signtest_for_treatment(RANK *rank, int treat1, int treat2, char *block_id, char *formula_str, int verbose, int zero_is_best, FILE *fp, double *confidence)
{
    int sum_plus=0, sum_minus=0, sum_equal=0, i;
    int block, max_len_block=0, max_len_treat=6;
    int tptr[2]; /* a sorting pointer array for the treatment numbers */
    char *pct_format, thresh_str[140];
    TEXT *title_line = (TEXT *)0;
    char *diff_line;
    int paper_width = 79, rep_width, diff_col_len;
    double equal_thresh = 0.005;
    double sum_trt1=0.0, sum_trt2=0.0;

    /* compute the maximum block title length */
    max_len_block = strlen(block_id);
    for (block=0; block <rank->n_blk ; block++)
         if (strlen(rank->blk_name [ block ] ) > max_len_block)
             max_len_block = strlen(rank->blk_name [ block ] );

    if (max_len_treat < strlen(rank->trt_name [ treat1 ] ))
        max_len_treat = strlen(rank->trt_name [ treat1 ] );

    if (max_len_treat < strlen(rank->trt_name [ treat2 ] ))
        max_len_treat = strlen(rank->trt_name [ treat2 ] );

    /* set the treatment sorting array */
    for (block=0; block <rank->n_blk ; block++){
        sum_trt1 += rank->pcts [ block ][ treat1 ] ;
        sum_trt2 += rank->pcts [ block ][ treat2 ] ;
    }
    if (sum_trt1 > sum_trt2)
        tptr[0] = treat1, tptr[1] = treat2;
    else
        tptr[0] = treat2, tptr[1] = treat1;

    /* set up format strings and titles */
    sprintf(thresh_str,"(Threshold for equal percentages +- %5.3f)",
                       equal_thresh);
    alloc_singarr(pct_format, max_len_treat + 2,char); 
    alloc_singarr(diff_line, max_len_treat*2 + 7,char); 
    pct_format[0] = '\0';
    strcat(pct_format,center("",(max_len_treat-6)/2));
    strcat(pct_format,"%6.2f");
    strcat(pct_format,center("",max_len_treat - ((max_len_treat-6)/2 + 6) ));
    sprintf(diff_line,"[%s - %s]",rank->trt_name[tptr[0]],
	  rank->trt_name[tptr[1]]);
    diff_col_len = 15;
    if (strlen(diff_line) > diff_col_len) diff_col_len=strlen(diff_line);

    rep_width = max_len_block + max_len_treat * 2 + 4 * 3 + diff_col_len;

    /*  Print a detailed table showing sign differences */
    if (verbose) {
        sum_trt1 = sum_trt2 = 0.0;
        fprintf(fp,"%s\n",center("Sign Test Calculations Table Comparing",
				 paper_width));
        title_line = TEXT_strdup(rsprintf("%s %s Percentages for Systems %s and %s",
		block_id, formula_str,rank->trt_name[tptr[0]],
		rank->trt_name [ tptr[1] ] ));
        fprintf(fp,"%s\n",center(title_line,paper_width));
        fprintf(fp,"%s\n\n",center(thresh_str,paper_width));

        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        fprintf(fp,"%s",center("",max_len_block));
        fprintf(fp,"    "); fprintf(fp,"%s",center("",max_len_treat));
        fprintf(fp,"    "); fprintf(fp,"%s",center("",max_len_treat));
        fprintf(fp,"    ");
        fprintf(fp,"%s\n",center("Difference Sign",diff_col_len));

        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        fprintf(fp,"%s",center(block_id,max_len_block));
        fprintf(fp,"    ");
        fprintf(fp,"%s",center(rank->trt_name [ tptr[0] ] ,max_len_treat));
        fprintf(fp,"    ");
        fprintf(fp,"%s",center(rank->trt_name [ tptr[1] ] ,max_len_treat));
        fprintf(fp,"    ");
        fprintf(fp,"%s\n",center(diff_line,diff_col_len));
        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        for (i=0; i<rep_width; i++)
            fprintf(fp,"-");
        fprintf(fp,"\n");
    }
    for (block=0; block <rank->n_blk ; block++){
         if (verbose) {
             fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
             fprintf(fp,"%s",center(rank->blk_name [ block ] ,max_len_block));
             fprintf(fp,"    ");
             fprintf(fp,pct_format,rank->pcts [ block ][ tptr[0] ] );
             fprintf(fp,"    ");
             fprintf(fp,pct_format,rank->pcts [ block ][ tptr[1] ] );
             fprintf(fp,"    ");
         }
         if (fabs(rank->pcts [ block ][ tptr[0] ] - rank->pcts [ block ][ tptr[1] ] ) <=
                  equal_thresh){
             if (verbose) fprintf(fp,"%s\n",center("0",diff_col_len));
             sum_equal++;
         }
         else if (rank->pcts [ block ][ tptr[0] ]  < rank->pcts [ block ][ tptr[1] ] ){
             if (verbose) fprintf(fp,"%s\n",center("-",diff_col_len));
             sum_minus++;
         }
         else {
             if (verbose) fprintf(fp,"%s\n",center("+",diff_col_len));
             sum_plus++;
         }
         sum_trt1 += rank->pcts [ block ][ tptr[0] ] ; 
         sum_trt2 += rank->pcts [ block ][ tptr[1] ] ;
    }
    if (verbose){
        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        for (i=0; i<rep_width; i++)
            fprintf(fp,"-");
        fprintf(fp,"\n");
        /* an Average line */
        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        fprintf(fp,"%s",center("Avg.",max_len_block));
        fprintf(fp,"    ");
        fprintf(fp,pct_format,sum_trt1 /rank->n_blk );
        fprintf(fp,"    ");
        fprintf(fp,pct_format,sum_trt2 /rank->n_blk );
        fprintf(fp,"\n\n");

        sprintf(title_line,"No. Speakers with Positive %s Differences = N(+) = %2d",formula_str,sum_plus);
        fprintf(fp,"%s\n",center(title_line,paper_width));
        sprintf(title_line,"No. Speakers with Negative %s Differences = N(-) = %2d",formula_str,sum_minus);
        fprintf(fp,"%s\n",center(title_line,paper_width));
        sprintf(title_line,"No. Speakers with No %s Differences = N(0) = %2d",formula_str,sum_equal);
        fprintf(fp,"%s\n",center(title_line,paper_width));
        fprintf(fp,"\n\n");
    }

    free_singarr(pct_format,char);
    free_singarr(diff_line,char);
    if (title_line != (TEXT *)0)
        free_singarr(title_line, TEXT);
    /* Analyze the Results */
    { int result;
      result = sign_test_analysis(sum_plus,sum_minus,sum_equal,"+","-",0,
				  0.05,verbose, rank->trt_name [ tptr[0] ] ,
				  rank->trt_name [ tptr[1] ] ,
				  tptr,zero_is_best,fp,confidence);
      /* if the result is significant, system which is better depends on if */
      /* the treatments have been swapped, a negative result means tprt[0] is */
      /* better, positive one tprt[1] is better */
      return(result * ((tptr[0] == treat1) ? 1 : (-1)));
    }
}

/****************************************************************/
/*   Given the vital numbers for computing a rank sum test,     */
/*   Compute it, and if requested, print a verbose analysis     */
/****************************************************************/
int sign_test_analysis(int num_a, int num_b, int num_z, char *str_a, char *str_b, int str_z, double alpha, int verbose, char *treat1_str, char *treat2_str, int *tptr, int zero_is_best, FILE *fp, double *confidence)
{
    double test_stat, p=0.5;
    int i;

    num_b += (num_z / 2) + (num_z % 2);
    num_a += (num_z / 2);
    num_z = 0;

    /* multiplication by 2 means it's a two-tailed test */
    if (num_a != num_b) 
        test_stat = 2.0 * compute_acc_binomial(MIN(num_a,num_b),num_a+num_b,p);
    else
        test_stat = 1.0;
    *confidence = test_stat;

    if (verbose){
        fprintf(fp,"The NULL Hypothesis:\n\n");
        fprintf(fp,"     The number of speakers for which the differences is positive\n");
        fprintf(fp,"     equals the number of speakers for which the differences is\n");
        fprintf(fp,"     negative.\n");
        fprintf(fp,"			 P(N(+)) = P(N(-)) = 0.50\n\n");

        fprintf(fp,"Alternate Hypothesis:\n\n");
        fprintf(fp,"     The number of speakers for which the differences is positive \n");
        fprintf(fp,"     is NOT equal to the number of speakers for which the difference\n");
        fprintf(fp,"     is negative.\n\n");
        fprintf(fp,"Decision Analysis:\n\n");
        fprintf(fp,"     Assumptions:\n");
        fprintf(fp,"        A1: The distibution of positive and negative differences\n");
        fprintf(fp,"            follows the binomial distribution for N fair coin tosses.\n");
        fprintf(fp,"\n");
        fprintf(fp,"        A2: In order to resolve the complication caused by cases where the\n");
        fprintf(fp,"            difference in Word Accuracy is zero, half of the cases will\n");
        fprintf(fp,"            be assigned to N(+) and half to N(-).  In the event of an\n");
        fprintf(fp,"            odd number of zero differences, N(-) will get one extra, this\n");
        fprintf(fp,"            reduces the probability of there being a difference between\n");
        fprintf(fp,"            the two systems.\n\n");
        fprintf(fp,"     Rejection criterion:\n");
        fprintf(fp,"        Reject the null hypothesis at the 95%% confidence level based\n");
        fprintf(fp,"        on the following critical values table.  N is the number of\n");
        fprintf(fp,"        speakers being compared and N(-) is the number of negative\n");
        fprintf(fp,"        differences.\n\n");
        /* print a table of critical values */
        fprintf(fp,"          MIN(N(-),N(+))      P(MIN(N(-),N(+)) | N=%2d)\n",num_a+num_b);

        fprintf(fp,"          --------------      ------------------------\n");
        for (i=0; i <= (num_a+num_b)/2 && (i - 3) <= num_b; i++){
	    double val = 2.0*compute_acc_binomial(i,num_a+num_b,p),
	           valp1 = 2.0*compute_acc_binomial(i+1,num_a+num_b,p);
	    if (val >= 0.0005)
	      fprintf(fp,"          %3d                 %5.3f", i,val);
	    else 
	      fprintf(fp,"          %3d                <0.001  ", i);
                   
            if ((val < alpha) && (valp1 > alpha))
                fprintf(fp,"  <---  Null Hypothesis rejected at or below this point\n");
            else
                fprintf(fp,"\n");
        }
        fprintf(fp,"\n");
        fprintf(fp,"     Decision:\n");

        fprintf(fp,"        There were N(-)=%d negative differences , the probability of\n",num_b);
        fprintf(fp,"        it occuring is %5.3f, therefore the null hypothesis ",test_stat);        
        if (test_stat < alpha){            
            fprintf(fp,"is REJECTED\n");
            fprintf(fp,"        in favor of the Alternate Hypothesis.  Further, %s is the\n",
                   (zero_is_best) ? treat2_str : treat1_str);
            fprintf(fp,"        better System.\n");
        } else{
            fprintf(fp,"is ACCEPTED\n");
            fprintf(fp,"        There is no statistical difference between %s and %s\n",treat1_str,treat2_str);
        }
        form_feed(fp);
    }
    if (test_stat < alpha){
        if (0) fprintf(fp,"Returning Result %d\n",TEST_DIFF * ((zero_is_best) ? 1 : -1));
        return(TEST_DIFF * ((zero_is_best) ? 1 : -1));
      }
    return(NO_DIFF);
}

