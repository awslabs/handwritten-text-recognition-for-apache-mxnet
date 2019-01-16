/**********************************************************************/
/*                                                                    */
/*           FILE: wilcoxon.c                                         */
/*           WRITTEN BY: Jonathan G. Fiscus                           */
/*           DATE: April 14 1989                                      */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*           USAGE: This uses the rank structure to perform           */
/*                  the non-parametric Wilcoxon Test and generates    */
/*                  a report.                                         */
/*                                                                    */
/*           SOURCE:Statistics: Probability, Inference and Decision   */
/*                  By Robert L. Winkler and William L. Hays,         */
/*                  Page 856                                          */
/*                                                                    */
/*           See Also: The documentation file "wilcoxon.doc"          */
/*                                                                    */
/**********************************************************************/

#include "sctk.h"

static int wilcoxon_test_analysis(int num_a, int num_b, double sum_a, double sum_b, char *treat1_str, char *treat2_str, double alpha, int verbose, int zero_is_best, double *conf, FILE *fp);

/****************************************************************/
/*   main procedure to perform the wilcoxon test on the RANK    */
/*   structure.                                                 */
/****************************************************************/
void perform_wilcoxon(RANK *rank, int verbose, int report, char *formula_str, char formula_id, int ***out_winner, char *outroot, int feedback, double ***out_conf)
{
    int comp1, comp2, result, **winner;
    double **conf;
    FILE *fp = stdout;
    
    if (report || verbose){
	char *f = rsprintf("%s.wilc",outroot);
	if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(f,"w")) ==
	    (FILE *)0){
	    fprintf(stderr,"Warning: Open of %s for write failed.  "
		           "Using stdout instead.\n",f);
	    fp = stdout;
	} else
	    if (feedback >= 1) printf("        Output written to '%s'\n",f);
    }

    alloc_2dimZ(winner,rank->n_trt,rank->n_trt,int,NO_DIFF);
    alloc_2dimZ(conf,rank->n_trt,rank->n_trt,double,0.0);
    *out_winner = winner;
    *out_conf = conf;
  
    if (!report) verbose=FALSE;
    for (comp1=0; comp1 < rank->n_trt  -1; comp1++)
        for (comp2=comp1+1; comp2< rank->n_trt ; comp2++){
            result = compute_wilcoxon_for_treatment(rank,comp1,comp2,"Spkr",
                                  formula_str,verbose,formula_id=='E',fp,
				  &(conf[comp1][comp2]));
            winner[comp1][comp2] = result;
        }

    if (report) print_trt_comp_matrix_for_RANK_one_winner(winner,rank,
                 "Comparison Matrix for the Wilcoxon Test",formula_str,
                 "Speaker",fp);
    if (fp != stdout) fclose(fp);
}

/****************************************************************/
/*   Given the two indexes of treatments to compare (in the     */
/*   RANK Struct) compute the Wilcoxon statistics               */
/*   vars:                                                      */
/*       zero_is_best : This identifies the "ideal" value for   */
/*                      the value computed in the rank struct   */
/*                      percentages.                            */
/****************************************************************/
int compute_wilcoxon_for_treatment(RANK *rank, int treat1, int treat2, char *block_id, char *formula_str, int verbose, int zero_is_best, FILE *fp, double *conf)
{
    int sum_plus=0, sum_minus=0, i;
    double sum_plus_rank = 0.0, sum_minus_rank = 0.0;
    double sum_trt1=0.0, sum_trt2=0.0, sum_trt_diff=0.0;
    int block, max_len_block=0, max_len_treat=6;
    char *pct_format, thresh_str[140];
    TEXT *title_line = (TEXT *)0;
    int paper_width = 79, rep_width;
    double equal_thresh = 0.05;
    double *block_diff, *block_rank;
    int *block_sort, tptr[2];
    int alternate;

    alloc_singarr(block_diff, rank->n_blk ,double);
    alloc_singarr(block_rank, rank->n_blk ,double);
    alloc_singarr(block_sort, rank->n_blk ,int);

    /* change the ordering if necessary */
    for (block=0; block <rank->n_blk ; block++){
        sum_trt1 += rank->pcts [ block ][ treat1 ] ;
        sum_trt2 += rank->pcts [ block ][ treat2 ] ;
    }
    if (sum_trt1 > sum_trt2)
        tptr[0] = treat1, tptr[1] = treat2;
    else
        tptr[0] = treat2, tptr[1] = treat1;
    sum_trt1 = sum_trt2 = 0.0;

    /* compute the test */
    for (block=0; block <rank->n_blk ; block++)
        block_diff[block] = fabs(rank->pcts [ block ][ tptr[0] ]  -
                               rank->pcts [ block ][ tptr[1] ] );
    rank_double_arr(block_diff, rank->n_blk ,block_sort,block_rank,INCREASING);
    for (block=0; block <rank->n_blk ; block++)
        block_diff[block] = rank->pcts [ block ][ tptr[0] ]  -
                               rank->pcts [ block ][ tptr[1] ] ;


    sprintf(thresh_str,"(Threshold for equal percentages +- %5.3f)",
                       equal_thresh);

    max_len_block = strlen(block_id);
    for (block=0; block <rank->n_blk ; block++)
         if (strlen(rank->blk_name [ block ] ) > max_len_block)

             max_len_block = strlen(rank->blk_name [ block ] );

    if (max_len_treat < strlen(rank->trt_name [ tptr[0] ] ))
        max_len_treat = strlen(rank->trt_name [ tptr[0] ] );
    if (max_len_treat < strlen(rank->trt_name [ tptr[1] ] ))
        max_len_treat = strlen(rank->trt_name [ tptr[1] ] );
    rep_width = max_len_block + max_len_treat * 2 + 37;

    alloc_singarr(pct_format, max_len_treat + 2,char); 
    pct_format[0] = '\0';
    strcat(pct_format,center("",(max_len_treat-6)/2));
    strcat(pct_format,"%6.2f");
    strcat(pct_format,center("",max_len_treat - ((max_len_treat-6)/2 + 6) ));

    /*  Print a detailed table showing sign differences */
    if (verbose) {
        fprintf(fp,"%s\n",center("Wilcoxon Test Calculations Table",
				 paper_width));
        title_line = TEXT_strdup(rsprintf("Comparing %s %s Percentages for Systems %s and %s",
		block_id, formula_str,rank->trt_name [ tptr[0] ] ,
		rank->trt_name [ tptr[1] ] ));
        fprintf(fp,"%s\n\n",center(title_line,paper_width));

        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        fprintf(fp,"%s",center("",max_len_block));
        fprintf(fp,"    "); fprintf(fp,"%s",center("",max_len_treat));
        fprintf(fp,"    "); fprintf(fp,"%s",center("",max_len_treat));
        fprintf(fp,"  ");
        fprintf(fp,"                    Signed\n");

        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        fprintf(fp,"%s",center(block_id,max_len_block));
        fprintf(fp,"    ");
        fprintf(fp,"%s",center(rank->trt_name [ tptr[0] ] ,max_len_treat));
        fprintf(fp,"    ");
        fprintf(fp,"%s",center(rank->trt_name [ tptr[1] ] ,max_len_treat));
        fprintf(fp,"  ");
        fprintf(fp,"Difference   Rank    Rank\n");
        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        for (i=0; i<rep_width; i++)
            fprintf(fp,"-");
        fprintf(fp,"\n");
    }
    alternate=0;
    for (block=0; block <rank->n_blk ; block++){
         if (verbose) {
             fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
             fprintf(fp,"%s",center(rank->blk_name [ block ] ,max_len_block));
             fprintf(fp,"    ");
             fprintf(fp,pct_format,rank->pcts [ block ][ tptr[0] ] );
             fprintf(fp,"    ");
             fprintf(fp,pct_format,rank->pcts [ block ][ tptr[1] ] );
             fprintf(fp,"   ");
	     fprintf(fp,"%6.2f    %5.1f    ",block_diff[block],block_rank[block]);
             sum_trt1 += rank->pcts [ block ][ tptr[0] ] ; 
             sum_trt2 += rank->pcts [ block ][ tptr[1] ] ;
             sum_trt_diff += block_diff[block];
         }
         if (block_diff[block] < 0.0 && (fabs(block_diff[block]) > 0.005)){
             if (verbose) fprintf(fp,"%5.1f\n",(-1.0) * block_rank[block]);
             sum_minus++;
             sum_minus_rank += block_rank[block];
         }
         else if (block_diff[block] > 0.0 &&(fabs(block_diff[block]) > 0.005)){
             if (verbose) fprintf(fp,"%5.1f\n",block_rank[block]);
             sum_plus++;
             sum_plus_rank += block_rank[block];
         } else {
             if (verbose) fprintf(fp,"%5.1f\n",block_rank[block]);
	     if ((alternate++ % 2) == 0){
		 sum_plus++;
		 sum_plus_rank += block_rank[block];
	     } else {
		 sum_minus++;
		 sum_minus_rank += block_rank[block];
	     }		 
	 }
    }
    if (verbose){
        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        for (i=0; i<rep_width; i++)
            fprintf(fp,"-");
        fprintf(fp,"\n");
        /* An average line */
        fprintf(fp,"%s",center("",(paper_width - rep_width)/2));
        fprintf(fp,"%s",center("Avg.",max_len_block));
        fprintf(fp,"    ");
        fprintf(fp,pct_format,sum_trt1 /rank->n_blk );
        fprintf(fp,"    ");
        fprintf(fp,pct_format,sum_trt2 /rank->n_blk );
        fprintf(fp,"   ");
	fprintf(fp,"%6.2f",sum_trt_diff /rank->n_blk );
        fprintf(fp,"\n\n");

        sprintf(title_line,"Sum of %2d positive ranks = %4.1f",
		sum_plus,sum_plus_rank);
        fprintf(fp,"%s\n",center(title_line,paper_width));
        sprintf(title_line,"Sum of %2d negative ranks = %4.1f",
		sum_minus,sum_minus_rank);
        fprintf(fp,"%s\n",center(title_line,paper_width));
        fprintf(fp,"\n\n");
    }
    
    free_singarr(block_diff,double);
    free_singarr(block_rank,double);
    free_singarr(block_sort,int);
    free_singarr(pct_format,char);
    if (title_line != (TEXT *)0)
        free_singarr(title_line, TEXT);

    /* Analyze the Results */
    { int result;
      result = wilcoxon_test_analysis(sum_plus,sum_minus,sum_plus_rank,
				      sum_minus_rank,rank->trt_name[tptr[0]],
				      rank->trt_name[tptr[1]],0.05,verbose,
				      zero_is_best,conf,fp);
      /* if the result is significant, system which is better depends on if */
      /* the treatments have been swapped, a negative result means tprt[0] is*/
      /* better, positive one tprt[1] is better */
      return(result * ((tptr[0] == treat1) ? 1 : (-1)));
    }
}

/****************************************************************/
/*   Given the vital numbers for computing a rank sum test,     */
/*   Compute it, and if requested, print a verbose analysis     */
/****************************************************************/
static int wilcoxon_test_analysis(int num_a, int num_b, double sum_a, double sum_b, char *treat1_str, char *treat2_str, double alpha, int verbose, int zero_is_best, double *conf, FILE *fp)
{
    double test_stat;
    int i, z_ind=0, dbg=0;
    char better;

    for (i=0; i <= PER90; i++)
        if (Z2tail[i].perc_interior-(1.0 - alpha) > -0.001){
            z_ind=i;
            if (dbg) printf("Z score %d %f %f\n",i,Z2tail[i].perc_interior,
                            Z2tail[i].perc_interior-(1.0 - alpha));
	    }
    /* compute the Z statistic */
    { double W, E;
      double var;
      int n;
      if (num_a < num_b)
          W = sum_a; 
      else
          W = sum_b; 
      n = num_a + num_b;
      better = (num_a < num_b) ? 'b' : 'a';
      if (zero_is_best)  /* make an adjustment for direction of PCTS*/
          better = (better == 'a') ? 'b' : 'a';
      E = ((double)n * (double)(n+1) / 4.0);
      var = (double)(n * (n+1) * (2*n + 1)) / 24.0;
      test_stat = ((W - E) / (double)sqrt(var));
      /* W will be rounded to the nearest whole number */
      if (dbg){
	  printf("   Z Approximation for Wilcoxon Signed-Rank test\n");
	  printf("        W = %f\n",W);
	  printf("        n = %d\n",n);
	  printf("     E(W) = %f\n",E);
	  printf("      var = %f\n",var);
	  printf("   Z_stat = %f\n",test_stat);
	  printf("   Better = %c\n",better);
	  printf(" Better S = %s\n",
		 (better == 'a') ? treat1_str : treat2_str);
      }
    }
    *conf = 1.0 - (2.0 * (normprob((double) (fabs(test_stat))) - 0.50));

    if (verbose){
        fprintf(fp,"    The Null Hypothesis:\n\n");
        fprintf(fp,"         The two populations represented by the respective matched\n");
        fprintf(fp,"         pairs are identical.\n\n");
        fprintf(fp,"    Alternate Hypothesis:\n\n");
        fprintf(fp,"         The two populations are not equal and therfore\n");
        fprintf(fp,"         there is a difference between systems.\n\n");
        fprintf(fp,"    Decision Analysis:\n\n");
        fprintf(fp,"         Assumptions:\n");
        fprintf(fp,"              The distribution of signed ranks is approximated\n");
        fprintf(fp,"              by the normal distribution if the number of Blocks\n");
        fprintf(fp,"              is greater than equal to 8.\n\n");
        fprintf(fp,"         Rejection Threshold:\n");
        fprintf(fp,"              Based on a %2.0f%% confidence interval, the Z-score should\n",
                              Z2tail[z_ind].perc_interior*100.0);
        fprintf(fp,"              be greater than %4.2f or less than -%4.2f.\n\n",Z2tail[z_ind].z,
                          Z2tail[z_ind].z);
        fprintf(fp,"         Decision:\n");
        fprintf(fp,"              The Z statistic = %4.2f, therefore the Hypothesis\n",test_stat);
        if (fabs(test_stat) > Z2tail[z_ind].z){
            fprintf(fp,"              is REJECTED in favor of the Null Hypothesis.\n\n");
            fprintf(fp,"              Further, %s is the better System.\n",
                   (better == 'a') ? treat1_str : treat2_str);
        } else
            fprintf(fp,"              is ACCEPTED.\n");
        form_feed(fp);
    }
    if (fabs(test_stat) > Z2tail[z_ind].z)
        return(TEST_DIFF * ((better == 'a') ? (-1) : 1));
    return(NO_DIFF);
}
        
        
    
