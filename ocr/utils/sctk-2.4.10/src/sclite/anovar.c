/**********************************************************************/
/*                                                                    */
/*           FILE: anovar.c                                           */
/*           WRITTEN BY: Jonathan G. Fiscus                           */
/*           DATE: April 14 1989                                      */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*           USAGE: This uses the rank structure to perform           */
/*                  a Friedman two-way analysis of variance           */
/*                  by ranks and generates a report.                  */
/*                                                                    */
/*           SOURCE:Applied Non Parametric Statistics by Daniel       */
/*                                                                    */
/**********************************************************************/

#include "sctk.h"

#define FRI_TITLE	"Friedman Two-way Analysis of Variance by Ranks"
#define RANK_TITLEA	"COMPARISON MATRIX: Comparing All Systems"
#define RANK_TITLEB	"COMPARISON MATRIX: Comparing All Speakers"
#define RANK_TITLE_1	"Using a Multiple Comparison Test"
#define ADJUST_THRESH	0.50000
#define MULTI_COMP_ALPHA	0.05


static int analyze_rank_sum(RANK *, int, int, double *, int, int, FILE *);
static void print_result_of_analyze_rank_sum(int, double, double, char *,FILE *);
static double dsum_sqr(double *, int);
static double calc_adjust_for_trt(RANK *);
static double calc_adjust_for_blks(RANK *);
static void do_multi_comp_for_anovar(int, int, char **, double *, double *, char *, int *, int **, int, FILE *);
static void calc_stat_ranks(int, int, int **, double*);
static double calc_comp_value(int, int);
static void print_ANOVAR_comp_matrix(int, int **, char **, char *, int *, FILE *);

/****************************************************************/
/*   main procedure to perform a two-way analysis of variance by*/
/*   ranks.  first the sum of the treatment and block ranks     */
/*   are summed then if there is a difference between at least  */
/*   one of the blocks or treatments, then a multiple comparison*/
/*   is performed                                               */
/****************************************************************/
void compute_anovar(RANK *rank, int verbose, int report, int ***out_sys_winner, char *outroot, int feedback, double ***out_conf)
{
    double *sum_trt_rank, *sum_blk_rank;
    char pad[FULL_SCREEN];
    int i,j, **trt_winner, **blk_winner;
    double **conf;
    FILE *fp = stdout;
    
    if (report || verbose){
	char *f = rsprintf("%s.anovar",outroot);
	if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(f,"w")) ==
	    (FILE *)0){
	    fprintf(stderr,"Warning: Open of %s for write failed.  "
		           "Using stdout instead.\n",f);
	    fp = stdout;
	} else
	    if (feedback >= 1) printf("        Output written to '%s'\n",f);
    }

    alloc_2dimZ(conf,rank->n_trt,rank->n_trt,double,0.0);
    alloc_2dimZ(trt_winner,rank->n_trt,rank->n_trt,int,NO_DIFF);
    alloc_2dimZ(blk_winner,rank->n_blk,rank->n_blk,int,NO_DIFF);
    *out_sys_winner = trt_winner;
    *out_conf = conf;

    alloc_singZ(sum_trt_rank,rnk_trt(rank),double,0.0);
    alloc_singZ(sum_blk_rank,rnk_blks(rank),double,0.0);

    /*  sum the ranks */
    for (i=0;i<rnk_blks(rank);i++)
        for (j=0;j<rnk_trt(rank);j++){
            sum_trt_rank[j] += Vrnk_t_rank(rank,i,j);
            sum_blk_rank[i] += Vrnk_b_rank(rank,i,j);
	}

    if (verbose){
        set_pad(pad,FRI_TITLE, FULL_SCREEN);
        fprintf(fp,"\n\n\n%s%s\n",pad,FRI_TITLE);
        fprintf(fp,"%s----------------------------------------------\n\n",pad);
        set_pad_cent_n(pad,SCREEN_WIDTH, FULL_SCREEN);
    fprintf(fp,"%s\tHo: Testing the hypothesis that all recognizers are the same",
                 pad);
    }
    /* if at least one difference, do multiple comparison */
    if (analyze_rank_sum(rank,rnk_trt(rank),rnk_blks(rank),
			 sum_trt_rank,
                         FOR_TREATMENTS,verbose,fp) == TEST_DIFF)
	clear_2dim(trt_winner,rank->n_trt,rank->n_trt,NO_DIFF);
        do_multi_comp_for_anovar(rnk_trt(rank),rnk_blks(rank),
                                 rnk_t_name(rank),sum_trt_rank,
                                 ovr_t_rank(rank),RANK_TITLEA,
                                 srt_t_rank(rank),trt_winner,verbose,fp);
    if (verbose) form_feed(fp);

    if (verbose){
        set_pad(pad,FRI_TITLE, FULL_SCREEN);
        fprintf(fp,"\n\n\n%s%s\n",pad,FRI_TITLE);
        fprintf(fp,"%s----------------------------------------------\n\n",pad);
        set_pad_cent_n(pad,SCREEN_WIDTH, FULL_SCREEN);
       fprintf(fp,"%s\tHo: Testing the hypothesis that all speakers are the same",
                 pad);
    }
    /* if at least one difference, do multiple comparison */
    if (analyze_rank_sum(rank,rnk_blks(rank),rnk_trt(rank),sum_blk_rank,
                         FOR_BLOCKS,verbose,fp) == TEST_DIFF)
	clear_2dim(blk_winner,rank->n_blk,rank->n_blk,NO_DIFF);
        do_multi_comp_for_anovar(rnk_blks(rank),rnk_trt(rank),
                                 rnk_b_name(rank),sum_blk_rank,
                                 ovr_b_rank(rank),RANK_TITLEB,
                                 srt_b_rank(rank),blk_winner, verbose,fp);
    if (verbose && (fp == stdout)) form_feed(fp);
    free_2dimarr(blk_winner,rank->n_blk,int); 
    free_singarr(sum_trt_rank,double);
    free_singarr(sum_blk_rank,double);
    if (fp != stdout) fclose(fp);
}

/*****************************************************************/
/*        X2_r formula:                                          */
/*                       k                                       */
/*             12        --.  2                                  */
/*          -------  *   \   R   - 3b(t+1)                       */
/*         bt(t+1)       /    j                                  */
/*                       --'                                     */
/*                      j = 1                                    */
/*                                                               */
/*   b = num_blk                                                 */
/*   t = num_trt                                                 */
/*   R = Sum of the ranks for that column                        */
/*                                                               */
/*  this is the initial test to see if the treatment or ranks    */
/*  are different                                                */
/*****************************************************************/
static int analyze_rank_sum(RANK *rank, int trt_num, int blk_num, double *sum_arr, int orient, int verbose,FILE *fp)
{
    int df;
    double X2_r, adjust;
    char *subject_blk = "speaker";
    char *subject_trt = "recognition system";
  
    /* calculate the test  statistic */
    df = trt_num - 2;     /* #trt is one based so subtract 2 */
    /* make sure df isn't negative */;
    if (df < 0) df = 0;
    /* make sure df isn't greater than MAX_DF */
    if (df > MAX_DF) df = MAX_DF;

    X2_r = (12.00000 /
                (double)(blk_num*trt_num*(trt_num+1))) *
           (dsum_sqr(sum_arr,trt_num)) -
           3.00000 * (double)blk_num * (double)(trt_num+1);
    if (orient == FOR_TREATMENTS)
        adjust = calc_adjust_for_trt(rank);
    else
        adjust = calc_adjust_for_blks(rank);

    /* if two identical systems are entered, adjust will be 0.0 and core*/
    /* dump.  this avoids this */
    if (adjust == 0.0)
       X2_r = 0.0;
    else
       X2_r /= adjust;

    if (verbose) 
        if (orient == FOR_TREATMENTS)
            print_result_of_analyze_rank_sum(df, X2_r, adjust, subject_trt,fp);        else
            print_result_of_analyze_rank_sum(df, X2_r, adjust, subject_blk,fp);

    /* return the result */
    if (X2_r > X2.df[df].level[GEN_X2_PER])
        return(TEST_DIFF);
    else
        return(NO_DIFF);
}

/*****************************************************************/
/*   print to stdout the results of the rank_sum test            */
/*****************************************************************/
static void print_result_of_analyze_rank_sum(int df, double X2_r, double adjust, char *subject, FILE *fp)
{
    char pad[FULL_SCREEN];
    int i;

    set_pad_cent_n(pad,SCREEN_WIDTH, FULL_SCREEN);
    fprintf(fp,"\n\n\n%s%35sReject if\n",pad,"");
    fprintf(fp,"%s%26sX2_r > X2 of %s %s (%2.3f)\n", pad, "", 
                          X2_pct_str(GEN_X2_PER),
                          X2_df_str(df),
                          X2_score(df,GEN_X2_PER));
    fprintf(fp,"\n"); 
    fprintf(fp,"%s%26sadjustment = %2.3f\n",pad,"",adjust);
    fprintf(fp,"%s%30s  X2_r = %2.3f\n",pad,"",X2_r);
    if (adjust < ADJUST_THRESH){
        fprintf(fp,"\n\n%s\t\t*** Warning:  ties adjustment may have severely\n",
                                                                   pad);
        fprintf(fp,"%s\t\t              exagerated the X2_r value\n\n",pad);
    }
    fprintf(fp,"%sANALYSIS:\n%s--------\n",pad,pad);
    if (X2_r > X2_score(df,GEN_X2_PER)){
     fprintf(fp,"%s\tThe test statistic X2_r shows, with %s confidence, that at\n"
                                            ,pad,X2.neg_per_str[GEN_X2_PER]),
     fprintf(fp,"%s\tleast one %s is significantly different.\n",pad,subject);
        fprintf(fp,"\n");
        fprintf(fp,"%s\tFurther, the probablity of there being a difference is\n",
               pad);
        for (i=GEN_X2_PER;i<MAX_X2_PER;i++)
            if (X2_r < X2_score(df,i+1))
                break;
        if (i==MAX_X2_PER)
            fprintf(fp,"%s\tgreater that %s.\n",pad,X2_neg_pct_str(i));
        else
            fprintf(fp,"%s\tbetween %s to %s.\n",pad,X2_neg_pct_str(i),
                                             X2_neg_pct_str(i+1));
        fprintf(fp,"\n\n");
    }
    else{
       fprintf(fp,"%s\tThe test statistic X2_r shows that at the %s confidence\n"
                          ,pad,X2_neg_pct_str(GEN_X2_PER)),
       fprintf(fp,"%s\tinterval, the %ss are not significantly\n",pad,subject);
        fprintf(fp,"%s\tdifferent.\n\n",pad);
        fprintf(fp,"%s\tFurther, the probablity of there being a difference is\n"
                          ,pad);
        for (i=GEN_X2_PER;i>MIN_X2_PER;i--)
            if (X2_r > X2_score(df,i-1))
                break;
        if (i==MIN_X2_PER)
            fprintf(fp,"%s\tless than %s.\n",pad,X2_neg_pct_str(i));
        else
            fprintf(fp,"%s\tbetween %s to %s.\n",pad,X2_neg_pct_str(i-1),
                                         X2_neg_pct_str(i));
        fprintf(fp,"\n\n");
    }
}

/********************************************************************/
/*  return the doubleing point value of the sum of the squares       */
/********************************************************************/
static double dsum_sqr(double *arr, int len)
{
    int i;
    double tot=0.000000;

    for (i=0;i<len;i++)
        tot += (arr[i]*arr[i]);
    return(tot);
}

/********************************************************************/
/*     Ties adjustments                                             */
/*              b                                                   */
/*             --.                                                  */
/*    1   -    \   T                                                */
/*             /    i                                               */
/*             --'                                                  */
/*          -----------                                             */
/*                2                                                 */
/*            bt(t -1)                                              */
/*                                                                  */
/*               3                                                  */
/*     T =  Sum t   - Sum t        where t = num ties in ith block  */
/*      i        i         i                                        */
/*                                                                  */
/*   to adjust for ties, the test statistic must be taken down      */
/*   by a certain factor, this function returns it                  */
/********************************************************************/
static double calc_adjust_for_trt(RANK *rank)
{
    double *sort_arr;
    int sort_num, *sort_count, i, j, sum_T=0;

    alloc_singarr(sort_arr,rnk_trt(rank),double);
    alloc_singarr(sort_count,rnk_trt(rank),int);

    /* find the ties ans sum Ti */
    for (i=0;i<rnk_blks(rank);i++){
        for (j=0;j<rnk_trt(rank); j++){
            sort_arr[j]=0.0;
            sort_count[j]=0;
	}
        for (j=0;j<rnk_trt(rank); j++){
            sort_num = 0;
            while ((sort_arr[sort_num] != 0.0) && 
                   (Vrnk_t_rank(rank,i,j) != sort_arr[sort_num]))
                sort_num++;
            sort_arr[sort_num] = Vrnk_t_rank(rank,i,j);
            sort_count[sort_num]++;
	}
        for (j=0;j<rnk_trt(rank); j++)
            if (sort_count[j] > 0)
                sum_T += (sort_count[j] * sort_count[j] * sort_count[j])
                           - sort_count[j];
    }
    free_singarr(sort_arr,double);
    free_singarr(sort_count,int);

    /* return the adjustment number */
    return( 1.000000 - (sum_T/
                        (double)(rnk_blks(rank)*
                                rnk_trt(rank)*
                                ((rnk_trt(rank) * rnk_trt(rank)) - 1))));
}

/********************************************************************/
/*  see calc_adjust_for_trt but substitute B for t                  */
/********************************************************************/
static double calc_adjust_for_blks(RANK *rank)
{
    double *sort_arr;
    int sort_num, *sort_count, i, j, sum_T=0;

    alloc_singarr(sort_arr,rnk_blks(rank),double);
    alloc_singarr(sort_count,rnk_blks(rank),int);

    /* find the ties ans sum Ti */
    for (i=0;i<rnk_trt(rank);i++){
        for (j=0;j<rnk_blks(rank); j++){
            sort_arr[j]=0.0;
            sort_count[j]=0;
	}
        for (j=0;j<rnk_blks(rank); j++){
            sort_num = 0;
            while ((sort_arr[sort_num] != 0.0) && 
                   (Vrnk_b_rank(rank,j,i) != sort_arr[sort_num]))
                sort_num++;
            sort_arr[sort_num] = Vrnk_b_rank(rank,j,i);
            sort_count[sort_num]++;
	}
        for (j=0;j<rnk_trt(rank); j++)
            if (sort_count[j] > 0)
                sum_T += (sort_count[j] * sort_count[j] * sort_count[j])
                           - sort_count[j];
    }
    free_singarr(sort_arr,double);
    free_singarr(sort_count,int);
    /* return the adjustment number */
    return( 1.000000 - (sum_T/
                        (double)(rnk_trt(rank)*
                                rnk_blks(rank)*
                                ((rnk_blks(rank) * rnk_blks(rank)) - 1))));
}

/********************************************************************/
/*  for the treatments, calculate and do the MULTIPLE comparison    */
/*  print everything out if requested                               */
/********************************************************************/
static void do_multi_comp_for_anovar(int trt_num, int blk_num, char **trt_names, double *sum_arr, double *ovr_rank_arr, char *title, int *srt_ptr, int **stat_sum, int verbose, FILE *fp)
{
    calc_stat_ranks(trt_num,blk_num,stat_sum,sum_arr);

    if (verbose)
        print_ANOVAR_comp_matrix(trt_num,stat_sum,trt_names,title,srt_ptr,fp);
}

/*********************************************************************/
/*      Multiple comparison test to rank the systems                 */
/*                             ________                              */
/*                            / bt(t+1)                              */
/*     | R  - R  |    >=   z / --------                              */
/*        j    i            V     6                                  */ 
/*                                                                   */
/*  R = the Sum of the nth  column ranks                             */
/*  z = Z score the alpha = 0.05                                     */
/*                          ----                                     */
/*                         t(t-1)                                    */
/*                                                                   */
/*  go through all possible comparisons and rank the treatments for  */
/*  according to the number of treatments beaten or beaten by        */
/*                                                                   */
/*********************************************************************/
static void calc_stat_ranks(int trt_num, int blk_num, int **stat_sum, double *sum_arr)
{
    int comp1, comp2, result;
    double Zcomp_value, compare_value;
    double sqr_root;

    /* calculate the comparison value for the confidence interval */
    Zcomp_value = calc_comp_value(trt_num,blk_num);
    sqr_root = sqrt(blk_num * trt_num * (trt_num - 1) / 6.000000);
    compare_value = (double)Zcomp_value * (double)sqr_root;

    /* go through the comparison and build an array to rank */
    for (comp1=0; comp1 <(trt_num-1); comp1++)
        for (comp2=comp1+1; comp2<(trt_num); comp2++){
            if (fabs(sum_arr[comp1] - sum_arr[comp2]) >= compare_value)
                if (sum_arr[comp1] > sum_arr[comp2])
                    result = 1;
                else
                    result = -1;
            else
                result = 0;
            stat_sum[comp1][comp2] += result;
            stat_sum[comp2][comp1] += result * (-1);
	}
}

/*********************************************************************/
/*   Given the alpha, find the closest approximate Z index to that   */
/*   percentage                                                      */
/*********************************************************************/
static double calc_comp_value(int trt_num, int blk_num)
{
    double Z_area;
    int i;

    Z_area = MULTI_COMP_ALPHA / (double)(trt_num * (trt_num-1));

    if (Z_area < Z_exter(Z1tail,MAX_Z_PER))
        return(Z_score(Z1tail,MAX_Z_PER));

    for (i=MAX_Z_PER+1; i< MIN_Z_PER ; i++)
        if (Z_area < Z_exter(Z1tail,i))
            return(Z_score(Z1tail,i));

    return(Z_score(Z1tail,MIN_Z_PER));
}


static void print_ANOVAR_comp_matrix(int trt_num, int **stat_sum, char **trt_names, char *title, int *srt_ptr, FILE *fp)
{
    char pad[FULL_SCREEN],name_format[50],*hyphens="--------------------";
    char *spaces="                    ";
    int t, t2;
    int max_trt_name_len=4, hy_l=20;

    set_pad(pad,title, FULL_SCREEN);
    fprintf(fp,"\n\n%s%s\n",pad,title);
    set_pad(pad,RANK_TITLE_1, FULL_SCREEN);
    fprintf(fp,"%s%s\n\n\n\n",pad,RANK_TITLE_1);

    for (t=0; t<trt_num; t++)
        if (max_trt_name_len < strlen(trt_names[t]))
            max_trt_name_len = strlen(trt_names[t]);
    set_pad_cent_n(pad,(trt_num+1) * (max_trt_name_len+2+1), FULL_SCREEN);
    sprintf(name_format,"| %%-%ds ",max_trt_name_len);

    /* first line */
    fprintf(fp,"%s|%s",pad,(hyphens+(hy_l-(max_trt_name_len+2))));
    for (t=0; t<trt_num; t++)
       fprintf(fp,"%s",(hyphens+(hy_l-(max_trt_name_len+3))));
    fprintf(fp,"|\n");

    /* systems on top */
    fprintf(fp,"%s|%s",pad,(spaces+(hy_l-(max_trt_name_len+2))));
    for (t=0; t<trt_num; t++)
       fprintf(fp,name_format,trt_names[srt_ptr[t]]);
    fprintf(fp,"|\n"); 

    /* separation line */
    fprintf(fp,"%s|%s+",pad,(hyphens+(hy_l-(max_trt_name_len+2))));
    for (t=0; t<trt_num-1; t++)
       fprintf(fp,"%s",(hyphens+(hy_l-(max_trt_name_len+3))));
    fprintf(fp,"%s|\n",(hyphens+(hy_l-(max_trt_name_len+2))));

    /* systems */
    for (t=0; t<trt_num; t++){
       fprintf(fp,"%s",pad);
       fprintf(fp,name_format,trt_names[srt_ptr[t]]);
       for (t2=0; t2<trt_num; t2++){
           if (t < t2)
               switch (stat_sum[srt_ptr[t]][srt_ptr[t2]]) {
                   case -1: fprintf(fp,name_format,trt_names[srt_ptr[t]]);
                            break;
                   case 1:  fprintf(fp,name_format,trt_names[srt_ptr[t2]]);
                            break;
                   default: fprintf(fp,name_format,"same");
                            break;
	       }
           else
               fprintf(fp,name_format,"");
       }
       fprintf(fp,"|\n");
    }

    /* last line */
    fprintf(fp,"%s|%s",pad,(hyphens+(hy_l-(max_trt_name_len+2))));
    for (t=0; t<trt_num; t++)
       fprintf(fp,"%s",(hyphens+(hy_l-(max_trt_name_len+3))));
    fprintf(fp,"|\n");
}


