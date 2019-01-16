/****************************************************************/
/*                                                              */
/*           FILE: stat_dist.h                                  */
/*           WRITTEN BY: Jonathan G. Fiscus                     */
/*           DATE: April 14 1989                                */
/*                 NATIONAL INSTITUTE OF STANDARDS              */
/*                         AND TECHNOLOGY                       */
/*                    SPEECH RECOGNITION GROUP                  */
/*           USAGE: for definition of the statistical table     */
/*                  structures and macros to access them        */
/*                                                              */
/****************************************************************/

/****************************************************************/
/*    test results						*/
/****************************************************************/
#define TEST_DIFF	1
#define NO_DIFF		0

/****************************************************************/
/*    Z table lookup defines                                    */
/****************************************************************/

#define PER99	0
#define PER98	1
#define PER97	2
#define PER96	3
#define PER95	4
#define PER94	5
#define PER93	6
#define PER92	7
#define PER91	8
#define PER90	9

#define MAX_Z_PER	PER99
#define MIN_Z_PER 	PER90
#define GEN_Z_PER	PER95

/****************************************************************/
/*    chi squared table lookup defines                          */
/****************************************************************/

#define PER99	0
#define DF1	0
#define DF2	1
#define DF3	2
#define DF4	3
#define DF5	4
#define DF6	5
#define DF7	6
#define DF8	7
#define DF9	8
#define DF10	9
#define DF11	10
#define DF12	11
#define DF13	12
#define DF14	13
#define DF15	14
#define DF16	15
#define DF17	16
#define DF18	17
#define DF19	18
#define DF20	19
#define DF21	20
#define DF22	21
#define DF23	22
#define DF24	23
#define DF25	24
#define DF26	25
#define DF27	26
#define DF28	27
#define DF29	28
#define DF30	29

#define  X2PER99 0
#define  X2PER98 1
#define  X2PER95 2
#define  X2PER90 3
#define  X2PER80 4
#define  X2PER70 5
#define  X2PER50 6
#define  X2PER30 7
#define  X2PER20 8
#define  X2PER10 9 
#define  X2PER5  10
#define  X2PER2  11
#define  X2PER1  12
#define  X2PER_1 13

#define MAX_DF DF30
#define MIN_DF DF1
#define MAX_X2_PER X2PER_1
#define MIN_X2_PER X2PER99
#define GEN_X2_PER X2PER5

/************************************************************/
/*     Statistical distribution structure definitions       */
/************************************************************/

typedef struct Z_struct{
    double z;
    char *str;
    char *exterior_str;
    double perc_interior;
} Z_STRUCT;

typedef struct X2_df{
    char *str;
    double  level[MAX_X2_PER+1];
} X2_DF;
    
typedef struct X2_struct{
    double per[MAX_X2_PER+1];
    char *per_str[MAX_X2_PER+1];
    char *neg_per_str[MAX_X2_PER+1];
    X2_DF df[MAX_DF+1];
} X2_STRUCT;

#define Z_score(_strct,_pct)	_strct[_pct].z
#define Z_str(_strct,_pct)	_strct[_pct].str
#define Z_ext_str(_strct,_pct)	_strct[_pct].exterior_str
#define Z_inter(_strct,_pct)	_strct[_pct].perc_interior
#define Z_exter(_strct,_pct)	(1.000000 - Z_inter(_strct,_pct))

#define X2_pct_str(_pct)	X2.per_str[_pct]
#define X2_neg_pct_str(_pct)	X2.neg_per_str[_pct]
#define X2_df(_df)		X2.df[_df]
#define X2_df_str(_df)		X2_df(_df).str
#define X2_score(_df,_pct)	X2_df(_df).level[_pct]

extern Z_STRUCT Z2tail[];
extern Z_STRUCT Z1tail[];
extern X2_STRUCT X2;

#define SILENT			FALSE
#define VERBOSE			TRUE

#define DEFAULT_MIN_NUM_GOOD	2

#if defined(__STDC__) || defined(__GNUC__) || defined(sgi)
#define PROTO(ARGS)	ARGS
#else
#define PROTO(ARGS)	()
#endif

/* statdist.c */ void dump_X2_table PROTO((void)) ;
/* statdist.c */ void calc_mean_var_std_dev_Zstat PROTO((int *Z_list, int num_Z, double *mean, double *variance, double *std_dev, double *median, double *Z_stat)) ;
/* statdist.c */ void calc_mean_var_std_dev_Zstat_double PROTO((double *Z_list, int num_Z, double *mean, double *variance, double *std_dev, double *median, double *Z_stat));
/* statdist.c */ int print_Z_analysis PROTO((double Z_stat)) ;
/* statdist.c */ int Z_pass PROTO((double Z_stat)) ;
/* statdist.c */ void calc_two_sample_z_test_double PROTO((double *l1, double *l2, int num_l1, int num_l2, double *Z)) ;
/* statdist.c */ double compute_acc_binomial PROTO((int R, int n, double p)) ;
/* statdist.c */ double seq_mult PROTO((int f, int )) ;
/* statdist.c */ double n_CHOOSE_r PROTO((int n, int r)) ;

void compute_anovar PROTO((RANK *, int, int, int ***, char *, int, double ***));

int compute_signtest_for_treatment(RANK *, int, int, char *, char *, int, int, FILE *, double *);
void perform_signtest PROTO((RANK *rank, int verbose, int report, char *formula_str, char formula_id, int ***winner, char *, int, double ***));
int sign_test_analysis PROTO((int, int, int, char *, char *, int, double, int, char *, char *, int *, int, FILE *, double *confidence));

int compute_wilcoxon_for_treatment(RANK *rank, int treat1, int treat2, char *block_id, char *formula_str, int verbose, int zero_is_best, FILE *fp, double *conf);
void perform_wilcoxon PROTO((RANK *rank, int verbose, int report, char *formula_str, char formula_id, int ***winner, char *, int, double ***confidence));

int do_McNemar_by_sent(SCORES *scor1, SCORES *scor2, int verbose, FILE *fp, double *conf);
void McNemar_sent(SCORES *scor[], int nscor, int ***winner, char *testname, int print_results, int verbose, char *, int, double ***conf);
int do_McNemar(int **table, char *name1, char *name2, int verbose, FILE *fp, double *conf);

void do_mtch_pairs_seg_analysis(SCORES *scor[], int nscor, char *t_name, int seg_ave, int seg_long);
int do_mtch_pairs_on_sys(SCORES *scor[], int nscor, int sys1_ind, int sys2_ind, double *sys1_pct, double *sys2_pct, int *num_seg, int *max_seg, double *Z_stat, double *seg_per_sent, int min_num_good, int verbose, FILE *fp, double *conf);
void do_mtch_pairs(SCORES *scor[], int nscor, char *min_num_good_str, char *test_name, int print_report, int verbose, int ***winner, char *, int, double ***confidence);


double normprob(double z);
