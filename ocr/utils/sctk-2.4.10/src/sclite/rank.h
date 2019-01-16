/**********************************************************************/
/*                                                                    */
/*           FILE: rank.h                                             */
/*           WRITTEN BY: Jonathan G. Fiscus                           */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*           USAGE: for definition of the RANK structure and          */
/*                  all it's macros for appropriate access            */
/*                                                                    */
/**********************************************************************/

/**********************************************************************/
/*   sizes of the rank structure                                      */
#define MAX_BLOCKS	60
#define MAX_TREATMENTS  60
#define FOR_BLOCKS	TRUE
#define FOR_TREATMENTS  FALSE

/**********************************************************************/
/*  command line values to change the means of calculating percentages*/
#define PER_CORR_REC		'R'
#define TOTAL_ERROR		'E'
#define WORD_ACCURACY		'W'

#define CORR_REC_STR		"Speaker Percent Correctly Recognized"
#define TOT_ERR_STR		"Speaker Word Error Rate (%)"
#define WORD_ACC_STR		"Speaker Word Accuracy Rate (%)"

/**********************************************************************/
/*   typedef for the RANK structure                                   */
typedef struct rank_struct{
   int n_blk;                  /* number of blocks */ 
   int n_trt;                  /* number of treatments */
   char **blk_name;            /* string names of blocks */
   int *blk_srt_ind;           /* indexes that to blocks to sort them */
   double **blk_ranks;          /* 2DIM array for ranks for blocks over trts */
   double *blk_Ovr_ranks;       /* ranks after ANOVAR */

   char **trt_name;            /* string names of treatments */
   double *trt_Ovr_ranks;       /* ranks after ANOVAR */
   double **trt_ranks;          /* 2DIM array for ranks for trts over blocks */
   double **pcts;               /* the actual percentages for trts and blks */
   int *trt_srt_ind;           /* indexes that sorts trts into orders */
} RANK;

void init_RANK_struct_from_SCORES(RANK *rank, SCORES *scor[], int nscor,  char *calc_formula);
void free_RANK(RANK *rank);
void rank_on_pcts(RANK *rank, int ordering);
void dump_full_RANK_struct(RANK *rank, char *t_name, char *b_name, char *blk_label, char *trt_label, char *formula_str, char *test_name, char *blk_desc, char *trt_desc);
void print_n_winner_comp_matrix(RANK *rank, int ***wins, char **win_ids, int win_cnt, int page_width,FILE *fp);
void print_composite_significance(RANK *rank, int pr_width, int num_win, int ***wins, char **win_desc, char **win_str1, int matrix, int report, char *test_name, char *outroot, int feedback, char *outdir);
void print_composite_significance2(RANK *rank, int pr_width, int num_win, int ***wins, double ***conf, char **win_desc, char **win_str, int matrix, int report, char *test_name, char *outroot, int feedback, char *outdir);
void print_n_winner_comp_report(RANK *rank, int ***wins, char **win_ids, char **win_str, char **win_desc, int win_cnt, int page_width, char *testname, char *outdir);
int formula_index(char *str);
char *formula_str(char *str);


void print_rank_ranges(RANK *rank, char *percent_desc, char *testname, char *outroot, int);
void print_gnu_rank_ranges(RANK *rank, char *percent_desc, char *testname,char *basename, int);
void print_gnu_rank_ranges2(RANK *rank, char *percent_desc, char *testname,char *basename, int);
void print_trt_comp_matrix_for_RANK_one_winner(int **winner, RANK *rank, char *title, char *formula_str, char *block_id, FILE *fp);

/**********************************************************************/
/*   RANK structure access macros                                     */
#define rnk_blks(_r)	_r->n_blk
#define rnk_trt(_r)	_r->n_trt

#define rnk_t_rank(_r)	_r->trt_ranks
#define rnk_t_rank_arr(_r,_n)	rnk_t_rank(_r)[_n]
#define Vrnk_t_rank(_r,_b,_t)	rnk_t_rank(_r)[_b][_t]

#define rnk_b_rank(_r)	_r->blk_ranks
#define rnk_b_rank_arr(_r,_n)	rnk_b_rank[_n]
#define Vrnk_b_rank(_r,_b,_t)	rnk_b_rank(_r)[_b][_t]

#define rnk_pcts(_r)	_r->pcts
#define rnk_pcts_arr(_r,_n)	_r->pcts[_n]
#define Vrnk_pcts(_r,_b,_t)	rnk_pcts(_r)[_b][_t]

#define rnk_b_name(_r)	_r->blk_name
#define Vrnk_b_name(_r,_b)	rnk_b_name(_r)[_b]

#define rnk_t_name(_r)	_r->trt_name
#define Vrnk_t_name(_r,_t)	rnk_t_name(_r)[_t]

#define ovr_t_rank(_r)		_r->trt_Ovr_ranks
#define Vovr_t_rank(_r,_t)	ovr_t_rank(_r)[_t]

#define ovr_b_rank(_r)		_r->blk_Ovr_ranks
#define Vovr_b_rank(_r,_b)	ovr_t_rank(_r)[_b]

#define srt_t_rank(_r)		_r->blk_srt_ind
#define Vsrt_t_rank(_r,_t)	srt_t_rank(_r)[_t]

#define srt_b_rank(_r)		_r->trt_srt_ind
#define Vsrt_b_rank(_r,_b)	srt_b_rank(_r)[_b]

