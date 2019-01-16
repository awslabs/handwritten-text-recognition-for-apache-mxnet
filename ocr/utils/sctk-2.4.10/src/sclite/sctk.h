/* main include file for the Scoring ToolKit (SCTK) */
/* Created: Jul, 28, 1997 */

#define TK_VERSION "1.3"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "text.h"
#include "booldef.h"
#include "wtokstr1.h"
#include "path.h"
#include "proto.h"
#include "order.h"
#include "stm.h"
#include "pad.h"
#include "sgml.h"
#include "alex.h"

#include "netstr1.h"
#include "llist.h"

#define db_enter_msg(_s,_db) {if (_db < db_level) printf("DB: Entering %s\n",_s);}
#define db_leave_msg(_s,_db) {if (_db < db_level) printf("DB: Leaving  %s\n",_s);}
#define strdup_safe(_s,_p) (char *)TEXT_strdup((TEXT *)_s)
#define malloc_safe(_s,_p) malloc(_s)
#define streq(cs,ct)       (strcmp(cs,ct) == 0)
#ifdef __STDC__
/* rsprinaztf.c */ char *rsprintf PROTO((char *format , ...));
#else
/* rsprintf.c */ char *rsprintf PROTO((va_alist));
#endif
#define F_ROUND(_n,_p)    (((float)((int)((_n) * pow(10.0,(float)(_p)) + ((_n>0.0)?0.5:-0.5)))) / pow(10.0,(float)(_p)))
#define F_CEIL(_n,_p)    ( ((double)ceil((double)((_n) * pow(10.0,(double)(_p))))) / pow(10.0,(double)(_p)))
#define pct(num,dem)    (((dem)==0) ? 0 : (double)(num)/(double)(dem) * 100.0)
#define form_feed(_f)     fprintf(_f,"");

#define TRUE		1
#define FALSE		0

/* bit field accessors, and query macros */ 
#define BF_isSET(_v,_f)  (((_v) & (_f)) > 0)
#define BF_notSET(_v,_f) (((_v) & (_f)) == 0)
#define BF_SET(_v,_f)    _v |= (_f)
#define BF_FLIP(_v,_f)   _v ^= (_f)
#define BF_UNSET(_v,_f)  _v &= ~(_f)

#ifndef MAX
#define MAX(_a,_b) (((_a) > (_b)) ? (_a) : (_b))
#endif
#ifndef MIN
#define MIN(_a,_b) (((_a) < (_b)) ? (_a) : (_b))
#endif

#define MAX3(_a,_b,_c) (MAX(_a,(MAX(_b,_c))))
#ifndef M_LOG2E
#define M_LOG2E         1.4426950408889634074
#endif

/* using slope to linearly interprelate Based on: M = (y2 - y1) / (x2 - y2)
   (y2 - y1) / (x2 - y2)                     = (y2 - QY) / (x2 - PX)
    (x2 - PX) * (y2 - y1) / (x2 - y2)        = (y2 - QY)
    y2 - ((x2 - PX) * (y2 - y1) / (x2 - x2)) = QY
*/
#define interpelate_y_from_x(_x1, _y1, _x2, _y2, _kn_X) \
	(_y2 - ((_x2 - _kn_X) * (_y2 - _y1) / (_x2 - _x1)))

#define SCREEN_WIDTH 80
#define FULL_SCREEN 132
#define INF_ASCII_TOO    1
#define CALI_ON          1
#define CALI_NOASCII     2
#define CALI_DELHYPHEN   4

#include "word.h"
#include "cores.h"
#include "memory.h"
#include "rpg.h"


typedef struct grp_score{
    char *name;
    int corr;                 /* num correct words in sent */
    int ins;                  /* num inserted words in sent */
    int del;                  /* num deleted words in sent */
    int sub;                  /* num substituted words in sent */
    int merges;               /* num of merges */
    int splits;               /* num of splits */

    double weight_ref;        /* weighted sum of reference words in sent */
    double weight_corr;       /* weighted sum of correct words in sent */
    double weight_ins;        /* weighted sum of inserted words in sent */
    double weight_del;        /* weighted sum of deleted words in sent */
    double weight_sub;        /* weighted sum of substituted words in sent */
    double weight_merges;     /* weighted sum of of merges */
    double weight_splits;     /* weighted sum of of splits */

    int nsent;                /* number of sentences, utts, ... */
    int serr;                 /* number of sents with an error */
    int max_path;             /* Size of PATH array */
    int num_path;             /* number of PATH's in PATH array */
    PATH **path;              /* array of PATH pointers */
} GRP;

typedef struct pathlabel_item_struct{
    char *id;              /* the label's id code */
    char *title;           /* Column heading title for the lable */
    char *desc;            /* Descriptive text for the label */
} PATHLABEL_ITEM;

typedef struct category_item_struct{
    char *id;              /* the category's id code */
    char *title;           /* Column heading title for the category */
    char *desc;            /* Descriptive text for the catagory */
} CATEGORY_ITEM;

typedef struct arbitrary_subset_label_struct{
    int max_plab;          /* max PATHLABEL_ITEMS in 'plab' array */
    int num_plab;          /* current number of items in 'plab' array */
    PATHLABEL_ITEM *plab;  /* list of path labels */
    int max_cat;           /* max CATEGORY_ITEMS in 'cat' array */
    int num_cat;           /* current number of items in 'cat' array */
    CATEGORY_ITEM *cat;    /* list of categories */
} ARB_SSET;


typedef struct set_score{
    char *title;           /* the title to call the system in reports */
    char *ref_fname;       /* filename of the reference file */
    char *hyp_fname;       /* filename of the hypothesis file */
    char *creation_date;   /* creation date */
    int frag_corr;         /* if the comline fragment correct flag */
    int opt_del;           /* if the comline opt_del flag set */
    int weight_ali;        /* if the comline word weight alignment set */
    char *weight_file;     /* if the comline word weight alignment set, wwl filename */

    int max_grp;           /* maximum number of path groups (speakers) */
    int num_grp;           /* current number of path groups */
    GRP *grp;              /* a list of path groups */
    
    ARB_SSET aset;         /* struct of arbitrary labelling */
} SCORES;

#if __STDC__
FILE * readpipe (char *progname, ...);
#else
FILE * readpipe ();
#endif

#include "corresp.h"
#include "wwscr_f.h"

SCORES *SCORES_init(char *name, int ngrp);
void SCORES_free(SCORES *scor);
int SCORES_get_grp(SCORES *sc, char *grpname);
int find_PATHLABEL_id(SCORES *sc, char *id);
int parse_input_comment_line(SCORES *sc, TEXT *buf);
void load_comment_labels_from_file(SCORES *scor, char *labfile);
void add_PATH_score(SCORES *sc, PATH *path, int grp, int keep_path);
void dump_SCORES(SCORES *sc, FILE *fp);
void dump_SCORES_alignments(SCORES *sc, FILE *fp, int lw, int full);
void dump_SCORES_sgml(SCORES *sc, FILE *fp, TEXT *token_separator, TEXT *token_attribute_separator);
int load_SCORES_sgml(FILE *fp, SCORES **scor, int *nscor, int maxscor);
void print_system_summary(SCORES *sc, char *sys_root_name, int do_sm, int do_raw, int do_weighted, int feedback);
void print_N_system_summary(SCORES *sc[], int nsc, char *out_root_name, char *test_name, int do_raw, int feedback);
void print_N_system_executive_summary(SCORES *sc[], int nsc, char *out_root_name, char *test_name, int do_raw, int feedback);
void print_SCORES_compare_matrix(SCORES *scor[], int nscor, int **winner, char *tname, char *matrix_name);
void print_lur(SCORES *sc, char *sys_root_name, int feedback);
void print_N_lur(SCORES *scor[], int nscor, char *outname, char *test_name, int feedback);
void compute_SCORE_nce(SCORES *sc, double *nce_system, double *nce_a);
void print_N_SCORE(SCORES *scor[], int nscor, char *outname, int max, int feedback, int score_diff);

enum id_types {WSJ,RM,ATIS,SWB,SPUID,SP};

SCORES *align_trans_mode_dp(char *ref_file, char *hyp_file, char *title, int keep_path, int case_sense, int feedback, int char_align, enum id_types idt, int infer_word_seg, char *lexicon, int frag_correct, int opt_del, int inf_no_ascii, WWL *wwl, char *lm_file);
SCORES *align_ctm_to_stm_dp(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, int char_align, enum id_types idt, int infer_word_seg, char *lexicon, int frag_correct, int opt_del, int inf_no_ascii, int reduce_ref, int reduce_hyp, int left_to_right, WWL *wwl, char *lm_file);

SCORES *align_trans_mode_diff(char *ref_file, char *hyp_file, char *title, int keep_path, int case_sense, int feedback, enum id_types idt);
SCORES *align_ctm_to_stm_diff(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, enum id_types idt);
SCORES *align_text_to_stm(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, enum id_types idt);
void expand_words_to_chars(ARC *arc, void *ptr);
void decode_opt_del(ARC *arc, void *ptr);
void decode_fragment(ARC *arc, void *ptr);

PATH *network_dp_align_texts(TEXT *ref, NETWORK *rnet, TEXT *hyp, NETWORK *hnet, int char_align, int case_sense, char *id, int fcorr, int opt_del, int time_align, WWL *wwl, char *lm_file);
PATH *infer_word_seg_algo1(TEXT *ref, TEXT *hyp, NETWORK *hnet, int case_sense,char *id, char *lex_fname, int fcorr, int opt_del, int no_ascii);
PATH *infer_word_seg_algo2(TEXT *ref, TEXT *hyp, NETWORK *hnet, int case_sense,char *id, char *lex_fname, int fcorr, int opt_del, int flags);

SCORES *align_ctm_to_ctm(char *hyp_file, char *ref_file, char *set_title, int feedback, int frag_corr, int opt_del, int case_sense, int time_align, int left_to_right, WWL *wwl, char *lm_file);

double overlap(double s1_t1, double s1_t2, double s2_t1, double s2_t2);

int score_dtl_sent(SCORES *scor, char *sys_root_name, int feedback);
int score_dtl_spkr(SCORES *scor, char *sys_root_name, int feedback);
int score_dtl_overall(SCORES *scor, char *sys_root_name, int feedback);

char *get_date(void);

int hyp_confidences_available(SCORES *scor);
int make_SCORES_DET_curve(SCORES *scor[], int nscor, char *outroot, int feedback, char *test_name);
int make_binned_confidence(SCORES *scor, char *outroot, int feedback);
int make_scaled_binned_confidence(SCORES *scor, char *outroot, int bins, int feedback);
int make_confidence_histogram(SCORES *scor, char *outroot, int feedback);

/* debug levels */
/*   level 1:  print function entrances */
/*   level 2:  print function arguements */
/*   level 5:  intermediate status */

extern int db;

#define LINE_LENGTH 500
#define INF_SEG_ALGO1   1
#define INF_SEG_ALGO2   2

#include "rank.h"
#include "statdist.h"


#define scfp stderr

#ifdef MAIN
int db=0, db_level;
char *pdb="";
#else
extern int db, db_level;
extern char *pdb;
#endif

#ifdef DIFF_EXE
#define DIFF_ENABLED 1
#define DIFF_PROGRAM DIFF_EXE
#else
#define DIFF_ENABLED 0
#define DIFF_PROGRAM ""
#endif



#ifdef WITH_SLM
#include "slm_v2/include/SLM2.h"
#endif

#include "slm_intf.h"


#ifdef __cplusplus
}
#endif

