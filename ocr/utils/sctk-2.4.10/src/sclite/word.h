/*  File: word.h
    Desc: Definition of the word struct
 */

typedef struct WORD_STRUCT{
    int use;                            /* number of times this structure */
                                        /* is pointed to */
    TEXT *value;                        /* the word text */
    TEXT *intern_value;                 /* the word text, for internal compares */
    int value_id;                       /* Dictionary index to word */
    TEXT *tag1;                         /* Optional tag associated with the word */
    TEXT *tag2;                         /* Optional tag associated with the word */
    int opt_del;                        /* boolean flag to identify 
					   optionally deletable words */
    int frag_corr;                      /* boolean flag to identify the word as a
					   fragment on correct if compared to its
					   substring */
    double conf;                        /* Opt. confidence value for the word */
    double T1,                          /* beginning word time */
           T2,                          /* ending word time */
           T_dur;                       /* word duration time (T2 - T1) */
    double weight;                      /* The Weight assigned to a word from a WWL list */
} WORD;

#define NULL_WORD (WORD *)0
#define WORD_OPT_DEL_PRE_STR  "("
#define WORD_OPT_DEL_PRE_CHAR '('
#define WORD_OPT_DEL_POST_STR  ")"
#define WORD_OPT_DEL_POST_CHAR ')'
#define WORD_FRAG_STR  "-"
#define WORD_FRAG_CHAR '-'
#define WORD_SGML_SUB_WORD_SEP_STR ";" 
#define WORD_SGML_SUB_WORD_SEP_CHR ';'
#define WORD_SGML_ESCAPE '\\'

/*  Functions defined in word.c */

float wwd_WORD(void *p1, void *p2, int (*cmp)(void *, void *));
float wwd_time_WORD(void *p1, void *p2, int (*cmp)(void *, void *));
float wwd_weight_WORD(void *p1, void *p2, int (*cmp)(void *p1, void *p2));
float wwd_WORD_rover(void *p1, void *p2, int (*cmp)(void *p1, void *p2));
WORD *get_WORD(void);
WORD *new_WORD_parseText(TEXT *t, int id, double t1, double t2, double conf, int fcorr, int odel, double weight);
WORD *new_WORD(TEXT *t, int id, double t1, double t2, double conf, TEXT *tag1, TEXT *tag2, int frag, int opt_del, double weight);
void release_WORD(void *p);
void *copy_WORD(void *p);
void *copy_WORD_via_use_count(void *p);
void *make_empty_WORD(void *p);
void print_WORD(void *p);
void print_WORD_wt(void *p);
void print_2_WORD_wt(void *p, void *p2, FILE *fp);
int equal_WORD2(void *tw1, void *tw2);
#ifdef old
int equal_WORD(void *tw1, void *tw2);
int equal_WORD_wfrag(void *p1, void *p2);
#endif
void *append_WORD(void *w1, void *w2);
void *append_WORD_no_NULL(void *w1, void *w2);
int null_alt_WORD(void *p);
int opt_del_WORD(void *p);
int use_count_WORD(void *p, int n);
void set_WORD_tag1(WORD *w, TEXT *t);
void sgml_dump_WORD(WORD *w, FILE *fp);
