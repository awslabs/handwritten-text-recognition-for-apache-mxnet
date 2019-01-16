/* file wtokstr1.h                           */
/* data structure for a list of word tokens, */
/* as in time-mark  (*.marked, *.mrk) files. */
/* Modified 2/28/95 by Jon Fiscus & Bill     */
/* Fisher to include a few more fields.      */
/* Modified 9/14/95 by JF to make it's size  */
/*          dynamic                          */

 typedef struct
   {char *turn; /* the Channel id */
    char *conv; /* the Conversation/File ID */
    double t1;  /* beginning time of token */
    double dur; /* duration of token       */
    TEXT *sp;   /* spelling of token       */
    float confidence; /* confidence factor of toke, OPTIONAL */
    int correct; /* OPTIONAL flag indicating if the word is correct */
    boolean overlapped;
    boolean mispronounced;
    boolean unsure;
    boolean comment;
    boolean bad_marking;
    boolean crosstalk;
    boolean alternate;
    boolean ignore;
   } WTOKE1;

 typedef struct
   {int n;                           /* number of word tokens in table */
    int max;                         /* maximum num of words the the table */
    int s;                           /* the current beginning of the table 
				       during processing */
    char *id;                       /* the utterance id of the segement/file */
    WTOKE1 *word;                   /* table of word tokens           */
    int has_conf;                   /* boolean flag, 1 if any words have the */
                                    /* optional confidence flag. */
   } WTOKE_STR1;


void fill_WTOKE_structure(WTOKE_STR1 *ctm_segs, FILE *fp_ctm, char *ctm_file, int *ctm_file_end, int case_sense);
void fill_mark_struct(FILE *fp, WTOKE_STR1 *word_tokens, char *fname, boolean *end_of_file, int *perr, int case_sense);
void locate_boundary(WTOKE_STR1 *seg, int start, int by_conv, int by_turn, int *end);
WTOKE_STR1 *WTOKE_STR1_init(char *);
void dump_word_tokens2(WTOKE_STR1 *word_tokens, int start, int lim);
void locate_WTOKE_boundary(WTOKE_STR1 *seg, int start, int by_conv, int by_turn, int *end);
void reset_WTOKE_flag(WTOKE_STR1 *seg,char *flag_name);
void dump_WTOKE_words(WTOKE_STR1 *word_tokens, int start, int lim, char *file);
void free_mark_file(WTOKE_STR1 *word_tokens);

