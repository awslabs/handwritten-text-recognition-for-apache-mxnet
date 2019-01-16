#define P_CORR      0x01
#define P_SUB       0x02
#define P_INS       0x04
#define P_DEL       0x08
#define P_MRG       0x10
#define P_SPL       0x20

typedef struct PATH_SET_struct {
    void *a_ptr,         /* pointer to A arc's data */
         *b_ptr;         /* pointer to B arc's data */
    char eval;           /* flag to designated the pair as Corr, Ins, Del, Sub */
} PATH_SET;


#define PA_NONE           0x0000
#define PA_CHAR_ALIGN     0x0001
#define PA_CASE_SENSE     0x0002
#define PA_HYP_WTIMES     0x0004
#define PA_REF_WTIMES     0x0008
#define PA_HYP_TIMES      0x0010
#define PA_REF_TIMES      0x0020
#define PA_HYP_CONF       0x0040
#define PA_REF_CONF       0x0080
#define PA_HYP_WEIGHT     0x0100
#define PA_REF_WEIGHT     0x0200
#define PA_HYP_SPKR       0x0400
#define PA_HYP_ISSPKRSUB  0x0800

typedef struct PATH_struct {
    int max;             /* maximum number of PATH_set structures in pset*/
    int num;             /* the current number of used structures in pset*/

    PATH_SET *pset;      /* the array of structures */
    
    char *id;            /* utterance id */
    char *labels;        /* any labels attached to the id's */
  
    char *file;          /* audio from whence the data came, only in place 
			    for data aligned by ctm or stm */
    char *channel;       /* channel . . . ditto */

    double ref_t1;        /* beginning time for the reference */
    double ref_t2;        /* ending time for the reference */

    double hyp_t1;        /* beginning time for the hypothesis */
    double hyp_t2;        /* ending time for the hypothesis */

    int attrib;          /* an attributes of the path */

    int sequence;         /* Sequence number: during alignment */
} PATH;

PATH *PATH_alloc(int size);
void PATH_print(PATH *path, FILE *fp, int max);
void PATH_n_print(PATH *, FILE *, int from, int to, int max);
void PATH_print_html(PATH *path, FILE *fp, int max, int header);
void PATH_n_print_html(PATH *path, FILE *fp, int from, int to, int max, int header);
void PATH_print_wt(PATH *path, FILE *fp);
void PATH_increment_WORD_use(PATH *path);
void PATH_free(PATH *path);
void PATH_add_utt_id(PATH *path, char *utt_id);
void PATH_add_label(PATH *path, char *label);
void PATH_add_file(PATH *path, char *file);
void PATH_add_channel(PATH *path, char *channel);
void PATH_set_sequence(PATH *path);
void check_space_in_PATH(PATH *path);
void PATH_append(PATH *path, void *ap, void *bp, int eval);
void PATH_remove(PATH *path);
void sort_PATH_time_marks(PATH *path);

