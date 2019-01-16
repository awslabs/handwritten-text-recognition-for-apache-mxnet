
typedef struct stm_seg_struct{
    TEXT *file;
    TEXT *chan;
    TEXT *spkr;
    double t1;
    double t2;
    int  flag1;
    TEXT *labels;
    TEXT *text;
} STM_SEG;

typedef struct stm_struct{
    int max;
    int s;
    int num;
    STM_SEG *seg;
} STM;

STM *alloc_STM(int n);
void expand_STM(STM *stm);
void free_STM(STM *stm);
void fill_STM(FILE *fp, STM *stm, char *fname, boolean *end_of_file, int case_sense, int *perr);
void locate_STM_boundary(STM *stm, int start, int by_file, int by_chan, int *end);
void dump_STM_words(STM *stm,int s, int e, char *file);
void dump_STM(STM *stm, int s, int e);
void read_stm_line(TEXT **buf, int *len, FILE *fp);
void parse_stm_line(STM_SEG *seg, TEXT **buf_ptr, int *buf_len, int case_sense, int dbg);
void free_STM_SEG(STM_SEG *seg);
void convert_stm_to_word_list(char *file, char *words, int case_sense, int *num_ref);
void fill_STM_structure(STM *stm, FILE *fp_stm, char *stm_file, int *stm_file_end, int case_sense);
