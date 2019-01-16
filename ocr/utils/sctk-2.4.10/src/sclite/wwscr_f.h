/*************************************************************************/
/*   The word weighting scoring program                                  */

#define MAX_W 10

typedef struct word_weight_struct{
    double weight[MAX_W];
    TEXT *str;
} WW;

typedef struct word_weight_list_struct{
    TEXT *filename;
    TEXT *weight_desc[MAX_W];
    int num_w;
    double default_weight;
    int max;
    int num;
    int curw;
    WW **words;
} WWL;

typedef struct WWL_spkr_struct{
    double ref, corr, sub, del, ins, spl, mrg;
    TEXT *id;
} WWL_SPKR;

typedef struct WWL_FUNC_struct{
    double ref, corr, sub, del, ins, spl, mrg;
    int n_spkrs;
    WWL_SPKR *spkr;
    TEXT *title;
} WWL_FUNC;

typedef struct WWL_score_struct{
    int numfunc;
    int maxfunc;
    WWL_FUNC *func;
} WWL_SCORE;

double Weight_wwl (TEXT *str, WWL *wwl);
double Weight_one (TEXT *str, WWL *wwl);
int load_WWL (WWL **wwl, TEXT *filename);
void dump_WWL (WWL *wwl, FILE *fp);
void free_WWL (WWL **wwl);

int perform_word_weighted_scoring(SCORES *sc, TEXT *sys_root_name, int do_weight_one, int n_wwlf, TEXT **wwl_files, int make_sum, int make_ovrall, int dbg, int feedback);

