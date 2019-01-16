

typedef struct CORRESP_STRUCT{
  int use;
  WORD *words[10];
  double sum_conf;
  double max_conf;
  double min_conf;
  double avg_conf;
  double avg_time;
  double avg_dur;
  int nword;
  double bias_nword;
} CORES;

void print_CORES(void *p);
int equal_CORES(void *p1, void *p2);
void *get_CORES(void);
void release_CORES(void *p);
int null_alt_CORES(void *p);
int opt_del_CORES (void *p);
void *copy_CORES(void *p);
void *make_empty_CORES(void *p);
int use_count_CORES(void *p, int i);
NETWORK *Network_WORD_to_CORES(NETWORK *net);






