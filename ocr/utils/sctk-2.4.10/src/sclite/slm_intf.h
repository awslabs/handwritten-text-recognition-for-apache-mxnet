/*
  File: slm_intf.h
  Desc: declare the function prototypes for the SLM interface.

  */

void lookup_lm_word_weight(ARC *arc, void *ptr);

#ifdef WITH_SLM

int tg_lookup(WORD *w, WORD *w_m1, WORD *w_m2, ng_t *ng, fb_info *fb_list, double *prob, int dbg);
void initialize_lm(ng_t *ng, fb_info **fb_list, char *lm_file, int dbg);

#endif
