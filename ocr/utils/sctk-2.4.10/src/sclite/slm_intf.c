#include "sctk.h"


/* 
   File: slm_intf.c
   Date: March 20, 1998
   

   This file contains interface functions to lookup probs from a language
   model read in by the SLM toolkit libraries.
   
   The only function that is always defined is lookup_lm_word_weight.  If the
   SLM is not included, then executing this function is an error. 
   
   */


void lookup_lm_word_weight(ARC *arc, void *ptr){

#ifdef WITH_SLM
  WORD *tw = (WORD *)(arc->data);
  static init = 1;
  static ng_t ng;
  static fb_info *fb_list;
  WORD *w = (WORD *)arc->data, *w_m1, *w_m2;
  double sum_weight, lu_prob, def_w;
  int nprob;
  ARC_LIST_ATOM *oarc_a, *oarc_b;
  int dbg = 0;

  if (init){
    initialize_lm(&ng,&fb_list, (char *)ptr, dbg);    
    init = 0;
  }    

#define W_func(_d) (log(1.0 / (_d)) * 1.4426950408889634074)

  /* Set the default weight for this word */
  def_w = (w->opt_del) ? 0.0 : 20.0;

  nprob = 0, sum_weight = 0;

  if (dbg) printf("w->value= %s\n",w->value);
  /* search for the previous word */
  if (! arc->from_node->start_state && ng.n > 1 && ! w->opt_del){
    for (oarc_a = arc->from_node->in_arcs; oarc_a != NULL; oarc_a=oarc_a->next){
      w_m1 = (WORD*)(oarc_a->arc->data);
      
      if (! oarc_a->arc->from_node->start_state && ng.n > 2){
	for (oarc_b = oarc_a->arc->from_node->in_arcs; oarc_b != NULL; oarc_b=oarc_b->next){
	  w_m2 = (WORD*)(oarc_b->arc->data);
	  if (dbg) printf("      w-1= %s w-2= %s\n",w_m1->value,w_m2->value);
	  sum_weight += 
	    (tg_lookup(w, w_m1, w_m2, &ng, fb_list, &lu_prob, dbg)) ? W_func(lu_prob) : def_w;
	  nprob ++;
	}
      } else {
	w_m2 = (WORD *)0;
	if (dbg) printf("      w-1= %s w-2= No-context\n",w_m1->value);
	sum_weight += 
	  (tg_lookup(w, w_m1, w_m2, &ng, fb_list, &lu_prob, dbg)) ? W_func(lu_prob) : def_w;
	nprob ++;
      }
    }
  } else {
    w_m1 = w_m2 = (WORD *)0;
    if (dbg) printf("      w-1= No-context w-2= No-context\n");
    if (! w->opt_del)
      sum_weight += 
	(tg_lookup(w, w_m1, w_m2, &ng, fb_list, &lu_prob, dbg)) ? W_func(lu_prob) : def_w;
    nprob ++;
  }
  if (dbg) printf("     AvgWeight=   \t%g\n",sum_weight/nprob);
  tw->weight = sum_weight / nprob;
#else
  fprintf(scfp,"Error: a call to lookup_lm_word_weight() was made even though\n"
	  "       the SLM toolkit has not been included in the compilation of SCTK\n");
  exit(1);
#endif
}

#ifdef WITH_SLM

/*****************************************************************/
/* given upto three WORDS, lookup the N-gram in a language model */

int tg_lookup(WORD *w, WORD *w_m1, WORD *w_m2, ng_t *ng, fb_info *fb_list, double *prob, int dbg){
    int32 h_lu, m1_lu, m2_lu;
    id__t h_id, m_id[3];
    int bo_case = 0, acl = 0;
    int actual_context;

    if (sih_lookup(ng->vocab_ht,
		   (char *)((w->opt_del)?w->intern_value:w->value),&h_lu) == 0){
      if (dbg) printf("   Head Word Not in Vocab\n");
      *prob = 0.0;
      return(0);
    }
    h_id = (id__t)h_lu;

    if (w_m1 != (WORD *)0){
      sih_lookup(ng->vocab_ht,
		 (char *)((w_m1->opt_del)?w_m1->intern_value:w_m1->value),&m1_lu);
      actual_context = 1;
      m_id[0] = (id__t)m1_lu;

      if (w_m2 != (WORD *)0){
	sih_lookup(ng->vocab_ht,
		   (char *)((w_m2->opt_del)?w_m2->intern_value:w_m2->value),&m2_lu);
	actual_context = 2;
	m_id[0] = (id__t)m2_lu;
	m_id[1] = (id__t)m1_lu;
      }
    } else {
      actual_context = 0;
    }

    *prob = calc_prob_of(h_id,m_id,actual_context,ng,(arpa_lm_t *)0,
			fb_list,&bo_case,&acl,0);

    if (dbg){
      printf("           Texts: P(%s | ",(char *)w->value);
      if (ng->n > 2) printf("%s ",(w_m2 == (WORD *)0) ? "UNK" : (char *)w_m2->value);
      if (ng->n > 1) printf("%s",(w_m1 == (WORD *)0) ? "UNK" : (char *)w_m1->value);
      printf(") = %g  bo_case ",*prob);
      decode_bo_case(bo_case,acl,stdout);
    }
    return(1);
}

/*****************************************************************/
/* Read in a lanagauge model, an set up the structures.          */

void initialize_lm(ng_t *ng, fb_info **fb_list, char *lm_file, int dbg){
  
  if (lm_file != (char *)0){
    if (dbg)   printf("Loading LM file '%s'\n",lm_file);
    load_lm(ng,lm_file);       /* if it failes, it dies */
    if (dbg) {
      printf("LM read in -\n");
      printf(" version: %d\n",ng->version);
      printf(" #-gram order: %d\n",ng->n);
      printf(" vocabulary size: %d\n",ng->vocab_size);
    }
    *fb_list = gen_fb_list(ng->vocab_ht,
			  ng->vocab_size,
			  ng->vocab,
			  ng->context_cue,
			  0, 0, 0, 0,
			  "");
    
    /* This code works as a test, except the printout for bo_case is wrong for  
      the trigram.  it output 3x2x1 instead of 3x2-1 */
    
    if (dbg){ 
      int32 lu1, lu2, lu3; id__t h_id, m_id[4]; double prob;
      int bo_case = 0, acl = 0;
      
      if (sih_lookup(ng->vocab_ht,"the",&lu1) == 0)
	printf("   Head Word Not in Vocab\n");
      m_id[0] = lu1;
      h_id = lu1;
      prob = calc_prob_of(h_id,m_id,0,ng,(arpa_lm_t *)0,*fb_list,&bo_case,&acl,0);
      printf("     P(%s| ) = %g   bo_case ",ng->vocab[lu1],prob);
      decode_bo_case(bo_case,acl,stdout);

      if (ng->n > 1) {
	if (sih_lookup(ng->vocab_ht,"dow",&lu2) == 0) 
	  printf("   Head Word Not in Vocab\n");
	m_id[0] = lu1;
	h_id = lu2;
	prob = calc_prob_of(h_id,m_id,1,ng,(arpa_lm_t *)0,*fb_list,&bo_case,&acl,0);
	printf("     P(%s | the) = %g   bo_case ",ng->vocab[lu2],prob);
	decode_bo_case(bo_case,acl,stdout); 
	
	if (ng->n > 2) {
	  if (sih_lookup(ng->vocab_ht,"jones",&lu3) == 0)
	    printf("   Head Word Not in Vocab\n");
	  m_id[0] = lu1;
	  m_id[1] = lu2;
	  h_id = lu3;
	  prob = calc_prob_of(h_id,m_id,2,ng,(arpa_lm_t *)0,*fb_list,&bo_case,&acl,0);
	  printf("     P(jones | the dow) = %g   bo_case ",prob);
	  decode_bo_case(bo_case,acl,stdout);
	}
      }
    }
    
    
  } else {
    fprintf(stderr,"Error: no lm_file name passed to alignment\n");
  }
}

#endif
