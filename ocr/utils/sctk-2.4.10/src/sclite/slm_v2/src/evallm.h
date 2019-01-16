/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1996, Carnegie Mellon University, Cambridge University,
Ronald Rosenfeld and Philip Clarkson.

All rights reserved.

This software is made available for research purposes only.  It may be
redistributed freely for this purpose, in full or in part, provided
that this entire copyright notice is included on any copies of this
software and applications and derivations thereof.

This software is provided on an "as is" basis, without warranty of any
kind, either expressed or implied, as to any matter including, but not
limited to warranty of fitness of purpose, or merchantability, or
results obtained from use of this software.
======================================================================*/

/* Function prototypes for evallm */

#ifndef _EVALLM_PROTS_
#define _EVALLM_PROTS_

#include "pc_libs/pc_general.h"
#include "rr_libs/general.h"
#include "ngram.h"
#include "toolkit.h"

/* Type specification for forced back-off list */

typedef struct {
  flag backed_off;
  flag inclusive;
} fb_info;

typedef float bo_t;
typedef float prob_t;

/* Type specification for arpa_lm type */

typedef struct {

  unsigned short n;                /* n=3 for trigram, n=4 for 4-gram etc. */

  /* Vocabulary stuff */

  sih_t          *vocab_ht;      /* Vocabulary hash table */
  unsigned short vocab_size;     /* Vocabulary size */
  char           **vocab;        /* Array of vocabulary words */
  flag           *context_cue;   /* True if word with this id is
				    a context cue */
  int            no_of_ccs;      /* The number of context cues in the LM */

  /* Tree */

  table_size_t   *table_sizes;   /* Pointer to table size array */
  id__t          **word_id;      /* Pointer to array of id lists */

  bo_t           **bo_weight;    /* Pointer to array of back-off weights */
  prob_t         **probs;        /* Pointer to array of probabilities */
  index__t       **ind;          /* Pointer to array of index lists */

  /* Index lookup tables */

  ptr_tab_t      **ptr_table;     /* Pointer to the tables used for compact 
				     representation of the indices */
  unsigned short *ptr_table_size; /* Pointer to array of pointer tables */

  /* Misc */

  int            *num_kgrams;     /* Array indicating how many 
				     2-grams, ... ,n-grams, have been 
				     processed so far */

  unsigned short vocab_type;      /* see #define stuff at the top */

  unsigned short first_id;        /* 0 if we have open vocab, 1 if we have
				     a closed vocab. */

} arpa_lm_t;

/* Function prototypes */

unsigned short num_of_types(int k,
			    int ind,
			    ng_t *ng);
void decode_bo_case(int bo_case,
		    int context_length,
		    FILE *annotation_fp);

void display_stats(ng_t *ng);

void display_arpa_stats(arpa_lm_t *arpa_ng);

void load_lm(ng_t *ng,
	     char *lm_filename);

void load_arpa_lm(arpa_lm_t *arpa_lm,
		  char *lm_filename);
		  

void parse_comline(char *input_line,
		  int *num_of_args,
		  char **args);

void compute_perplexity(ng_t *ng,
			arpa_lm_t *arpa_ng,
			char *text_stream_filename,
			char *probs_stream_filename,
			char *annotation_filename,
			char *oov_filename,
			char *fb_list_filename,
			flag backoff_from_unk_inc,
			flag backoff_from_unk_exc,
			flag backoff_from_ccs_inc,
			flag backoff_from_ccs_exc,
			flag arpa_lm,
			flag include_unks,
			double log_base);

fb_info *gen_fb_list(sih_t *vocab_ht,
		     int vocab_size,
		     char **vocab,
		     flag *context_cue,
		     flag backoff_from_unk_inc,
		     flag backoff_from_unk_exc,
		     flag backoff_from_ccs_inc,
		     flag backoff_from_ccs_exc,
		     char *fb_list_filename);

void validate(ng_t *ng,
	      arpa_lm_t *arpa_ng,
	      char **words,
	      flag backoff_from_unk_inc,
	      flag backoff_from_unk_exc,
	      flag backoff_from_ccs_inc,
	      flag backoff_from_ccs_exc,
	      flag arpa_lm,
	      char *fb_list_filename);

double calc_prob_of(id__t sought_word,
		    id__t *context,
		    int context_length,
		    ng_t *ng,
		    arpa_lm_t *arpa_ng,
		    fb_info *fb_list,
		    int *bo_case,
		    int *actual_context_length,
		    flag arpa_lm);

void arpa_bo_ng_prob(int context_length,
		     id__t *sought_ngram,
		     arpa_lm_t *arpa_ng,
		     int verbosity,
		     double *p_prob,
		     int *bo_case);


#endif
