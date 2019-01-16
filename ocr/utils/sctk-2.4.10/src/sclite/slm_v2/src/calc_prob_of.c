
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

/* Return the probability of word, given a language
   model a context, and a forced backoff list */

#include <stdlib.h>
#include "evallm.h"
#include "idngram2lm.h"

double calc_prob_of(id__t sought_word,
		    id__t *context,
		    int context_length,
		    ng_t *ng,
		    arpa_lm_t *arpa_ng,
		    fb_info *fb_list,
		    int *bo_case,
		    int *acl,
		    flag arpa_lm) {

  int i;
  flag exc_back_off;
  int most_recent_fb;
  int actual_context_length;
  id__t *sought_ngram;
  double prob;

  exc_back_off = 0;

  if (arpa_lm) {
    if (sought_word == 0 && arpa_ng->vocab_type == CLOSED_VOCAB) {
      quit(-1,"Error : Cannot generate probability for <UNK> since this is a closed \nvocabulary model.\n");
    }   
  }
  else {
    if (sought_word == 0 && ng->vocab_type == CLOSED_VOCAB) {
      quit(-1,"Error : Cannot generate probability for <UNK> since this is a closed \nvocabulary model.\n");
    }
  }

  most_recent_fb = -1;
  
  /* Find most recent word in the forced back-off list */
  
  for (i=context_length-1;i>=0;i--) {

    if (fb_list[context[i]].backed_off) {
      most_recent_fb = i;
      if (fb_list[context[i]].inclusive) {
	exc_back_off = 0;
      }
      else {
	exc_back_off = 1;
      }
      i = -2;
    }

  }
  
  actual_context_length = context_length - most_recent_fb -1;

  if (!exc_back_off && most_recent_fb != -1) {
    actual_context_length++;
  }

  sought_ngram = (id__t *) rr_malloc(sizeof(id__t)*(actual_context_length+1));

  for (i=0;i<=actual_context_length-1;i++) {
    if (exc_back_off) {
      sought_ngram[i] = context[i+most_recent_fb+1];
    }
    else {
      if (most_recent_fb == -1) {
	sought_ngram[i] = context[i+most_recent_fb+1];
      }
      else {
	sought_ngram[i] = context[i+most_recent_fb];
      }
    }
  }
  sought_ngram[actual_context_length] = sought_word;


  if (arpa_lm) {
    arpa_bo_ng_prob(actual_context_length,
		    sought_ngram,
		    arpa_ng,
		    2,       /* Verbosity */
		    &prob,
		    bo_case);
  }
  else {
    bo_ng_prob(actual_context_length,
	       sought_ngram,
	       ng,
	       2,       /* Verbosity */
	       &prob,
	       bo_case);
  }

  *acl = actual_context_length;

  free(sought_ngram);
  
  return(prob);

}

