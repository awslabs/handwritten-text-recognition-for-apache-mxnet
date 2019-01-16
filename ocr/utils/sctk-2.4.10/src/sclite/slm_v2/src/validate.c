
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

#include "evallm.h"
#include <stdlib.h>

void validate(ng_t *ng,
	      arpa_lm_t *arpa_ng,
	      char **words,
	      flag backoff_from_unk_inc,
	      flag backoff_from_unk_exc,
	      flag backoff_from_ccs_inc,
	      flag backoff_from_ccs_exc,
	      flag arpa_lm,
	      char *fb_list_filename) {


  int *context;
  id__t *short_context;
  int dummy1;
  int dummy2;
  int i;
  fb_info *fb_list;
  double prob_so_far;
  flag found_unk_wrongly;
  int n;

  if (arpa_lm) {
    n = arpa_ng->n;
  }
  else {
    n = ng->n;
  }

  if (arpa_lm) {
    fb_list = gen_fb_list(arpa_ng->vocab_ht,
			  arpa_ng->vocab_size,
			  arpa_ng->vocab,
			  arpa_ng->context_cue,
			  backoff_from_unk_inc,
			  backoff_from_unk_exc,
			  backoff_from_ccs_inc,
			  backoff_from_ccs_exc,
			  fb_list_filename);
  }
  else {
    fb_list = gen_fb_list(ng->vocab_ht,
			  ng->vocab_size,
			  ng->vocab,
			  ng->context_cue,
			  backoff_from_unk_inc,
			  backoff_from_unk_exc,
			  backoff_from_ccs_inc,
			  backoff_from_ccs_exc,
			  fb_list_filename);
  }
  
  context = (int *) rr_malloc(sizeof(int)*(n-1));
  short_context = (id__t *) rr_malloc(sizeof(id__t)*(n-1));
  
  found_unk_wrongly = 0;

  for (i=0;i<=n-2;i++) {
    if (arpa_lm) {
      if (sih_lookup(arpa_ng->vocab_ht,words[i],&context[i]) == 0) {
	if (arpa_ng->vocab_type == CLOSED_VOCAB) {
	  fprintf(stderr,"Error : %s is not in the vocabulary, and this is a closed \nvocabulary model.\n",words[i]);
	  found_unk_wrongly = 1;
	}
	else {
	  fprintf(stderr,"Warning : %s is an unknown word.\n",words[i]);
	}
      }
      if (context[i] > 65535) {
	quit(-1,"Error : returned value from sih_lookup is too high.\n");
      }
      else {
	short_context[i] = context[i];
      }
    }
    else {
      if (sih_lookup(ng->vocab_ht,words[i],&context[i]) == 0) {
	if (ng->vocab_type == CLOSED_VOCAB) {
	  fprintf(stderr,"Error : %s is not in the vocabulary, and this is a closed \nvocabulary model.\n",words[i]);
	  found_unk_wrongly = 1;
	}
	else {
	  fprintf(stderr,"Warning : %s is an unknown word.\n",words[i]);
	}
      }
      if (context[i] > 65535) {
	quit(-1,"Error : returned value from sih_lookup is too high.\n");
      }
      else {
	short_context[i] = context[i];
      }
    }
  }

  /* Map down from context array to short_context array */
  /* sih_lookup requires the array to be ints, but prob_so_far
     requires short ints. */

  if (!found_unk_wrongly) {

    prob_so_far = 0.0;
    
    if (arpa_lm) {
      for (i=arpa_ng->first_id;i<=arpa_ng->vocab_size;i++) {
	prob_so_far += calc_prob_of(i,
				    short_context,
				    n-1,
				    ng,
				    arpa_ng,
				    fb_list,
				    &dummy1,
				    &dummy2,
				    arpa_lm);
      }

    }
    else {
      for (i=ng->first_id;i<=ng->vocab_size;i++) {
	prob_so_far += calc_prob_of(i,
				    short_context,
				    n-1,
				    ng,
				    arpa_ng,
				    fb_list,
				    &dummy1,
				    &dummy2,
				    arpa_lm);
      }
    }
    
    printf("Sum of P( * | ");
    for (i=0;i<=n-2;i++) {
      printf("%s ",words[i]);
    }
    printf(") = %f\n",prob_so_far);
    
  }

  free(context);
  free(fb_list);

}
