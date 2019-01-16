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
#include "idngram2lm.h"
#include <stdlib.h>
#include <math.h>

void arpa_bo_ng_prob(int context_length,
		     id__t *sought_ngram,
		     arpa_lm_t *arpa_ng,
		     int verbosity,
		     double *p_prob,
		     int *bo_case) {

  
  flag found;
  flag found_ngram;
  flag found_context;
  flag still_found;

  int length_exists;
  int ng_begin;
  int ng_end;
  int ng_middle;
  int *ng_index;

  int i;
  
  int temp_case;

  double alpha;
  double prob;

  alpha = 0.0; /* To give no warnings at compilation time */

  ng_index = (int *) rr_malloc((context_length+1)*sizeof(int));

  if (context_length == 0) {
    *p_prob = pow(10.0,arpa_ng->probs[0][sought_ngram[0]]);
  }
  else {

    found_ngram = 0;
    found_context = 0;

    /* Find the appropriate (context-length+1)-gram */

    length_exists = 0;
    still_found = 1;

    while (still_found && (length_exists < (context_length+1))) {
      
      found = 0;

      /* Look for (length_exists+1)-gram */

      if (length_exists == 0) {

	if (get_full_index(arpa_ng->ind[0][sought_ngram[0]],
			   arpa_ng->ptr_table[0],
			   arpa_ng->ptr_table_size[0],
			   sought_ngram[0]) <
	    get_full_index(arpa_ng->ind[0][sought_ngram[0]+1],
			   arpa_ng->ptr_table[0],
			   arpa_ng->ptr_table_size[0],
			   sought_ngram[0]+1)) {
	   found = 1;
	  ng_index[0] = sought_ngram[0];
	}
      }
      else {

	/* Binary search for right ngram */

	ng_begin = 
	  get_full_index(arpa_ng->ind[length_exists-1][ng_index[length_exists-1]],
			 arpa_ng->ptr_table[length_exists-1],
			 arpa_ng->ptr_table_size[length_exists-1],
			 ng_index[length_exists-1]);

	if (length_exists == 1) {
	  if (ng_index[0] < arpa_ng->vocab_size) {
	    ng_end = 
	      get_full_index(arpa_ng->ind[length_exists-1][ng_index[length_exists-1]+1],
			     arpa_ng->ptr_table[length_exists-1],
			     arpa_ng->ptr_table_size[length_exists-1],
			     ng_index[length_exists-1]+1)-1;
	  }
	  else {
	    ng_end = arpa_ng->num_kgrams[1];
	  }
	}
	else {
	  if (ng_index[length_exists-1] < 
	      arpa_ng->num_kgrams[length_exists-1]-1) {
	    ng_end = 
	      get_full_index(arpa_ng->ind[length_exists-1][ng_index[length_exists-1]+1],
			     arpa_ng->ptr_table[length_exists-1],
			     arpa_ng->ptr_table_size[length_exists-1],
			     ng_index[length_exists-1]+1)-1;
	  }
	  else {
	    ng_end = arpa_ng->num_kgrams[length_exists];
	  }
	} 

	while (ng_begin <= ng_end) {
	  ng_middle = ng_begin + ((ng_end - ng_begin) >> 1);
	  if (sought_ngram[length_exists] < 
	      arpa_ng->word_id[length_exists][ng_middle]) {
	    ng_end = ng_middle - 1;
	  }
	  else {
	    if (sought_ngram[length_exists] > 
		arpa_ng->word_id[length_exists][ng_middle]) {
	      ng_begin = ng_middle + 1;
	    }
	    else {
	      found = 1;
	      ng_index[length_exists] = ng_middle;
	      break;
	    }
	  }
	}
      }

      if (found) {
	length_exists++;
      }
      else {
	still_found = 0;
      }

    }
    if (length_exists == (context_length+1)) {
      found_ngram = 1;
    }
    if (length_exists >= context_length) {
      found_context = 1;
    }
    if (found_context) {
      alpha = pow(10.0,arpa_ng->bo_weight[context_length-1][ng_index[context_length-1]]);
    }

    if (found_ngram) {
      prob = pow(10.0,arpa_ng->probs[context_length][ng_index[context_length]]);
      temp_case = 0;
    }
    else {
      arpa_bo_ng_prob(context_length-1,
		      &sought_ngram[1],
		      arpa_ng,
		      verbosity,
		      &prob,
		      bo_case);

      temp_case = 2;      
      if (found_context) {
	prob*=alpha;
	temp_case=1;
      }
      
      

    }

    /*
     * PWP: coding change.  Numbers were previously coded base-3.
     * Now base-4, since (int) pow(4,i) == 1 << (2*i)
     */
    
    *p_prob = prob;
    *bo_case += temp_case * (1 << (2*(context_length-1)));
  

  }

  if (*p_prob > 1.0) {
    fprintf(stderr,"Error : P( %d | ",sought_ngram[context_length]);
    for (i=0;i<=context_length-1;i++) {
      fprintf(stderr,"%d ",sought_ngram[i]);
    }
    fprintf(stderr,") = %g\n",*p_prob);
    exit(1);
  }
  free(ng_index);

}
