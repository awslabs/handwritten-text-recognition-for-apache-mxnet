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


/* Return the probability of the (context_length+1)-gram stored in
   sought_ngram */

#include "pc_libs/pc_general.h"
#include "rr_libs/general.h"
#include "ngram.h"
#include "idngram2lm.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

void bo_ng_prob(int context_length,
		id__t *sought_ngram,
		ng_t *ng,
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
  int ncount;
  int contcount;
  int *ng_index;

  int i;
  
  int temp_case;

  double alpha;
  double prob;
  double discounted_ncount;

  /* Initialise variables (unnecessary, but gives warning-free compilation */

  ncount = 0;
  contcount = 0;
  alpha = 0.0;
  discounted_ncount = 0.0;

  ng_index = (int *) rr_malloc((context_length+1)*sizeof(int));

  if (context_length == 0) {

    *p_prob = ng->uni_probs[sought_ngram[0]];
    if (*p_prob<= 0.0 || *p_prob >= 1.0) {
      pc_message(verbosity,1,"Warning : P( %d ) == %g\n", 
		 sought_ngram[0], *p_prob);
    }
  }

  else {

    found_ngram = 0;
    found_context = 0;
    ncount = -1;

    /* Find the appropriate (context-length+1)-gram */

    length_exists = 0;
    still_found = 1;

    while (still_found && (length_exists < (context_length+1))) {
      
      found = 0;

      /* Look for (length_exists+1)-gram */

      if (length_exists == 0) {
	if (return_count(ng->four_byte_counts,
			 ng->count_table[0],
			 ng->marg_counts,
			 ng->marg_counts4,
			 sought_ngram[0]) > 0) {
	  found = 1;
	  ng_index[0] = sought_ngram[0];
	}
      }
      else {

	/* Binary search for right ngram */
	
	ng_begin = 
	  get_full_index(ng->ind[length_exists-1][ng_index[length_exists-1]],
			 ng->ptr_table[length_exists-1],
			 ng->ptr_table_size[length_exists-1],
			 ng_index[length_exists-1]);

	if (length_exists == 1) {
	  if (ng_index[0] < ng->vocab_size) {
	    ng_end = 
	      get_full_index(ng->ind[length_exists-1][ng_index[length_exists-1]+1],
			     ng->ptr_table[length_exists-1],
			     ng->ptr_table_size[length_exists-1],
			     ng_index[length_exists-1]+1)-1;
	  }
	  else {
	    ng_end = ng->num_kgrams[1];
	  }
	}
	else {
	  if (ng_index[length_exists-1] < ng->num_kgrams[length_exists-1]-1) {
	    ng_end = 
	      get_full_index(ng->ind[length_exists-1][ng_index[length_exists-1]+1],
			     ng->ptr_table[length_exists-1],
			     ng->ptr_table_size[length_exists-1],
			     ng_index[length_exists-1]+1)-1;
	  }
	  else {
	    ng_end = ng->num_kgrams[length_exists];
	  }
	}

	while (ng_begin <= ng_end) {
	  ng_middle = ng_begin + ((ng_end - ng_begin) >> 1);
	  if (sought_ngram[length_exists] < 
	      ng->word_id[length_exists][ng_middle]) {
	    ng_end = ng_middle - 1;
	  }
	  else {
	    if (sought_ngram[length_exists] > 
		ng->word_id[length_exists][ng_middle]) {
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

      ncount = return_count(ng->four_byte_counts,
			    ng->count_table[context_length],
			    ng->count[context_length],
			    ng->count4[context_length],
			    ng_index[context_length]);
    }

    if (length_exists >= context_length) {
      found_context = 1;
      if (context_length == 1) {
	contcount = return_count(ng->four_byte_counts,
				 ng->count_table[0],
				 ng->marg_counts,
				 ng->marg_counts4,
				 ng_index[0]);
      }
      else {
	contcount = return_count(ng->four_byte_counts,
				 ng->count_table[context_length-1],
				 ng->count[context_length-1],
				 ng->count4[context_length-1],
				 ng_index[context_length-1]);
      }
    }
    if (found_context) {
      if (ng->four_byte_alphas) {
	alpha = ng->bo_weight4[context_length-1][ng_index[context_length-1]];
      }
      else {
	alpha = double_alpha(ng->bo_weight[context_length-1][ng_index[context_length-1]],
			     ng->alpha_array,
			     ng->size_of_alpha_array,
			     65535 - ng->out_of_range_alphas,
			     ng->min_alpha,
			     ng->max_alpha);
      }
    }

    /* If it occurred then return appropriate prob */

    if (found_ngram) {
      switch (ng->discounting_method) {
      case GOOD_TURING:
	if (ncount <= ng->disc_range[context_length]) {
	  discounted_ncount = 
	    ng->gt_disc_ratio[context_length][ncount]*ncount;
	}
	else {
	  discounted_ncount = ncount;
	}	  
	break;
      case LINEAR:
	discounted_ncount = ng->lin_disc_ratio[context_length]*ncount;
	break;
      case ABSOLUTE:
	discounted_ncount = ncount - ng->abs_disc_const[context_length];
	break;
      case WITTEN_BELL:
	discounted_ncount = ( (double) ((double) contcount*ncount))/
	  ( (double) contcount+ num_of_types(context_length-1,
					     ng_index[context_length-1],ng));
	break;
      }

      prob = discounted_ncount / (double) contcount;
      temp_case = 0;

      if (prob <= 0.0 || prob >= 1.0) {
	pc_message(verbosity,1,"Warning : P(%d) = %g (%g / %d)\n",
		   sought_ngram[0],prob, discounted_ncount,contcount);
	pc_message(verbosity,1,"ncount = %d\n",ncount);
      }

    }
    
    else {

      bo_ng_prob(context_length-1,
		 &sought_ngram[1],
		 ng,
		 verbosity,
		 &prob,
		 bo_case);

      temp_case = 2;

      if (found_context) {
	prob*=alpha;
	temp_case=1;
      }
      
      

    }
    

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
