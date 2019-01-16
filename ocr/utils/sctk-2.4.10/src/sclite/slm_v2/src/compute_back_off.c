
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

/* Compute the back off alphas for table n */
/* ie if called with n=2, then compute the bigram alphas */

#include "ngram.h"
#include "idngram2lm.h"
#include "pc_libs/pc_general.h"
#include <stdlib.h>

void compute_back_off(ng_t *ng,int n, int verbosity) {

  int *current_pos;
  int *end_pos;
  id__t *sought_ngram;
  int current_table;
  int ng_count;
  int i;
  double sum_cond_prob;
  double sum_bo_prob;
  double discounted_ngcount;
  double cond_prob;
  double bo_prob;
  double discount_mass;
  double leftout_bo_prob;
  double alpha;

  int bo_case;

  sum_cond_prob = 0.0;
  sum_bo_prob = 0.0;

  /* For the sake of warning-free compilation... */

  discounted_ngcount = 0.0;
  
  current_pos = (int *)rr_calloc(n+1,sizeof(int));
  sought_ngram = (id__t *) rr_calloc(n+1,sizeof(id__t));
  end_pos = (int *)rr_calloc(n+1,sizeof(int)); 
  
  /* Process the tree so that we get all the n-grams out in the right
     order. */
  
  for (current_pos[0]=ng->first_id;
       current_pos[0]<=ng->vocab_size;
       current_pos[0]++) {
    
    if (return_count(ng->four_byte_counts,
		     ng->count_table[0],
		     ng->marg_counts,
		     ng->marg_counts4,
		     current_pos[0]) > 0) {

      current_table = 1;
      
      if (current_pos[0] == ng->vocab_size) {
	end_pos[1] = ng->num_kgrams[1]-1;
      }
      else {
 	end_pos[1] = get_full_index(ng->ind[0][current_pos[0]+1],
				    ng->ptr_table[0],
				    ng->ptr_table_size[0],
				    current_pos[0]+1)-1;
      }

      while (current_table > 0) {

	if (current_table == n) {

	  if (current_pos[n] <= end_pos[n]){

	    ng_count = return_count(ng->four_byte_counts,
				    ng->count_table[n],
				    ng->count[n],
				    ng->count4[n],
				    current_pos[n]);

	    switch (ng->discounting_method) {
	    case GOOD_TURING:
	      if (ng_count <= ng->disc_range[n]) {
		discounted_ngcount = ng->gt_disc_ratio[n][ng_count] * ng_count;
	      }
	      else {
		discounted_ngcount = ng_count;
	      }
	      break;
	    case LINEAR:
	      discounted_ngcount = ng->lin_disc_ratio[n] * ng_count;
	      break;
	    case ABSOLUTE:
	      discounted_ngcount = ng_count - ng->abs_disc_const[n];
	      break;
	    case WITTEN_BELL:
	      if (n==1) {

		discounted_ngcount = ((double) 
				      return_count(ng->four_byte_counts,
						   ng->count_table[0],
						   ng->marg_counts,
						   ng->marg_counts4,
						   current_pos[0]) * ng_count)
		  / (return_count(ng->four_byte_counts,
				  ng->count_table[0],
				  ng->marg_counts,
				  ng->marg_counts4,
				  current_pos[0]) + 
		     num_of_types(0,current_pos[0],ng));
	      }
	      else {
		
		discounted_ngcount = ((double) 
				      return_count(ng->four_byte_counts,
						   ng->count_table[n-1],
						   ng->count[n-1],
						   ng->count4[n-1],
						   current_pos[n-1])* ng_count)
		  / (return_count(ng->four_byte_counts,
				  ng->count_table[n-1],
				  ng->count[n-1],
				  ng->count4[n-1],
				  current_pos[n-1]) + 
		     num_of_types(n-1,current_pos[n-1],ng));

	      }	  
	      
	      break;
	    }

	    if (n==1) {
	      cond_prob = ((double) discounted_ngcount / 
			   return_count(ng->four_byte_counts,
					ng->count_table[0],
					ng->marg_counts,
					ng->marg_counts4,
					current_pos[0]));
	    }
	    else {
	      cond_prob = ((double) discounted_ngcount /  
			   return_count(ng->four_byte_counts,
					ng->count_table[n-1],
					ng->count[n-1],
					ng->count4[n-1],
					current_pos[n-1]));

	    }
	    sum_cond_prob += cond_prob;

	    /* Fill up sought ngram array with correct stuff */

	    for (i=1;i<=n;i++) {
	      sought_ngram[i-1] = ng->word_id[i][current_pos[i]];
	    }


	    bo_ng_prob(n-1,sought_ngram,ng,verbosity,&bo_prob,&bo_case);
	    sum_bo_prob += bo_prob;
	    current_pos[n]++;			
					       
	  }
	  else {

	    discount_mass = 1.0 - sum_cond_prob;

	    if (discount_mass < 1e-10) {
	      discount_mass = 0.0;
	      pc_message(verbosity,2,"Warning : Back off weight for %s(id %d) ",
			 ng->vocab[current_pos[0]],current_pos[0]);
	      for (i=1;i<=n-1;i++) {
		pc_message(verbosity,2,"%s(id %d) ",ng->vocab[ng->word_id[i][current_pos[i]]],ng->word_id[i][current_pos[i]]);
	      }
	      pc_message(verbosity,2,
			 "is set to 0 (sum of probs = %f).\nMay cause problems with zero probabilities.\n",sum_cond_prob);
	    }

	    leftout_bo_prob = 1.0 - sum_bo_prob;
	    if (leftout_bo_prob < 1e-10) {
	      leftout_bo_prob = 0.0;
	    }

	    if (leftout_bo_prob > 0.0) {
	      alpha = discount_mass / leftout_bo_prob;
	    }
	    else {
	      alpha = 0.0;	/* Will not be used. Should happen very rarely. */
	      pc_message(verbosity,2,"Warning : Back off weight for %s(id %d) ",
			 ng->vocab[current_pos[0]],current_pos[0]);
	      for (i=1;i<=n-1;i++) {
		pc_message(verbosity,2,"%s(id %d) ",ng->vocab[ng->word_id[i][current_pos[i]]],ng->word_id[i][current_pos[i]]);
	      }
	      pc_message(verbosity,2,
			 "is set to 0.\nMay cause problems with zero probabilities.\n");

	    }
	  
	    if (ng->four_byte_alphas) {
	      ng->bo_weight4[n-1][current_pos[n-1]] = alpha;
	    }
	    else {
	      ng->bo_weight[n-1][current_pos[n-1]] = 
		short_alpha(alpha,
			    ng->alpha_array,
			    &(ng->size_of_alpha_array),
			    65535 - ng->out_of_range_alphas,
			    ng->min_alpha,
			    ng->max_alpha);
	    }
	  
	    /* Finished current (n-1)-gram */

	    sum_cond_prob = 0.0;
	    sum_bo_prob = 0.0;
	    current_table--;
	    if (current_table > 0) {
	      current_pos[current_table]++;
	    }
	  }
	}
	else {

	  if (current_pos[current_table] <= end_pos[current_table]) {
	    current_table++;
	    if (current_pos[current_table-1] == ng->num_kgrams[current_table-1]-1) {
	      end_pos[current_table] = ng->num_kgrams[current_table]-1;
	    }
	    else {
	      end_pos[current_table] = get_full_index(ng->ind[current_table-1][current_pos[current_table-1]+1],ng->ptr_table[current_table-1],ng->ptr_table_size[current_table-1],current_pos[current_table-1]+1)-1;
	    }
	  }
	  else {
	    current_table--;
	    if (current_table > 0) {
	      current_pos[current_table]++;
	    }
	  }
	}
      }
    }

    /* Now deal with zeroton unigrams */

    else {
      if (n == 1) {
	if (ng->four_byte_alphas) {
	  ng->bo_weight4[0][current_pos[0]] = 1.0;
	}
	else {
	  ng->bo_weight[0][current_pos[0]] = 
	    short_alpha(1.0,
			ng->alpha_array,
			&(ng->size_of_alpha_array),
			65535 - ng->out_of_range_alphas,
			ng->min_alpha,
			ng->max_alpha);
	}
      }
    }
  }
  free(end_pos);
  free(current_pos);
  free(sought_ngram);
  
}







