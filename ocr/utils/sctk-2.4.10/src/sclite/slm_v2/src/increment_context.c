
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

#include "ngram.h"
#include "idngram2lm.h"
#include <stdlib.h>

void increment_context(ng_t *ng,int k,int verbosity) {
 
  
  int current_count;
  int j;
  int current_table;
  int *current_pos;
  int *end_pos;
  
  flag discounted;



  /* Scan all the (k+1)-grams (i.e. those in table k). If any of them
     are followed by only one (k+2)-gram, and its count is bigger
     than the discounting range, then increment the count of the
     (k+1)-gram. Technique first introduced by Doug Paul. */

  current_pos = (int *)rr_calloc(k+1,sizeof(int));
  end_pos = (int *)rr_calloc(k+1,sizeof(int)); 

  current_count = 0;
  discounted = 0;

  
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
	
	if (current_table == k) {
	  
	  if (current_pos[k] <= end_pos[k]) {

	    current_count += return_count(ng->four_byte_counts,
					  ng->count_table[k],
					  ng->count[k],
					  ng->count4[k],
					  current_pos[k]);

	    if (return_count(ng->four_byte_counts,
			     ng->count_table[k],
			     ng->count[k],
			     ng->count4[k],
			     current_pos[k]) <= ng->disc_range[k]) {
	      discounted = 1;
	    }
	    current_pos[k]++;
	  }

	  else {

	    if (k == 1) {
	      if (current_count >= return_count(ng->four_byte_counts,
						ng->count_table[0],
						ng->marg_counts,
						ng->marg_counts4,
						current_pos[k-1]) 
		  && !discounted) {
		
		store_count(ng->four_byte_counts,
			    ng->count_table[0],
			    ng->count_table_size,
			    ng->marg_counts,
			    ng->marg_counts4,
			    current_pos[0],
			    return_count(ng->four_byte_counts,
					 ng->count_table[0],
					 ng->marg_counts,
					 ng->marg_counts4,
					 current_pos[0])+1); 


	      }
	    }
	    else {
	      if ((current_count >= return_count(ng->four_byte_counts,
						 ng->count_table[k-1],
						 ng->count[k-1],
						 ng->count4[k-1],
						 current_pos[k-1])) && 
		  !discounted) {

		for (j=1;j<=k-1;j++) {
		  store_count(ng->four_byte_counts,
			      ng->count_table[j],
			      ng->count_table_size,
			      ng->count[j],
			      ng->count4[j],
			      current_pos[j],
			      return_count(ng->four_byte_counts,
					   ng->count_table[j],
					   ng->count[j],
					   ng->count4[j],
					   current_pos[j])+1);
		}

		store_count(ng->four_byte_counts,
			    ng->count_table[0],
			    ng->count_table_size,
			    ng->marg_counts,
			    ng->marg_counts4,
			    current_pos[0],
			    return_count(ng->four_byte_counts,
					 ng->count_table[0],
					 ng->marg_counts,
					 ng->marg_counts4,
					 current_pos[0])+1);

	      }
	    }
	    current_count = 0;
	    discounted = 0;
	    current_table--;
	    if (current_table > 0) {
	      current_pos[current_table]++;
	    }
	  }
	}
	else {
	  if (current_pos[current_table] <= end_pos[current_table]) {
	    current_table++;
	    if (current_pos[current_table-1] == 
		ng->num_kgrams[current_table-1]-1) {
	      end_pos[current_table] = ng->num_kgrams[current_table]-1;
	    }
	    else {
	      end_pos[current_table] = 
		get_full_index(ng->ind[current_table-1][current_pos[current_table-1]+1],
			       ng->ptr_table[current_table-1],
			       ng->ptr_table_size[current_table-1],
			       current_pos[current_table-1]+1)-1;
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
  } 

  free(current_pos);
  free(end_pos);

}

