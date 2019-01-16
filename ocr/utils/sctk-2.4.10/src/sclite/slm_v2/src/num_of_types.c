
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


#include "idngram2lm.h"

unsigned short num_of_types(int k,
			    int ind,
			    ng_t *ng) {
  
  int start;
  int end;

  start = get_full_index(ng->ind[k][ind],
			 ng->ptr_table[k],
			 ng->ptr_table_size[k],
			 ind);

  if (k>0) {

    if (ind < (ng->num_kgrams[k]-1)) {
      
      end = get_full_index(ng->ind[k][ind+1],
			   ng->ptr_table[k],
			   ng->ptr_table_size[k],
			   ind+1);
    }
  
    else {
      
      end = ng->num_kgrams[k+1];
      
    }

  }
  else {
    if (ind < ng->vocab_size) {
      end = get_full_index(ng->ind[k][ind+1],
			   ng->ptr_table[k],
			   ng->ptr_table_size[k],
			   ind+1);
    }
  
    else {
      
      end = ng->num_kgrams[k+1];
      
    }
  }

  return(end-start);

}
