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


/* Function to calculate the memory required for each of the count
   tables, given a path to an id_ngram file, and a pointer to an array
   of cutoffs */



#include <stdio.h>
#include "rr_libs/general.h"
#include "ngram.h"
#include "idngram2lm.h"

void calc_mem_req(ng_t *ng,flag is_ascii) {

  ngram current_ngram;
  ngram previous_ngram;
  count_t *ng_count;
  int i,j;

  current_ngram.id_array = (id__t *) rr_malloc(sizeof(id__t)*ng->n);
  previous_ngram.id_array = (id__t *) rr_malloc(sizeof(id__t)*ng->n);
  
  ng_count = (count_t *) rr_calloc(ng->n,sizeof(count_t));

  current_ngram.n = ng->n;

  rewind(ng->id_gram_fp);

  while (!rr_feof(ng->id_gram_fp)) {
    for (i=0;i<=ng->n-1;i++) {
      previous_ngram.id_array[i]=current_ngram.id_array[i];
    }
    get_ngram(ng->id_gram_fp,&current_ngram,is_ascii);
    for (i=0;i<=ng->n-1;i++) {
      if (current_ngram.id_array[i] != previous_ngram.id_array[i]) {
	for (j=i;j<=ng->n-1;j++) {
	  if (j>0) {
	    if (ng_count[j] > ng->cutoffs[j-1]) {
	      ng->table_sizes[j]++;
	    }
	  }
	  ng_count[j] =  current_ngram.count;
	}
	i=ng->n;
      }
      else {
	ng_count[i] += current_ngram.count;
      }
    }
  }

  for (i=1;i<=ng->n-1;i++) {
    if (ng_count[i] > ng->cutoffs[i-1]) {
      ng->table_sizes[i]++;
    }
  }

  /* Add a fudge factor, as problems can crop up with having to
     cut-off last few n-grams. */

  for (i=1;i<=ng->n-1;i++) {
    ng->table_sizes[i]+=10;
  }

  rr_iclose(ng->id_gram_fp);
  ng->id_gram_fp = rr_iopen(ng->id_gram_filename);

}
   






