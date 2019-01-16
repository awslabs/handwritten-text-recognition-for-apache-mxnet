
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

/* Rewritten 6 May 1997. No longer try to guess based on filesize and
cutoffs. Merely allocate an equal amount of memory to each of the
tables for 2,3,...,n-grams. Total memory allocation should equal
STD_MEM. This is far from optimal, but is at least robust. */

#include <stdio.h>
#include "toolkit.h"
#include "ngram.h"
#include "rr_libs/general.h"
#include "pc_libs/pc_general.h"

void guess_mem(int total_mem,
	       int middle_size,   /* Size of 2,3,(n-1)-gram records */
	       int end_size,      /* Size of n-gram record */
	       int n,
	       table_size_t *table_sizes,
	       int verbosity) {



  int *num_kgrams;
  int i;

  num_kgrams = (int *) rr_malloc(sizeof(int)*(n-1));
  
  /* First decide on file size (making allowances for compressed files) */

  for (i=0;i<=n-3;i++) {

    num_kgrams[i] = total_mem*1000000/(middle_size*(n-1));

  }

  num_kgrams[n-2] = total_mem*1000000/(end_size*(n-1));

  for (i=0;i<=n-2;i++) {
    table_sizes[i+1] = num_kgrams[i];
    pc_message(verbosity,2,"Allocated space for %d %d-grams.\n",
	       table_sizes[i+1],i+2);
  }
  
  

}


