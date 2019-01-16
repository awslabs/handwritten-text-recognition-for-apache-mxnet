
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

/* Procedures to allow the indices which point into the next table to
   be stored as two-byte integers, rather than four byte integers. 

   All that is stored of the indices are two bytes corresponding to
   the offset. For each table, we also store an array indicating where
   the non-offset part increases (which we can do, since the indices
   will be monotonically increasing).
   
  This is probably best illustrated with an example. Suppose our array
  contains the following:

  ind_table[0]=0
  ind_table[1]=30
  ind_table[2]=73
  
  and the stored indices are called index[0..100]. Then the actual
  indices are

  actual_index[0..29] = (0*key) + index[0..29]
  actual_index[30..72] = (1*key) + index[30..72]
  actual_index[73..100] = (2*key) + index[73..100]

*/

#include "rr_libs/general.h"
#include "ngram.h"

unsigned short new_index(int full_index,
			 int *ind_table,
			 unsigned short *ind_table_size,
			 int position_in_list) {

  
  if (full_index - (((*ind_table_size)-1)*KEY) >= KEY) {
    ind_table[*ind_table_size] = position_in_list;
    (*ind_table_size)++;
  }

  return (full_index % KEY);

}

int get_full_index(unsigned short short_index,
		   int *ind_table,
		   unsigned short ind_table_size,
		   int position_in_list) {

  int lower_bound;
  int upper_bound;
  int new_try;

  /* Binary search for position_in_list */

  lower_bound = 0;
  upper_bound = ind_table_size-1;
  
  /* Search range is inclusive */

  if (ind_table_size > 0) {
  
    if (position_in_list >= ind_table[upper_bound]) {
      lower_bound = upper_bound;
    }

    while (upper_bound-lower_bound > 1) {
      new_try = (upper_bound + lower_bound) / 2;
      if (ind_table[new_try] > position_in_list) {
	upper_bound = new_try;
      }
      else {
	lower_bound = new_try;
      }
    }
  }

  /* Return the appropriate value */


  return((lower_bound*KEY)+short_index);

}

