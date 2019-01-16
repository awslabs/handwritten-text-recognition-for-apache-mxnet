/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1997, Carnegie Mellon University, Cambridge University,
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

/* Procedures to deal with counts correctly, taking account of whether
we are dealing with two- or four-byte counts. */

#include "ngram.h"
#include "idngram2lm.h"

void store_count(flag four_byte_counts,
		 int *count_table,
		 int count_table_size,
		 unsigned short *short_counts,
		 int *long_counts,
		 int position,
		 int count) {

  if (four_byte_counts) {
    long_counts[position] = count;
  }
  else {
    short_counts[position] = lookup_index_of(count_table,
					     count_table_size,
					     count);
  }
}

int return_count(flag four_byte_counts,
		 int *count_table,
		 unsigned short *short_counts,
		 int *long_counts,
		 int position) {

  if (four_byte_counts) {
    return(long_counts[position]);
  }
  else {
    return(count_table[short_counts[position]]);
  }

}
		 
