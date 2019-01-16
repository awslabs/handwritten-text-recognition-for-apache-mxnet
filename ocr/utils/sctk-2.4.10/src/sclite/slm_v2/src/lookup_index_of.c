/* lookup_index_of() */
/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1994, Carnegie Mellon University and Ronald Rosenfeld.
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

/* return an index to be used for lookup of supplied value.
   If value hasn't been entered in the table yet, add it.
   Note that we first try to assign the identity mapping.
   If that's occupied, we next search from the end, till the first empty slot.
*/

/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */

#include "ngram.h"

int lookup_index_of(lookup_table, lookup_table_size, intval)
int *lookup_table, lookup_table_size, intval;
{
  int i;
  if (intval>0 && intval<lookup_table_size) {
     if (lookup_table[intval]==intval) return(intval);
     else if (lookup_table[intval]==0) {
	lookup_table[intval] = intval;
	return(intval);
     }
  }
  for (i=lookup_table_size-1; i>=0; i--) {
     if (lookup_table[i]==intval) return(i);
     if (lookup_table[i]==0) {
	lookup_table[i] = intval;
	return(i);
     }
  }
  quit(-1,"Error - more than %d entries required in the count table. \nCannot store counts in two bytes. Use the -four_byte_counts flag.\n",lookup_table_size);
  
  /* Clearly we will never get this far in the code, but the compiler
     doesn't realise this, so to stop it spewing out warnings... */

  return(0);

}
