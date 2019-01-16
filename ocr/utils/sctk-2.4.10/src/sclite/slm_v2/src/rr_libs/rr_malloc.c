/* rr_malloc():  call malloc and quit if it fails */
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

/* 8/93: if 0 bytes requested, allocate 1 anyway, to placate the alphas during 
	 address computations */

/* need to include stdlib to prevent warnings at compilation time,
   Philip Clarkson, March 1997 */

#include <stdlib.h>

#include "general.h"

char *rr_malloc(size_t n_bytes)
{
  char *result;
  result = (char *) malloc(MAX(n_bytes,1));
  if (! result) quit(-1,"rr_malloc: could not allocate %d bytes\n",n_bytes);
  return(result);
}
