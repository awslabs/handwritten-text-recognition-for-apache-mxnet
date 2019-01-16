/* salloc(): Re-implement salloc(), for platforms where it doesn't exist */
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
/* Roni Rosenfeld, 12/92  */

#include <strings.h>
#include "general.h"

char *salloc(char *str)
{
  char *copy = rr_malloc(strlen(str)+1);
  strcpy(copy,str);
  return(copy);
}
