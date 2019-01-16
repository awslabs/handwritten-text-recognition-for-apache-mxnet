/* rr_feof() */
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

/* Check for EOF condition on a given fp.
   Needed because "feof()" does not work for some reason.
   Roni Rosenfeld, 11/92  */
#include <stdio.h>
#include "general.h"

int rr_feof(FILE *fp)
{
  int dummychar;
  dummychar = getc(fp);
  if (dummychar == EOF) return(1);
  if (ungetc(dummychar,fp)==EOF) quit(-1,"rr_feof: ungetc error\n");
  return(0);
}

