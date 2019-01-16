/* rr_fseek() */
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

/* call fseek and quit if it fails 
   Roni Rosenfeld, 10/92  */

/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */

#include <stdio.h>
#include "general.h"

void *rr_fseek(FILE *fp, int offset, int mode, char *description)
{
  if (fseek(fp,offset,mode) < 0)
    quit(-1,"ERROR: fseek(<%s>,%d,%d) is improper\n",
	     description, offset, mode);

  return(0); /* Not relevant, but stops compilation warnings. */

}
