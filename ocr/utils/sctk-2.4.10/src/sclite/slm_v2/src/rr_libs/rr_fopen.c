/* rr_fopen(): call fopen and quit if it fails */
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

#include <stdio.h>
#include "general.h"

FILE *rr_fopen(char *filename, char *mode)
{
  FILE *fp;
  fp = fopen(filename,mode);
  if (!fp) 
    quit(-1,"rr_fopen: problems opening %s for \"%s\".\n", filename, mode);
  return(fp);
}
