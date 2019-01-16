/* rr_fexists(): check whether a given pathname points to a valid file */
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

#include <sys/types.h>
#include <sys/stat.h>

int rr_fexists (char *path)
{
  struct stat file_stat;
  
  if (stat(path,&file_stat)==0) return 1;
  else return 0;
}
