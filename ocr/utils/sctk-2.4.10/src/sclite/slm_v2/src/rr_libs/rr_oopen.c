/* rr_oopen(), rr_oclose():  a generalized output file opener */
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

/* Open a file for output. */
/* If pathanme ends in ".Z", prepare to write thru a 'compress' pipe.  */
/* If pathname is "-", prepare to write to stdout (uncompressed) */

/* Unforgiving: quit if open() fails */
/* Also entry point for closing the associated stream */

/*****************************************************************

  Modified by Philip Clarkson 1/10/96 to allow gzipped files also. 

*****************************************************************/

/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */


#include <stdio.h>
#include "general.h"
#include "strings.h"
char  RRo_is_Z[100];

FILE *rr_oopen(char *path)
{
  static char rname[]="rr_oopen";
  FILE *fp;
  char pipe[256], is_Z;
  int  lpath;

  if (strcmp(path,"-")==0) return(stdout);

  lpath = strlen(path);
  if (strcmp(&path[lpath-2],".Z")==0) {
    if (lpath > sizeof(pipe) - strlen("compress >!  ") - 4)
      quit(-1,"%s: pathname '%s' is too long\n",rname,path);
     sprintf(pipe,"compress > %s",path);
     fp = popen(pipe,"w");
     if (!fp) quit(-1,"%s: problems opening the pipe '%s' for output.\n", rname,pipe);
     is_Z = 1;
  }
  else {
    if (strcmp(&path[lpath-3],".gz")==0) {
      if (lpath > sizeof(pipe) - strlen("gzip >!  ") -4)
	quit(-1,"%s: pathname '%s' is too long\n",rname,path);
      sprintf(pipe,"gzip > %s",path);
      fp = popen(pipe,"w");
      if (!fp) quit(-1,"%s: problems opening the pipe '%s' for output.\n", rname,pipe);
      is_Z = 1;
    }
    else {
      fp = rr_fopen(path,"w");
      is_Z = 0;
    }
  }

  if (fileno(fp) > sizeof(RRo_is_Z)-1) quit(-1,"%s: fileno = %d is too large\n",rname,fileno(fp));
  RRo_is_Z[fileno(fp)] = is_Z;

  return(fp);
}

void *rr_oclose(FILE *fp)
{
  if (fp==stdout) return(0);
  fflush(fp); 
  if (RRo_is_Z[fileno(fp)]) 
    pclose(fp);
  else
    fclose(fp);

  return(0); /* Not relevant, but stops compilation warnings. */

}
