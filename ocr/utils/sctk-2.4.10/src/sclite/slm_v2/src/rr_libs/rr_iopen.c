/* rr_iopen(), rr_iclose():  a generalized input file opener */
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

/* Open for input a file which may potentially be Z compressed */
/* If pathanme ends in ".Z", assume the file is compressed.
   Otherwise, look for it and assume it is uncompressed.
 	      If not found, append .Z and look again, assuming compressed */
/* If pathname is "-", use stdin and assume it is uncompressed */

/* Unforgiving: quit if open() fails */
/* Also entry point for closing the associated stream */

/*****************************************************************

  Modified by Philip Clarkson 1/10/96 to allow gzipped files also. 

*****************************************************************/

/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */

#include <stdio.h>
#include "general.h"
#include "strings.h"
char  RRi_is_Z[100];

FILE *rr_iopen(char *path)
{
  static char rname[]="rr_iopen";
  FILE *fp;
  char pipe[256], is_Z;
  int  lpath;

  if (strcmp(path,"-")==0) return(stdin);

  lpath = strlen(path);
  if (lpath > sizeof(pipe) - strlen("cat | gunzip ") - 4)
    quit(-1,"%s: pathname '%s' is too long\n",rname,path);

  if (strcmp(&path[lpath-2],".Z")==0) {
     /* popen() does not report error if file doesn't exist, so: */
     if (!rr_fexists(path)) quit(-1,"%s: file '%s' not found\n",rname,path);
     sprintf(pipe,"zcat %s",path);
     goto Z;
  }

  else if (strcmp(&path[lpath-3],".gz")==0) {
     /* popen() does not report error if file doesn't exist, so: */
     if (!rr_fexists(path)) quit(-1,"%s: file '%s' not found\n",rname,path);
     sprintf(pipe,"cat %s | gunzip",path);
     goto Z;
  }

  else if (!rr_fexists(path)) {
     sprintf(pipe,"%s.Z",path);
     /* popen() does not report error if file doesn't exist, so: */
     if (!rr_fexists(pipe)) {
       sprintf(pipe,"%s.gz",path);
       if (!rr_fexists(pipe)) {
	 quit(-1,"%s: None of '%s' '%s.Z' or '%s.gz' exist.\n",rname,path,path,path);
       }
       sprintf(pipe,"cat %s.gz | gunzip",path);
       goto Z;
     }
     sprintf(pipe,"zcat %s.Z",path);
     goto Z;
  }
  else {
     fp = rr_fopen(path,"r");
     is_Z = 0;
     goto record;
  }

Z:
  fp = popen(pipe,"r");
  if (!fp) quit(-1,"%s: problems opening the pipe '%s' for input.\n", rname,pipe);
  is_Z = 1;

record:
  if (fileno(fp) > sizeof(RRi_is_Z)-1) quit(-1,"%s: fileno = %d is too large\n",rname,fileno(fp));
  RRi_is_Z[fileno(fp)] = is_Z;

  return(fp);
}

void *rr_iclose(FILE *fp)
{
  if (fp==stdin) return(0);
  else if (RRi_is_Z[fileno(fp)]) pclose(fp);
  else fclose(fp);

  return(0); /* Not relevant, but stops compilation warnings. */

}
