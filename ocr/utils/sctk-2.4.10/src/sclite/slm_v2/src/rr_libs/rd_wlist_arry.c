/* read_wlist_into_array */
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

/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */


#include <stdio.h>
#include <strings.h>
#include "general.h"

/* allocate an lagre enough array and read in a list of words (first word on each line)
   Leave entry no. 0 empty.
*/

void read_wlist_into_array(wlist_filename, verbosity,  p_wlist, p_n_wlist)
char    *wlist_filename;
char    ***p_wlist;
int     verbosity,  *p_n_wlist;
{
  static char rname[]="read_wlist_into_array";
  FILE   *wlist_fp = rr_iopen(wlist_filename);
  char   **wlist;
  int    n_wlist, c, lastc, entry_no;
  char   wlist_entry[1024], word[256];


  lastc = '\0';
  n_wlist = 0;
  while ((c=getc(wlist_fp)) != EOF) {
     if (c == '\n') n_wlist++;
     lastc = c;
  }
  if (lastc != '\n') quit(-1,"%s: no newline at end of %s\n",rname,wlist_filename);
  rr_iclose(wlist_fp);
  wlist_fp = rr_iopen(wlist_filename);

  wlist = (char **) rr_malloc((n_wlist+1)*sizeof(char *));
  entry_no = 0;

  while (fgets (wlist_entry, sizeof (wlist_entry), wlist_fp)) {
     if (strncmp(wlist_entry,"##",2)==0) continue;
     /* warn ARPA sites who may have comments starting with a single '#' */
     sscanf (wlist_entry, "%s ", word);
     if (strncmp(wlist_entry,"#",1)==0) {
	fprintf(stderr,"\n\n===========================================================\n");
        fprintf(stderr,"%s:\nWARNING: line assumed NOT a comment:\n",rname);
	fprintf(stderr,     ">>> %s <<<\n",wlist_entry);
        fprintf(stderr,     "         '%s' will be included in the vocabulary\n",word);
        fprintf(stderr,     "         (comments must start with '##')\n");
	fprintf(stderr,"===========================================================\n\n");
     }
     wlist[++entry_no] = salloc(word);
  }
  rr_iclose(wlist_fp);
  if (verbosity) fprintf(stderr,"%s: a list of %d words was read from \"%s\".\n",
	  	 		 rname,entry_no,wlist_filename);
  *p_wlist = wlist;
  *p_n_wlist = entry_no;
}


