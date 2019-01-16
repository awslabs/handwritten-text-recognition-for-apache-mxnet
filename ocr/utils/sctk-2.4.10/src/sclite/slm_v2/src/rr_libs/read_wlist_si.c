/* read_wlist_into_siht: read a word list into a (string-to-intval) hash table */
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
#include <string.h>
#include "general.h"
#include "sih.h"

void read_wlist_into_siht(char *wlist_filename, int verbosity,  
			  sih_t *p_word_id_ht, int *p_n_wlist)
{
  static char rname[]="read_wlist_into_siht";
  FILE   *wlist_fp = rr_iopen(wlist_filename);
  char   wlist_entry[1024], word[256], *word_copy;
  int    entry_no = 0;

  while (fgets (wlist_entry, sizeof (wlist_entry), wlist_fp)) {
     if (strncmp(wlist_entry,"##",2) == 0) continue;
     entry_no++;
     sscanf (wlist_entry, "%s ", word);
     if (strncmp(wlist_entry,"#",1)==0) {
	fprintf(stderr,"\n\n===========================================================\n");
        fprintf(stderr,"%s:\nWARNING: line assumed NOT a comment:\n",rname);
	fprintf(stderr,     ">>> %s <<<\n",wlist_entry);
        fprintf(stderr,     "         '%s' will be included in the vocabulary\n",word);
        fprintf(stderr,     "         (comments must start with '##')\n");
	fprintf(stderr,"===========================================================\n\n");
     }
     word_copy = salloc(word);
     sih_add(p_word_id_ht, word_copy, entry_no);
  }
  rr_iclose(wlist_fp);
  if (verbosity) 
     fprintf(stderr,"%s: a list of %d words was read from \"%s\".\n",
	  	    rname,entry_no,wlist_filename);
  *p_n_wlist = entry_no;
}
