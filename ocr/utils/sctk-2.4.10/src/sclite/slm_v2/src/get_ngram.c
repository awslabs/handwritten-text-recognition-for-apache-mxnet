/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1996, Carnegie Mellon University, Cambridge University,
Ronald Rosenfeld and Philip Clarkson.

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
/* Function to read in a n-gram from the file specified */

/* Returns zero if we have reached eof */

#include <stdio.h>
#include <stdarg.h>
#include "rr_libs/general.h"
#include "ngram.h"


int get_ngram(FILE *id_ngram_fp, ngram *ng, flag ascii) {  

  int i;

  if (ascii) {

    for (i=0;i<=ng->n-1;i++) {
      if (fscanf(id_ngram_fp,"%hu",&ng->id_array[i]) != 1) {
	if (rr_feof(id_ngram_fp)) {
	  return 0;
	}
	quit(-1,"Error reading from id_ngram file.\n");
      }
    }
    if (fscanf(id_ngram_fp,"%d",&ng->count) != 1) {
      if (rr_feof(id_ngram_fp)) {
	return 0;
      }
      quit(-1,"Error reading from id_ngram file.2\n");
    }
  }
  else {

    /* Read in array of ids */

    /* Read in one at a time - slower, but spares us having to think 
       about byte-orders. */
    
    for (i=0;i<=ng->n-1;i++) {
      if (rr_feof(id_ngram_fp)) {
	return 0;
      }
      rr_fread(&ng->id_array[i],sizeof(id__t),1,id_ngram_fp,
	       "from id_ngram file",0);
    }

    /* Read in count */
    if (rr_feof(id_ngram_fp)) {
      return 0;
    }
    rr_fread(&ng->count,sizeof(count_t),1,id_ngram_fp,
	     "count from id_ngram file",0);

  }

  return 1;
  
}



