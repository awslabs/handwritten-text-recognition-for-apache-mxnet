/* rr_fread() */
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

/* call fread and quit if it fails 
   Also SWAP the data if read on a MIPS machine 
   Roni Rosenfeld, 9/92  */


/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */


#include <stdio.h>
#include "mips_swap.h"
#include "general.h"

void *rr_fread(char *ptr, int elsize, int n_elem, FILE *fp,
	       char *header, int not_more)
{
  int n_read, i;
  char dummychar;

  if (n_elem > 0) {
     n_read = fread(ptr,elsize,n_elem,fp);
     if (n_read != n_elem) quit(-1,
	"rr_fread: problems reading %s.  Only %d of %d elements were read\n",
	 header, n_read, n_elem);

     if (elsize == sizeof(int)) {
        for (i=0; i<n_elem; i++) {
	   SWAPWORD(ptr+(elsize*i));
	}
     }
     else if (elsize == sizeof(short)) {
        for (i=0; i<n_elem; i++) {
	   SWAPHALF(ptr+(elsize*i));
        }
     }
     else if (elsize == sizeof(double)) {
        for (i=0; i<n_elem; i++) {
	   SWAPDOUBLE(ptr+(elsize*i));
	}
     }
  }

  if (not_more) {
     if (fread(&dummychar,1,1,fp) != 0)
       quit(-1,"rr_fread: more data after %s - should not be there\n",header);
  }

  return(0); /* Not relevant, but stops compilation warnings. */

}
