
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

#include <math.h>
#include <stdio.h>

 /*
  * PWP: coding change.  Numbers were previously coded base-3.
  * Now base-4, since (int) pow(4,i) == 1 << (2*i)
  */
 
void decode_bo_case(int bo_case,
		    int context_length,
		    FILE *annotation_fp) {
  
  int i;
  int current_bo_case;
  
  for(i=context_length-1;i>=0;i--) {

    fprintf(annotation_fp,"%d",i+2);

    current_bo_case = bo_case / (1 << (2*i));

    if (current_bo_case == 1) {
      fprintf(annotation_fp,"-");
    }
    else {
      if (current_bo_case == 2) {
	fprintf(annotation_fp,"x");
      }
      else {
	i=-2;
      }
    }
    bo_case -= (current_bo_case * (1 << (2*i)));

  }

  if (i>-2) {
    fprintf(annotation_fp,"1");
  }

  fprintf(annotation_fp,"\n");

}
