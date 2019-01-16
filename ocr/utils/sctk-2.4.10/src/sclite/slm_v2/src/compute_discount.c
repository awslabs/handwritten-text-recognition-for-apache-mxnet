

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

/* Basically copied from version 1 */

#include "rr_libs/general.h"
#include "ngram.h"
#include "pc_libs/pc_general.h"

void compute_gt_discount(int n,
			 int            *freq_of_freq,
			 int fof_size,
			 unsigned short *disc_range,
			 int cutoff,
			 int verbosity,
			 disc_val_t     **discounted_values) {


  /* Lots of this is lifted straight from V.1 */

  flag done;
  int r;
  int K;
  double common_term;
  double first_term;
  double *D;

  D = (double *) rr_calloc((*disc_range)+1,sizeof(double));
  *discounted_values = D; 

  /* Trap standard things (taken from V.1) */

  if (fof_size == 0) {
    return;
  }

  if (freq_of_freq[1] == 0) {
    pc_message(verbosity,2,"Warning : %d-gram : f-of-f[1] = 0 --> %d-gram discounting is disabled.\n",n,n);
    *disc_range=0;
    return;
  }

  if (*disc_range + 1 > fof_size) {
    pc_message(verbosity,2,"Warning : %d-gram : max. recorded f-o-f is only %d\n",n,fof_size);
    pc_message(verbosity,2,"%d-gram discounting range is reset to %d.\n",fof_size,n,fof_size-1);
    *disc_range = fof_size-1;
  }

  done = 0;

  while (!done) {
    if (*disc_range == 0) {
      pc_message(verbosity,2,"Warning : %d-gram : Discounting is disabled.\n",n);
      return;
    }

    if (*disc_range == 1) {
      /* special treatment for 1gram if there is a zeroton count: */
      if ((n==1) && freq_of_freq[0]>0) {
	D[1] = freq_of_freq[1] / ((float) (freq_of_freq[1] + freq_of_freq[0]));
	pc_message(verbosity,2,"Warning : %d-gram : Discounting range is 1; setting P(zeroton)=P(singleton).\nDiscounted value : %.2f\n",n,D[1]);
	return;
      }
      else {
	pc_message(verbosity,2,"Warning : %d-gram : Discounting range of 1 is equivalent to excluding \nsingletons.\n",n);

      }
    }

    K = *disc_range;
    common_term = ((double) (K+1) * freq_of_freq[K+1]) / freq_of_freq[1];
    if (common_term<=0.0 || common_term>=1.0) {
      pc_message(verbosity,2,"Warning : %d-gram : GT statistics are out of range; lowering cutoff to %d.\n",n,K-1);
      (*disc_range)--;
    }
    else {
      for (r=1;r<=K;r++) {
	first_term = ((double) ((r+1) * freq_of_freq[r+1]))
		             /  (r    * freq_of_freq[r]);
	D[r]=(first_term - common_term)/(1.0 - common_term);
      }
      pc_message(verbosity,3,"%d-gram : cutoff = %d, discounted values:",n,K);
      for (r=1;r<=K;r++) {
	pc_message(verbosity,3," %.2f",D[r]);
      }
      pc_message(verbosity,3,"\n");
      done = 1;
      for (r=1; r<=K; r++) {
	if (D[r]<0 || D[r]>1.0) {
	  pc_message(verbosity,2,"Warning : %d-gram : Some discount values are out of range;\nlowering discounting range to %d.\n",n,K-1);
	  (*disc_range)--;
	  r=K+1;
	  done = 0;
	}
      }
    }	      
  }

   for (r=1; r<=MIN(cutoff,K); r++) D[r] = 0.0;

}

