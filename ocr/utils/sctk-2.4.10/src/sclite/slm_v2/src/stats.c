
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

#include "evallm.h"

void display_stats(ng_t *ng) {
  
  int i;
  int j;

  fprintf(stderr,"This is a %d-gram language model, based on a vocabulary of %d words,\n",ng->n,ng->vocab_size);
  fprintf(stderr,"  which begins \"%s\", \"%s\", \"%s\"...\n",ng->vocab[1],ng->vocab[2],ng->vocab[3]);

  if (ng->no_of_ccs == 1) {
    fprintf(stderr,"There is 1 context cue.");
  }
  else {
    fprintf(stderr,"There are %d context cues.\n",ng->no_of_ccs);
  }
  if (ng->no_of_ccs > 0 && ng->no_of_ccs < 10) {
    if (ng->no_of_ccs == 1) {
      fprintf(stderr,"This is : ");
    }
    else {
      fprintf(stderr,"These are : ");
    }
    for (i=ng->first_id;i<=ng->vocab_size;i++) {
      if (ng->context_cue[i]) {
	fprintf(stderr,"\"%s\" ",ng->vocab[i]);
      }
    }
    fprintf(stderr,"\n");
  }

  if (ng->vocab_type == CLOSED_VOCAB) {
    fprintf(stderr,"This is a CLOSED-vocabulary model\n");
    fprintf(stderr,"  (OOVs eliminated from training data and are forbidden in test data)\n");
  }
  else {
    if (ng->vocab_type == OPEN_VOCAB_1) {
      fprintf(stderr,"This is an OPEN-vocabulary model (type 1)\n");
      fprintf(stderr,"  (OOVs were mapped to UNK, which is treated as any other vocabulary word)\n");
    }
    else {
      if (ng->vocab_type == OPEN_VOCAB_2) {
	fprintf(stderr,"This is an OPEN-vocabulary model (type 2)\n");
	fprintf(stderr,"  (%.2f of the unigram discount mass was allocated to OOVs)\n",ng->oov_fraction); 
      }
    }
  }

  if (ng->four_byte_alphas) {
    fprintf(stderr,"The back-off weights are stored in four bytes.\n");
  }
  else {
    fprintf(stderr,"The back-off weights are stored in two bytes.\n");
  }
  
  for (i=2;i<=ng->n;i++) {
    fprintf(stderr,"The %d-gram component was based on %d %d-grams.\n",i,ng->num_kgrams[i-1],i);
  }

  switch (ng->discounting_method) {
  case GOOD_TURING:
    fprintf(stderr,"Good-Turing discounting was applied.\n");
    for (i=1;i<=ng->n;i++) {
      fprintf(stderr,"%d-gram frequency of frequency : ",i);
      for (j=1;j<=ng->fof_size[i-1]-1;j++) {
	fprintf(stderr,"%d ",ng->freq_of_freq[i-1][j]);
      }
      fprintf(stderr,"\n");
    }
    for (i=1;i<=ng->n;i++) {
      fprintf(stderr,"%d-gram discounting ratios : ",i);
      for (j=1;j<=ng->disc_range[i-1];j++) {
	fprintf(stderr,"%.2f ",ng->gt_disc_ratio[i-1][j]);
      }
      fprintf(stderr,"\n");
    }
    break;
  case LINEAR:
    fprintf(stderr,"Linear discounting was applied.\n");
    for (i=1;i<=ng->n;i++) {
      fprintf(stderr,"%d-gram discounting ratio : %g\n",i,ng->lin_disc_ratio[i-1]);
    }
    break;
  case ABSOLUTE:
    fprintf(stderr,"Absolute discounting was applied.\n");
    for (i=1;i<=ng->n;i++) {
      fprintf(stderr,"%d-gram discounting constant : %g\n",i,ng->abs_disc_const[i-1]);
    }
    break;
  case WITTEN_BELL:
    fprintf(stderr,"Witten Bell discounting was applied.\n");
    break;
  }

}

void display_arpa_stats(arpa_lm_t *arpa_ng) {

  int i;

  fprintf(stderr,"This is a %d-gram language model, based on a vocabulary of %d words,\n",arpa_ng->n,arpa_ng->vocab_size);
  fprintf(stderr,"  which begins \"%s\", \"%s\", \"%s\"...\n",
	  arpa_ng->vocab[1],arpa_ng->vocab[2],arpa_ng->vocab[3]);
  
  if (arpa_ng->no_of_ccs == 1) {
    fprintf(stderr,"There is 1 context cue.");
  }
  else {
    fprintf(stderr,"There are %d context cues.\n",arpa_ng->no_of_ccs);
  }
  if (arpa_ng->no_of_ccs > 0 && arpa_ng->no_of_ccs < 10) {
    if (arpa_ng->no_of_ccs == 1) {
      fprintf(stderr,"This is : ");
    }
    else {
      fprintf(stderr,"These are : ");
    }
    for (i=arpa_ng->first_id;i<=arpa_ng->vocab_size;i++) {
      if (arpa_ng->context_cue[i]) {
	fprintf(stderr,"\"%s\" ",arpa_ng->vocab[i]);
      }
    }
    fprintf(stderr,"\n");
  }

  if (arpa_ng->vocab_type == CLOSED_VOCAB) {
    fprintf(stderr,"This is a CLOSED-vocabulary model\n");
    fprintf(stderr,"  (OOVs eliminated from training data and are forbidden in test data)\n");
  }
  else {
    if (arpa_ng->vocab_type == OPEN_VOCAB_1) {
      fprintf(stderr,"This is an OPEN-vocabulary model (type 1)\n");
      fprintf(stderr,"  (OOVs were mapped to UNK, which is treated as any other vocabulary word)\n");
    }
    else {
      if (arpa_ng->vocab_type == OPEN_VOCAB_2) {
	fprintf(stderr,"This is an OPEN-vocabulary model (type 2)\n");
      }
    }
  }

  for (i=2;i<=arpa_ng->n;i++) {
    fprintf(stderr,"The %d-gram component was based on %d %d-grams.\n",i,
	    arpa_ng->num_kgrams[i-1],i);
  }

}
