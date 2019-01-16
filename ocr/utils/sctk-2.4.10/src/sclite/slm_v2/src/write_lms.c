
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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "pc_libs/pc_general.h"
#include "idngram2lm.h"
#include "rr_libs/mips_swap.h"
#include "rr_libs/general.h"
#include "ngram.h"


#define BBO_FILE_VERSION 970314

void write_arpa_lm(ng_t *ng,int verbosity) {

/* This is the format introduced and first used by Doug Paul.
   Optionally use a given symbol for the UNK word (id==0).
*/
/*
Format of the .arpabo file:
------------------------------
<header info - ignored by programs>
\data\
ngram 1=4989
ngram 2=835668
ngram 3=12345678

\1-grams:
...
-0.9792 ABC   -2.2031
...
log10_uniprob(ZWEIG)   ZWEIG   log10_alpha(ZWEIG)

\2-grams:
...
-0.8328 ABC DEFG -3.1234
...
log10_bo_biprob(WAS | ZWEIG)  ZWEIG  WAS   log10_bialpha(ZWEIG,WAS)

\3-grams:
...
-0.234 ABCD EFGHI JKL
...

\end\


*/

  int *current_pos;
  int *end_pos;
  int i;
  int j;
  double log_10_of_e = 1.0 / log(10.0);

  /* HEADER */

  pc_message(verbosity,1,"ARPA-style %d-gram will be written to %s\n",ng->n,ng->arpa_filename);

  fprintf(ng->arpa_fp,"#############################################################################\n");
  fprintf(ng->arpa_fp,"## Copyright (c) 1996, Carnegie Mellon University, Cambridge University,\n");
  fprintf(ng->arpa_fp,"## Ronald Rosenfeld and Philip Clarkson\n");
  fprintf(ng->arpa_fp,"#############################################################################\n");
  fprintf(ng->arpa_fp,"=============================================================================\n");
  fprintf(ng->arpa_fp,"===============  This file was produced by the CMU-Cambridge  ===============\n");
  fprintf(ng->arpa_fp,"===============     Statistical Language Modeling Toolkit     ===============\n"); 
  fprintf(ng->arpa_fp,"=============================================================================\n");
  fprintf(ng->arpa_fp,"This is a %d-gram language model, based on a vocabulary of %d words,\n",ng->n,ng->vocab_size);
  fprintf(ng->arpa_fp,"  which begins \"%s\", \"%s\", \"%s\"...\n",ng->vocab[1],ng->vocab[2],ng->vocab[3]);
  if (ng->vocab_type == CLOSED_VOCAB) {
    fprintf(ng->arpa_fp,"This is a CLOSED-vocabulary model\n");
    fprintf(ng->arpa_fp,"  (OOVs eliminated from training data and are forbidden in test data)\n");
  }
  else {
    if (ng->vocab_type == OPEN_VOCAB_1) {
      fprintf(ng->arpa_fp,"This is an OPEN-vocabulary model (type 1)\n");
      fprintf(ng->arpa_fp,"  (OOVs were mapped to UNK, which is treated as any other vocabulary word)\n");
    }
    else {
      if (ng->vocab_type == OPEN_VOCAB_2) {
	fprintf(ng->arpa_fp,"This is an OPEN-vocabulary model (type 2)\n");
	fprintf(ng->arpa_fp,"  (%.2f of the unigram discount mass was allocated to OOVs)\n",ng->oov_fraction); 
      }
    }
  }
  
  switch (ng->discounting_method) {
  case GOOD_TURING:
    fprintf(ng->arpa_fp,"Good-Turing discounting was applied.\n");
    for (i=1;i<=ng->n;i++) {
      fprintf(ng->arpa_fp,"%d-gram frequency of frequency : ",i);
      for (j=1;j<=ng->fof_size[i-1]-1;j++) {
	fprintf(ng->arpa_fp,"%d ",ng->freq_of_freq[i-1][j]);
      }
      fprintf(ng->arpa_fp,"\n");
    }
    for (i=1;i<=ng->n;i++) {
      fprintf(ng->arpa_fp,"%d-gram discounting ratios : ",i);
      for (j=1;j<=ng->disc_range[i-1];j++) {
	fprintf(ng->arpa_fp,"%.2f ",ng->gt_disc_ratio[i-1][j]);
      }
      fprintf(ng->arpa_fp,"\n");
    }
    break;
  case LINEAR:
    fprintf(ng->arpa_fp,"Linear discounting was applied.\n");
    for (i=1;i<=ng->n;i++) {
      fprintf(ng->arpa_fp,"%d-gram discounting ratio : %g\n",i,ng->lin_disc_ratio[i-1]);
    }
    break;
  case ABSOLUTE:
    fprintf(ng->arpa_fp,"Absolute discounting was applied.\n");
    for (i=1;i<=ng->n;i++) {
      fprintf(ng->arpa_fp,"%d-gram discounting constant : %g\n",i,ng->abs_disc_const[i-1]);
    }
    break;
  case WITTEN_BELL:
    fprintf(ng->arpa_fp,"Witten Bell discounting was applied.\n");
    break;
}


  fprintf(ng->arpa_fp,"This file is in the ARPA-standard format introduced by Doug Paul.\n");
  fprintf(ng->arpa_fp,"\n");
  fprintf(ng->arpa_fp,"p(wd3|wd1,wd2)= if(trigram exists)           p_3(wd1,wd2,wd3)\n");
  fprintf(ng->arpa_fp,"                else if(bigram w1,w2 exists) bo_wt_2(w1,w2)*p(wd3|wd2)\n");
  fprintf(ng->arpa_fp,"                else                         p(wd3|w2)\n");
  fprintf(ng->arpa_fp,"\n");
  fprintf(ng->arpa_fp,"p(wd2|wd1)= if(bigram exists) p_2(wd1,wd2)\n");
  fprintf(ng->arpa_fp,"            else              bo_wt_1(wd1)*p_1(wd2)\n");
  fprintf(ng->arpa_fp,"\n");
  fprintf(ng->arpa_fp,"All probs and back-off weights (bo_wt) are given in log10 form.\n");
  fprintf(ng->arpa_fp,"\n");
  fprintf(ng->arpa_fp,"Data formats:\n");
  fprintf(ng->arpa_fp,"\n");
  fprintf(ng->arpa_fp,"Beginning of data mark: \\data\\\n");

  for (i=1;i<=ng->n;i++) {
    fprintf(ng->arpa_fp,"ngram %d=nr            # number of %d-grams\n",i,i);
  }
  fprintf(ng->arpa_fp,"\n");
  for (i=1;i<=ng->n;i++) {
    fprintf(ng->arpa_fp,"\\%d-grams:\n",i);
    fprintf(ng->arpa_fp,"p_%d     ",i);
    for (j=1;j<=i;j++) {
      fprintf(ng->arpa_fp,"wd_%d ",j);
    }
    if (i == ng->n) {
      fprintf(ng->arpa_fp,"\n");
    }
    else {
      fprintf(ng->arpa_fp,"bo_wt_%d\n",i);
    }
  }  

  fprintf(ng->arpa_fp,"\n");
  fprintf(ng->arpa_fp,"end of data mark: \\end\\\n");
  fprintf(ng->arpa_fp,"\n");

  fprintf(ng->arpa_fp,"\\data\\\n");
  fprintf(ng->arpa_fp,"ngram 1=%d\n",1+ng->vocab_size-ng->first_id);
  for (i=1;i<=ng->n-1;i++) {
    fprintf(ng->arpa_fp,"ngram %d=%d\n",i+1,ng->num_kgrams[i]);
  }

  /* Print unigram info */

  fprintf(ng->arpa_fp,"\n\\1-grams:\n");

  for (i=ng->first_id; i<=ng->vocab_size;i++) {
    
    double log10_uniprob;
    double log10_alpha;
    
    log10_uniprob = ng->uni_log_probs[i]*log_10_of_e;

    if (ng->uni_probs[i]<=0.0) {
      log10_uniprob = -99.999;
    }
    
    if (ng->four_byte_alphas) {
      if (ng->bo_weight4[0][i] > 0.0) {
	log10_alpha = log10(ng->bo_weight4[0][i]);
      }
      else {
	log10_alpha = -99.999;
      }
    }
    else {

      if (double_alpha(ng->bo_weight[0][i],
		       ng->alpha_array,
		       ng->size_of_alpha_array,
		       65535 - ng->out_of_range_alphas,
		       ng->min_alpha,
		       ng->max_alpha) > 0.0) {
	log10_alpha = log10(double_alpha(ng->bo_weight[0][i],
					 ng->alpha_array,
					 ng->size_of_alpha_array,
					 65535 - ng->out_of_range_alphas,
					 ng->min_alpha,
					 ng->max_alpha));
      }
      else {
	log10_alpha = -99.999;
      }

    }

    if (ng->n>1) {
      fprintf(ng->arpa_fp,"%.4f %s\t%.4f\n",
	      log10_uniprob,ng->vocab[i],log10_alpha);
    }
    else {
      fprintf(ng->arpa_fp,"%.4f %s\n",
	      log10_uniprob,ng->vocab[i]);
    }

  }

  current_pos = (int *) rr_malloc(ng->n*sizeof(int));
  end_pos = (int *) rr_malloc(ng->n*sizeof(int)); 


  /* Print 2-gram, ... (n-1)-gram info. */

  for (i=1;i<=ng->n-1;i++) {

    /* Print out the (i+1)-gram */


    int current_table;
    int j;

    int ngcount;
    int marg_count;
    double discounted_ngcount;
    
    double ngprob;
    double log_10_ngprob;
    double ngalpha;
    double log_10_ngalpha;

    /* Initialise variables for the sake of warning-free compilation */
    
    discounted_ngcount = 0.0;
    log_10_ngalpha = 0.0;

    fprintf(ng->arpa_fp,"\n\\%d-grams:\n",i+1);

    /* Go through the n-gram list in order */
    
    for (j=0;j<=ng->n-1;j++) {
      current_pos[j] = 0;
      end_pos[j] = 0;
    }

    for (current_pos[0]=ng->first_id;
	 current_pos[0]<=ng->vocab_size;
	 current_pos[0]++) {
      
      if (return_count(ng->four_byte_counts,
		       ng->count_table[0], 
		       ng->marg_counts,
		       ng->marg_counts4,
		       current_pos[0]) > 0) {
    
	current_table = 1;
      
	if (current_pos[0] == ng->vocab_size) {
	  end_pos[1] = ng->num_kgrams[1]-1;
	}
	else {
	  end_pos[1] = get_full_index(ng->ind[0][current_pos[0]+1],
				      ng->ptr_table[0],
				      ng->ptr_table_size[0],
				      current_pos[0]+1)-1;
	}

	while (current_table > 0) {

	  if (current_table == i) {

	    if (current_pos[i] <= end_pos[i]) {

	      ngcount = return_count(ng->four_byte_counts,
				     ng->count_table[i],
				     ng->count[i],
				     ng->count4[i],
				     current_pos[i]);
	    
	      if (i==1) {
		marg_count = return_count(ng->four_byte_counts,
					  ng->count_table[0], 
					  ng->marg_counts,
					  ng->marg_counts4,
					  current_pos[0]);
	      }
	      else {
		marg_count = return_count(ng->four_byte_counts,
				     ng->count_table[i-1],
				     ng->count[i-1],
				     ng->count4[i-1],
				     current_pos[i-1]);
	      }

	      switch (ng->discounting_method) {
	      case GOOD_TURING:
		if (ngcount <= ng->disc_range[i]) {
		  discounted_ngcount = ng->gt_disc_ratio[i][ngcount] * ngcount;
		}
		else {
		  discounted_ngcount = ngcount;
		}
		break;
	      case ABSOLUTE:
		discounted_ngcount =  ngcount - ng->abs_disc_const[i];
		break;
	      case LINEAR:
		discounted_ngcount = ng->lin_disc_ratio[i]*ngcount; 
		break;
	      case WITTEN_BELL:
		discounted_ngcount = ( ((double) marg_count * ngcount) /
                  (marg_count + num_of_types(i-1,current_pos[i-1],ng)));
		break;
	      }

	      ngprob = (double) discounted_ngcount / marg_count;

	      if (ngprob > 1.0) {
		fprintf(stderr,
			"discounted_ngcount = %f marg_count = %d %d %d %d\n",
		       discounted_ngcount,marg_count,current_pos[0],
		       current_pos[1],current_pos[2]);
		quit(-1,"Error : probablity of ngram is greater than one.\n");
	      }

	      if (ngprob > 0.0) {
		log_10_ngprob = log10(ngprob);
	      }
	      else {
		log_10_ngprob = -99.999;
	      }


	      if (i <= ng->n-2) {
		if (ng->four_byte_alphas) {
		  ngalpha = ng->bo_weight4[i][current_pos[i]];
		}
		else {
		  ngalpha = double_alpha(ng->bo_weight[i][current_pos[i]],
					 ng->alpha_array,
					 ng->size_of_alpha_array,
					 65535 - ng->out_of_range_alphas,
					 ng->min_alpha,
					 ng->max_alpha);
		}
		if (ngalpha > 0.0) {
		  log_10_ngalpha = log10(ngalpha);
		}
		else {
		  log_10_ngalpha = -99.999;
		}
	      }

	      fprintf(ng->arpa_fp,"%.4f ",log_10_ngprob);
	      fprintf(ng->arpa_fp,"%s ",ng->vocab[current_pos[0]]);
	      for (j=1;j<=i;j++) {
		fprintf(ng->arpa_fp,"%s ",ng->vocab[ng->word_id[j][current_pos[j]]]);
	      }
	      if (i <= ng->n-2) {
		fprintf(ng->arpa_fp,"%.4f\n",log_10_ngalpha);
	      }		
	      else {
		fprintf(ng->arpa_fp,"\n");
	      }
	      current_pos[i]++;
	    }
	    else {
	      current_table--;
	      if (current_table > 0) {
		current_pos[current_table]++;
	      }
	    }
	  }
	  else {
	    
	    if (current_pos[current_table] <= end_pos[current_table]) {
	      current_table++;
	      if (current_pos[current_table-1] == ng->num_kgrams[current_table-1]-1) {
		end_pos[current_table] = ng->num_kgrams[current_table]-1;
	      }
	      else {
		end_pos[current_table] = get_full_index(ng->ind[current_table-1][current_pos[current_table-1]+1],ng->ptr_table[current_table-1],ng->ptr_table_size[current_table-1],current_pos[current_table-1]+1)-1;
	      }
	    }
	    else {
	      current_table--;
	      if (current_table > 0) {
		current_pos[current_table]++;
	      }
	    }
	  }
	}
      }
    }
  } 

  free(current_pos);
  free(end_pos);


  fprintf(ng->arpa_fp,"\n\\end\\\n");

  rr_oclose(ng->arpa_fp);

} 
 
void write_bin_lm(ng_t *ng,int verbosity) {
    
  int l_chunk;
  int from_rec;
  int i;
  int j;

  pc_message(verbosity,1,"Binary %d-gram language model will be written to %s\n",ng->n,ng->bin_filename);
  
  ng->version = BBO_FILE_VERSION;

  /* Scalar parameters */

  rr_fwrite(&ng->version,sizeof(int),1,ng->bin_fp,"version");
  rr_fwrite(&ng->n,sizeof(unsigned short),1,ng->bin_fp,"n");

  rr_fwrite(&ng->vocab_size,sizeof(unsigned short),1,ng->bin_fp,"vocab_size");
  rr_fwrite(&ng->no_of_ccs,sizeof(unsigned short),1,ng->bin_fp,"no_of_ccs");
  rr_fwrite(&ng->vocab_type,sizeof(unsigned short),1,ng->bin_fp,"vocab_type");

  rr_fwrite(&ng->count_table_size,sizeof(count_ind_t),1,
	    ng->bin_fp,"count_table_size");
  rr_fwrite(&ng->discounting_method,sizeof(unsigned short),1,
	    ng->bin_fp,"discounting_method");

  rr_fwrite(&ng->min_alpha,sizeof(double),
	    1,ng->bin_fp,"min_alpha");
  rr_fwrite(&ng->max_alpha,sizeof(double),
	    1,ng->bin_fp,"max_alpha");
  rr_fwrite(&ng->out_of_range_alphas,sizeof(unsigned short),
	    1,ng->bin_fp,"out_of_range_alphas");
  rr_fwrite(&ng->size_of_alpha_array,sizeof(unsigned short),
	    1,ng->bin_fp,"size_of_alpha_array");  

  rr_fwrite(&ng->n_unigrams,sizeof(int),1,ng->bin_fp,"n_unigrams");
  rr_fwrite(&ng->zeroton_fraction,sizeof(double),1,
	    ng->bin_fp,"zeroton_fraction");
  rr_fwrite(&ng->oov_fraction,sizeof(double),1,
	    ng->bin_fp,"oov_fraction");
  rr_fwrite(&ng->four_byte_counts,sizeof(flag),1,
	    ng->bin_fp,"four_byte_counts");

  rr_fwrite(&ng->four_byte_alphas,sizeof(flag),1,
	    ng->bin_fp,"four_byte_alphas");

  rr_fwrite(&ng->first_id,sizeof(unsigned short),1,
	    ng->bin_fp,"first_id");

  /* Short and shortish arrays */

  sih_val_write_to_file(ng->vocab_ht,ng->bin_fp,ng->bin_filename,0);

  /* (ng->vocab is not stored in file - will be derived from ng->vocab_ht) */

  if (ng->four_byte_counts) {
    rr_fwrite(ng->marg_counts4,sizeof(int),
	      ng->vocab_size+1,ng->bin_fp,"marg_counts");
  }
  else {
    rr_fwrite(ng->marg_counts,sizeof(count_ind_t),
	      ng->vocab_size+1,ng->bin_fp,"marg_counts");
  }

  rr_fwrite(ng->alpha_array,sizeof(double),
	    ng->size_of_alpha_array,ng->bin_fp,"alpha_array");

  if (!ng->four_byte_counts) {
    for (i=0;i<=ng->n-1;i++) {
      rr_fwrite(ng->count_table[i],sizeof(count_t),
		ng->count_table_size+1,ng->bin_fp,"count_table");
    } 
  }

  /* Could write count_table as one block, but better to be safe and
     do it in chunks. For motivation, see comments about writing tree
     info. */

  rr_fwrite(ng->ptr_table_size,sizeof(unsigned short),
	    ng->n,ng->bin_fp,"ptr_table_size");

  for (i=0;i<=ng->n-1;i++) {
    rr_fwrite(ng->ptr_table[i],sizeof(ptr_tab_t),
	      ng->ptr_table_size[i],ng->bin_fp,"ptr_table");
  }
  
  /* Unigram statistics */

  rr_fwrite(ng->uni_probs,sizeof(uni_probs_t),ng->vocab_size+1,
	    ng->bin_fp,"uni_probs");
  rr_fwrite(ng->uni_log_probs,sizeof(uni_probs_t),ng->vocab_size+1,
	    ng->bin_fp,"uni_log_probs");
  rr_fwrite(ng->context_cue,sizeof(flag),ng->vocab_size+1,
	    ng->bin_fp,"context_cue");

  
  rr_fwrite(ng->cutoffs,sizeof(cutoff_t),ng->n,ng->bin_fp,"cutoffs");

  switch (ng->discounting_method) {
  case GOOD_TURING:
    rr_fwrite(ng->fof_size,sizeof(unsigned short),ng->n,ng->bin_fp,"fof_size");
    rr_fwrite(ng->disc_range,sizeof(unsigned short),ng->n,
	      ng->bin_fp,"disc_range");
    for (i=0;i<=ng->n-1;i++) {
      rr_fwrite(ng->freq_of_freq[i],sizeof(int),
		ng->fof_size[i]+1,ng->bin_fp,"freq_of_freq");
    }    
    for (i=0;i<=ng->n-1;i++) {
      rr_fwrite(ng->gt_disc_ratio[i],sizeof(disc_val_t),
		ng->disc_range[i]+1,ng->bin_fp,"gt_disc_ratio");
    }    
  case WITTEN_BELL:
    break;
  case LINEAR:
    rr_fwrite(ng->lin_disc_ratio,sizeof(disc_val_t),
		ng->n,ng->bin_fp,"lin_disc_ratio");
    break;
  case ABSOLUTE:
    rr_fwrite(ng->abs_disc_const,sizeof(double),
	      ng->n,ng->bin_fp,"abs_disc_const");
    break;
  }

  /* Tree information */

  /* Unigram stuff first, since can be dumped all in one go */

  rr_fwrite(ng->num_kgrams,sizeof(int),ng->n,ng->bin_fp,"num_kgrams");

  if (ng->four_byte_counts) {
    rr_fwrite(ng->count4[0],sizeof(int),ng->vocab_size+1,
	      ng->bin_fp,"unigram counts");
  }
  else {
    rr_fwrite(ng->count[0],sizeof(count_ind_t),ng->vocab_size+1,
	      ng->bin_fp,"unigram counts");
  }

  if (ng->four_byte_alphas) {
    rr_fwrite(ng->bo_weight4[0],sizeof(four_byte_t),ng->vocab_size+1,
	      ng->bin_fp,"unigram backoff weights");
  }
  else {
    rr_fwrite(ng->bo_weight[0],sizeof(bo_weight_t),ng->vocab_size+1,
	      ng->bin_fp,"unigram backoff weights");
  }

  if (ng->n > 1) {
    rr_fwrite(ng->ind[0],sizeof(index__t),ng->vocab_size+1,
	      ng->bin_fp,"unigram -> bigram pointers");
  }

  /* Write the rest of the tree structure in chunks, otherwise the
      kernel buffers are too big. */

  /* Need to do byte swapping */

  for (i=1;i<=ng->n-1;i++) {
    for (j=0;j<=ng->num_kgrams[i];j++) {
      SWAPHALF(&ng->word_id[i][j]);
    }
    if (ng->four_byte_counts) {
      for (j=0;j<=ng->num_kgrams[i];j++) {
	SWAPWORD(&ng->count4[i][j]);
      }
    }
    else {
      for (j=0;j<=ng->num_kgrams[i];j++) {
	SWAPHALF(&ng->count[i][j]);
      }
    }
  }

  for (i=1;i<=ng->n-2;i++) {
    for (j=0;j<=ng->num_kgrams[i];j++) {
      if (ng->four_byte_alphas) {
	SWAPWORD(&ng->bo_weight4[i][j]);
      }
      else {
	SWAPHALF(&ng->bo_weight[i][j]);
      }
    }
    for (j=0;j<=ng->num_kgrams[i];j++) {
      SWAPHALF(&ng->ind[i][j]);
    }
  }

  for (i=1;i<=ng->n-1;i++) {

    from_rec = 0;
    l_chunk = 100000;
    while(from_rec < ng->num_kgrams[i]) {
      if (from_rec+l_chunk > ng->num_kgrams[i]) {
	l_chunk = ng->num_kgrams[i] - from_rec;
      }
      rr_fwrite(&ng->word_id[i][from_rec],1,sizeof(id__t)*l_chunk,ng->bin_fp,"word ids");
      from_rec += l_chunk;
    }
   
  }

  for (i=1;i<=ng->n-1;i++) {

    from_rec = 0;
    l_chunk = 100000;
    while(from_rec < ng->num_kgrams[i]) {
      if (from_rec+l_chunk > ng->num_kgrams[i]) {
	l_chunk = ng->num_kgrams[i] - from_rec;
      }
      if (ng->four_byte_counts) {
	rr_fwrite(&ng->count4[i][from_rec],1,sizeof(int)*l_chunk,ng->bin_fp,"counts");
      }
      else {
	rr_fwrite(&ng->count[i][from_rec],1,sizeof(count_ind_t)*l_chunk,ng->bin_fp,"counts");
      }
      from_rec += l_chunk;
    }
    
  }

  for (i=1;i<=ng->n-2;i++) {

    from_rec = 0;
    l_chunk = 100000;
    while(from_rec < ng->num_kgrams[i]) {
      if (from_rec+l_chunk > ng->num_kgrams[i]) {
	l_chunk = ng->num_kgrams[i] - from_rec;
      }
      if (ng->four_byte_alphas) {
	rr_fwrite(&ng->bo_weight4[i][from_rec],1,sizeof(four_byte_t)*l_chunk,
		  ng->bin_fp,"backoff weights");
      }
      else {
	rr_fwrite(&ng->bo_weight[i][from_rec],1,sizeof(bo_weight_t)*l_chunk,
		  ng->bin_fp,"backoff weights");
      }
      from_rec += l_chunk;
    }
  }

  for (i=1;i<=ng->n-2;i++) {


    from_rec = 0;
    l_chunk = 100000;
    while(from_rec < ng->num_kgrams[i]) {
      if (from_rec+l_chunk > ng->num_kgrams[i]) {
	l_chunk = ng->num_kgrams[i] - from_rec;
      }
      rr_fwrite(&ng->ind[i][from_rec],1,sizeof(index__t)*l_chunk,ng->bin_fp,
		"indices");
      from_rec += l_chunk;
    }

  }

  rr_oclose(ng->bin_fp);

  /* Swap back */

  for (i=1;i<=ng->n-1;i++) {
    for (j=0;j<=ng->num_kgrams[i];j++) {
      SWAPHALF(&ng->word_id[i][j]);
    }
    if (ng->four_byte_counts) {
      for (j=0;j<=ng->num_kgrams[i];j++) {
	SWAPWORD(&ng->count4[i][j]);
      }
    }
    else {
      for (j=0;j<=ng->num_kgrams[i];j++) {
	SWAPHALF(&ng->count[i][j]);
      }
    }
  }

  for (i=1;i<=ng->n-2;i++) {
    for (j=0;j<=ng->num_kgrams[i];j++) {
      if (ng->four_byte_alphas) {
	SWAPWORD(&ng->bo_weight4[i][j]);
      }
      else {
	SWAPHALF(&ng->bo_weight[i][j]);
      }

    }
    for (j=0;j<=ng->num_kgrams[i];j++) {
      SWAPHALF(&ng->ind[i][j]);
    }
  }
  
}






