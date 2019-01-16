
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


/* Must at all times ensure consistency with write_lms.c */


#include <stdio.h>
#include "rr_libs/general.h"
#include "rr_libs/sih.h"
#include "ngram.h"
#include "evallm.h"
#include <string.h>
#include <stdlib.h>
#include "idngram2lm.h"

#define BBO_FILE_VERSION 970314


void load_lm(ng_t *ng,
	     char *lm_filename) {

  int i;

  ng->bin_fp = rr_iopen(lm_filename);

  rr_fread(&ng->version,sizeof(int),1,ng->bin_fp,"from lm file",0);

  if (ng->version != BBO_FILE_VERSION) {
    quit(-1,"Error : Language model file %s appears to be corrupted.\n",
	 lm_filename);
  }

  /* Scalar parameters */

  rr_fread(&ng->n,sizeof(unsigned short),1,ng->bin_fp,"n",0);

  rr_fread(&ng->vocab_size,sizeof(unsigned short),1,ng->bin_fp,"vocab_size",0);
  rr_fread(&ng->no_of_ccs,sizeof(unsigned short),1,ng->bin_fp,"no_of_ccs",0);

  rr_fread(&ng->vocab_type,sizeof(unsigned short),1,ng->bin_fp,"vocab_type",0);

  rr_fread(&ng->count_table_size,sizeof(count_ind_t),1,
	    ng->bin_fp,"count_table_size",0);
  rr_fread(&ng->discounting_method,sizeof(unsigned short),1,
	    ng->bin_fp,"discounting_method",0);
 
  rr_fread(&ng->min_alpha,sizeof(double),
	    1,ng->bin_fp,"min_alpha",0);
  rr_fread(&ng->max_alpha,sizeof(double),
	    1,ng->bin_fp,"max_alpha",0);
  rr_fread(&ng->out_of_range_alphas,sizeof(unsigned short),
	    1,ng->bin_fp,"out_of_range_alphas",0);

  rr_fread(&ng->size_of_alpha_array,sizeof(unsigned short),
	   1,ng->bin_fp,"size_of_alpha_array",0);


  
  rr_fread(&ng->n_unigrams,sizeof(int),1,ng->bin_fp,"n_unigrams",0);
  rr_fread(&ng->zeroton_fraction,sizeof(double),1,
	    ng->bin_fp,"zeroton_fraction",0);

  rr_fread(&ng->oov_fraction,sizeof(double),1,
	   ng->bin_fp,"oov_fraction",0);
  rr_fread(&ng->four_byte_counts,sizeof(flag),1,
	   ng->bin_fp,"four_byte_counts",0); 
  rr_fread(&ng->four_byte_alphas,sizeof(flag),1,
	   ng->bin_fp,"four_byte_alphas",0);
  rr_fread(&ng->first_id,sizeof(unsigned short),1,
	   ng->bin_fp,"first_id",0);

  ng->vocab_ht = (sih_t *) rr_malloc(sizeof(sih_t));
  sih_val_read_from_file(ng->vocab_ht,ng->bin_fp,lm_filename,0);
  get_vocab_from_vocab_ht(ng->vocab_ht,ng->vocab_size,0,&ng->vocab);
  ng->vocab[0] = salloc("<UNK>");

  if (ng->four_byte_counts) {
    ng->marg_counts4 = (int *) 
      rr_malloc(sizeof(int)*(ng->vocab_size+1));
    rr_fread(ng->marg_counts4,sizeof(int),ng->vocab_size+1,
	     ng->bin_fp,"marg_counts",0);
  } 
  else {
    ng->marg_counts = (count_ind_t *) 
      rr_malloc(sizeof(count_ind_t)*(ng->vocab_size+1));
    rr_fread(ng->marg_counts,sizeof(count_ind_t),ng->vocab_size+1,
	     ng->bin_fp,"marg_counts",0);
  }

  ng->alpha_array = (double *) 
    rr_malloc(sizeof(double)*(ng->size_of_alpha_array));
  rr_fread(ng->alpha_array,sizeof(double),
	   ng->size_of_alpha_array,ng->bin_fp,"alpha_array",0);

  

  ng->count_table = (count_t **) rr_malloc(sizeof(count_t *)*ng->n);
  if (!ng->four_byte_counts) {
    for (i=0;i<=ng->n-1;i++) {
      ng->count_table[i] = (count_t *) 
	rr_malloc(sizeof(count_t)*(ng->count_table_size+1));
      rr_fread(ng->count_table[i],sizeof(count_t),
	       ng->count_table_size+1,ng->bin_fp,"count_table",0);
    } 
  }

  ng->ptr_table_size = (unsigned short *) 
    rr_malloc(sizeof(unsigned short)*ng->n);
  rr_fread(ng->ptr_table_size,sizeof(unsigned short),
	   ng->n,ng->bin_fp,"ptr_table_size",0);

  ng->ptr_table = (ptr_tab_t **) rr_malloc(sizeof(ptr_tab_t *)*ng->n);

  for (i=0;i<=ng->n-1;i++) {
    ng->ptr_table[i] = (ptr_tab_t *) 
      rr_malloc(sizeof(ptr_tab_t)*ng->ptr_table_size[i]);
    rr_fread(ng->ptr_table[i],sizeof(ptr_tab_t),
	     ng->ptr_table_size[i],ng->bin_fp,"ptr_table",0);
  }

  ng->uni_probs = (uni_probs_t *) 
    rr_malloc(sizeof(uni_probs_t)*(ng->vocab_size+1));
  ng->uni_log_probs = (uni_probs_t *) 
    rr_malloc(sizeof(uni_probs_t)*(ng->vocab_size+1));
  ng->context_cue = (flag *) 
    rr_malloc(sizeof(flag)*(ng->vocab_size+1));

  rr_fread(ng->uni_probs,sizeof(uni_probs_t),ng->vocab_size+1,
	  ng->bin_fp,"uni_probs",0);
  rr_fread(ng->uni_log_probs,sizeof(uni_probs_t),ng->vocab_size+1,
	  ng->bin_fp,"uni_log_probs",0);
  rr_fread(ng->context_cue,sizeof(flag),ng->vocab_size+1,
	  ng->bin_fp,"context_cue",0);

  ng->cutoffs = (cutoff_t *) rr_malloc(sizeof(cutoff_t)*ng->n);
  rr_fread(ng->cutoffs,sizeof(cutoff_t),ng->n,ng->bin_fp,"cutoffs",0);

  switch (ng->discounting_method) {
  case GOOD_TURING:
    ng->fof_size = (unsigned short *) rr_malloc(sizeof(unsigned short)*ng->n);
    ng->disc_range = (unsigned short *) 
      rr_malloc(sizeof(unsigned short)*ng->n);
    rr_fread(ng->fof_size,sizeof(unsigned short),ng->n,
	     ng->bin_fp,"fof_size",0);
    rr_fread(ng->disc_range,sizeof(unsigned short),ng->n,
	      ng->bin_fp,"disc_range",0);
    ng->freq_of_freq = (int **) rr_malloc(sizeof(int *)*ng->n);
    for (i=0;i<=ng->n-1;i++) {
      ng->freq_of_freq[i] = (int *) rr_calloc(ng->fof_size[i]+1,sizeof(int));
    }
    ng->gt_disc_ratio = (disc_val_t **) rr_malloc(sizeof(disc_val_t *)*ng->n);
    for (i=0;i<=ng->n-1;i++){
      ng->gt_disc_ratio[i] = (disc_val_t *) 
	rr_malloc(sizeof(disc_val_t)*(ng->disc_range[i]+1));
    }
    for (i=0;i<=ng->n-1;i++) {
      rr_fread(ng->freq_of_freq[i],sizeof(int),
	       ng->fof_size[i]+1,ng->bin_fp,"freq_of_freq",0);

    }    
    for (i=0;i<=ng->n-1;i++) {
      rr_fread(ng->gt_disc_ratio[i],sizeof(disc_val_t),
	       ng->disc_range[i]+1,ng->bin_fp,"gt_disc_ratio",0);
    }    
    break;
  case WITTEN_BELL:
    break;
  case LINEAR:
    ng->lin_disc_ratio = (disc_val_t *) rr_malloc(sizeof(disc_val_t)*ng->n);
    rr_fread(ng->lin_disc_ratio,sizeof(disc_val_t),ng->n,ng->bin_fp,"lin_disc_ratio",0);
    break;
  case ABSOLUTE:
    ng->abs_disc_const = (double *) rr_malloc(sizeof(double)*ng->n);
    rr_fread(ng->abs_disc_const,sizeof(double),ng->n,ng->bin_fp,"abs_disc_const",0);
    break;
  }

  /* Tree information */

  ng->num_kgrams = (int *) rr_malloc(sizeof(int)*ng->n);
  rr_fread(ng->num_kgrams,sizeof(int),ng->n,ng->bin_fp,"num_kgrams",0);

  ng->count = (count_ind_t **) rr_malloc(sizeof(count_ind_t *)*ng->n);
  ng->count4 = (int **) rr_malloc(sizeof(int *)*ng->n);

  if (ng->four_byte_counts) {
    ng->count4[0] = (int *) rr_malloc(sizeof(int)*(ng->vocab_size+1));
    for (i=1;i<=ng->n-1;i++) {
      ng->count4[i] = (int *) rr_malloc(sizeof(int)*ng->num_kgrams[i]);
    }
  }
  else {
    ng->count[0] = (count_ind_t *) 
      rr_malloc(sizeof(count_ind_t)*(ng->vocab_size+1));
    for (i=1;i<=ng->n-1;i++) {
      ng->count[i] = (count_ind_t *) 
	rr_malloc(sizeof(count_ind_t)*ng->num_kgrams[i]);
    }
  }  
  
  if (ng->four_byte_alphas) {
    ng->bo_weight4 = (four_byte_t **) rr_malloc(sizeof(four_byte_t *)*ng->n);
    ng->bo_weight4[0] = (four_byte_t *) 
      rr_malloc(sizeof(four_byte_t)*(ng->vocab_size+1)); 
    for (i=1;i<=ng->n-2;i++) {
      ng->bo_weight4[i] = (four_byte_t *) 
	rr_malloc(sizeof(four_byte_t)*ng->num_kgrams[i]);
    }
  }
 
  else {

    ng->bo_weight = (bo_weight_t **) rr_malloc(sizeof(bo_weight_t *)*ng->n);
    ng->bo_weight[0] = (bo_weight_t *)
      rr_malloc(sizeof(bo_weight_t)*(ng->vocab_size+1));
    for (i=1;i<=ng->n-2;i++) {
      ng->bo_weight[i] = (bo_weight_t *) 
	rr_malloc(sizeof(bo_weight_t)*ng->num_kgrams[i]);
    }
  }

  ng->ind = (index__t **) rr_malloc(sizeof(index__t *)*ng->n);
  ng->ind[0] = (index__t *)
    rr_malloc(sizeof(index__t)*(ng->vocab_size+1));
  for (i=1;i<=ng->n-2;i++) {
    ng->ind[i] = (index__t *) 
      rr_malloc(sizeof(index__t)*ng->num_kgrams[i]);
  }
  
  ng->word_id = (id__t **) rr_malloc(sizeof(id__t *)*ng->n);
  for (i=1;i<=ng->n-1;i++) {
    ng->word_id[i] = (id__t *) 
      rr_malloc(sizeof(id__t)*ng->num_kgrams[i]);
  }
  
  if (ng->four_byte_counts) {
    rr_fread(ng->count4[0],sizeof(int),ng->vocab_size+1,
	     ng->bin_fp,"unigram counts",0); 
  }
  else {
    rr_fread(ng->count[0],sizeof(count_ind_t),ng->vocab_size+1,
	     ng->bin_fp,"unigram counts",0);
  }
  if (ng->four_byte_alphas) {
    rr_fread(ng->bo_weight4[0],sizeof(four_byte_t),ng->vocab_size+1,
	     ng->bin_fp,"unigram backoff weights",0);
  }
  else {
    rr_fread(ng->bo_weight[0],sizeof(bo_weight_t),ng->vocab_size+1,
	     ng->bin_fp,"unigram backoff weights",0);
  }

  if (ng->n > 1) {
    rr_fread(ng->ind[0],sizeof(index__t),ng->vocab_size+1,
	     ng->bin_fp,"unigram -> bigram pointers",0);
  }

  for(i=1;i<=ng->n-1;i++) {
    rr_fread(ng->word_id[i],sizeof(id__t),ng->num_kgrams[i],
	     ng->bin_fp,"word ids",0);
  }

  if (ng->four_byte_counts) {
    for(i=1;i<=ng->n-1;i++) {
      rr_fread(ng->count4[i],sizeof(int),ng->num_kgrams[i],
	       ng->bin_fp,"counts",0);
    }
  }
  else {
    for(i=1;i<=ng->n-1;i++) {
      rr_fread(ng->count[i],sizeof(count_ind_t),ng->num_kgrams[i],
	       ng->bin_fp,"counts",0);
    }
  }

  for(i=1;i<=ng->n-2;i++) {
    if (ng->four_byte_alphas) {
      rr_fread(ng->bo_weight4[i],sizeof(four_byte_t),ng->num_kgrams[i],
	       ng->bin_fp,"back off weights",0);
    }
    else {
      rr_fread(ng->bo_weight[i],sizeof(bo_weight_t),ng->num_kgrams[i],
	       ng->bin_fp,"back off weights",0);
    }
  }

  for(i=1;i<=ng->n-2;i++) {
    rr_fread(ng->ind[i],sizeof(index__t),ng->num_kgrams[i],
	     ng->bin_fp,"indices",0);
  }
  
  rr_oclose(ng->bin_fp);

}

void load_arpa_lm(arpa_lm_t *arpa_lm,
		  char *lm_filename) {

  /* Debugged by prc14 9th Nov 1997. Should now work for n-grams with
     n>3 with cutoffs >0 */

  FILE *arpa_fp;
  char *in_line;
  char *input_line;
  char temp_word[15][1024];
  int i,j,k;
  int num_of_args;
  int pos_of_novelty;
  char *input_line_ptr_orig;
  char *word_copy;
  id__t *previous_ngram;
  id__t *current_ngram;
  int temp_id;
  int *pos_in_list;
  int previd;
  flag first_one;

  in_line = (char *) rr_malloc(1024*sizeof(char));
  input_line = (char *) rr_malloc(1024*sizeof(char));

  input_line_ptr_orig = input_line;
  
  /* Attempt to parse an ARPA standard LM in a fairly robust way */

  /* Open file */

  arpa_fp = rr_iopen(lm_filename);

  /* Find start of data marker */
  
  while (strncmp("\\data\\",in_line,6)) {
    if (!rr_feof(arpa_fp)) {
      fgets(in_line,1024,arpa_fp);
    }
    else {
      quit(-1,"Error reading arpa language model file. Unexpected end of file.\n");
    }
  }

  
  /* Read number of each k-gram */

  arpa_lm->table_sizes = (int *) rr_malloc(sizeof(int)*11);
  arpa_lm->num_kgrams = (int *) rr_malloc(sizeof(int)*11);

  fgets(in_line,1024,arpa_fp);

  i = 0;

  while (strncmp("\\1-grams",in_line,8)) {
    if (sscanf(in_line,"%s %s",temp_word[1],temp_word[2]) == 2) {
      if (!strcmp("ngram",temp_word[1])) {
	i = temp_word[2][0]-48;
	arpa_lm->table_sizes[i-1]=atoi(&(temp_word[2][2]));
      }
    }

    fgets(in_line,1024,arpa_fp);

  }

  if (i==0) {
    quit(-1,"Error parsing ARPA format language model.\n");
  }

  arpa_lm->n = i;

  previous_ngram = (id__t *) rr_calloc(arpa_lm->n,sizeof(id__t));
  current_ngram = (id__t *) rr_calloc(arpa_lm->n,sizeof(id__t));

  printf("Reading in a %d-gram language model.\n",arpa_lm->n);
  for (i=0;i<=arpa_lm->n-1;i++) {
    printf("Number of %d-grams = %d.\n",i+1,arpa_lm->table_sizes[i]);
    arpa_lm->num_kgrams[i]=arpa_lm->table_sizes[i];
  }

  /* Allocate memory */

  pos_in_list = (int *) rr_malloc(sizeof(int) * arpa_lm->n);

  arpa_lm->word_id = (id__t **) rr_malloc(sizeof(id__t *) * arpa_lm->n);
  for (i=1;i<=arpa_lm->n-1;i++) { /* Don't allocate for i = 0 */
    arpa_lm->word_id[i] = (id__t *) rr_malloc(sizeof(id__t) * 
					     arpa_lm->table_sizes[i]);
  }

  arpa_lm->bo_weight = (bo_t **) rr_malloc(sizeof(bo_t *) * (arpa_lm->n-1));
  for (i=0;i<=arpa_lm->n-2;i++) {
    arpa_lm->bo_weight[i] = (bo_t *) rr_malloc(sizeof(bo_t) * 
					     arpa_lm->table_sizes[i]);
  }

  arpa_lm->ind = (index__t **) rr_malloc(sizeof(index__t *) * (arpa_lm->n-1));
  for (i=0;i<=arpa_lm->n-2;i++) {
    arpa_lm->ind[i] = (index__t *) rr_malloc(sizeof(index__t) * 
					   arpa_lm->table_sizes[i]);
  }

  arpa_lm->probs = (prob_t **) rr_malloc(sizeof(prob_t *) * arpa_lm->n);
  for (i=0;i<=arpa_lm->n-1;i++) {
    arpa_lm->probs[i] = (prob_t *) rr_malloc(sizeof(prob_t) * 
					     arpa_lm->table_sizes[i]);
  }

  arpa_lm->ptr_table = (int **) rr_malloc(sizeof(int *)*arpa_lm->n);
  arpa_lm->ptr_table_size = (unsigned short *) 
    rr_calloc(arpa_lm->n,sizeof(unsigned short));

  for (i=0;i<=arpa_lm->n-1;i++) {
    arpa_lm->ptr_table[i] = (int *) rr_calloc(65535,sizeof(int));
  }

  arpa_lm->vocab_ht = sih_create(1000,0.5,2.0,1);
  arpa_lm->vocab = (char **) rr_malloc(sizeof(char *)*
				       (arpa_lm->table_sizes[0]+1));
  arpa_lm->vocab_size = arpa_lm->table_sizes[0];

  /* Process 1-grams */

  printf("Reading unigrams...\n");
  
  i=0;

  fgets(in_line,1024,arpa_fp);
  
  if (arpa_lm->n > 1) {

    while (strncmp("\\2-grams",in_line,8)) {
      if (sscanf(in_line,"%f %s %f",&arpa_lm->probs[0][i],
		 temp_word[1],&arpa_lm->bo_weight[0][i]) == 3) {
	word_copy = salloc(temp_word[1]);
	
	/* Do checks about open or closed vocab */
	
	if (i==0) {
	  if (strcmp("<UNK>",word_copy)) {
	    
	    /* We have a closed vocabulary model */
	    
	    i++;
	    arpa_lm->vocab_type = CLOSED_VOCAB;
	    arpa_lm->first_id = 1;
	    
	  }
	  else {
	    
	    /* We have an open vocabulary model */
	    
	    arpa_lm->vocab_type = OPEN_VOCAB_1;
	    arpa_lm->first_id = 0;
	    arpa_lm->vocab_size--;
	    
	  }
	}
	
	arpa_lm->vocab[i] = word_copy;
	sih_add(arpa_lm->vocab_ht,word_copy,i);
	i++;
	if (((arpa_lm->vocab_type == OPEN_VOCAB_1) && 
	     (i>arpa_lm->table_sizes[0])) || 
	    ((arpa_lm->vocab_type == CLOSED_VOCAB) &&
	     (i>arpa_lm->table_sizes[0]+1))){
	  quit(-1,"Error - Header information in ARPA format language model is incorrect.\nMore than %d unigrams needed to be stored.\n",arpa_lm->table_sizes[0]);
	}
      }
      else {
	if (strlen(in_line)>1) {
	  fprintf(stderr,"Warning, reading line -%s- gave unexpected input.\n",in_line);
	}
      }
      fgets(in_line,1024,arpa_fp);
      
    }
  }
  else {

    while (strncmp("\\end\\",in_line,5)) {
      if (sscanf(in_line,"%f %s",&arpa_lm->probs[0][i],
		 temp_word[1]) == 2) {
	word_copy = salloc(temp_word[1]);
	
	/* Do checks about open or closed vocab */
	
	if (i==0) {
	  if (strcmp("<UNK>",word_copy)) {
	    
	    /* We have a closed vocabulary model */
	    
	    i++;
	    arpa_lm->vocab_type = CLOSED_VOCAB;
	    arpa_lm->first_id = 1;
	    
	  }
	  else {
	    
	    /* We have an open vocabulary model */
	    
	    arpa_lm->vocab_type = OPEN_VOCAB_1;
	    arpa_lm->first_id = 0;
	    arpa_lm->vocab_size--;
	    
	  }
	}
	
	arpa_lm->vocab[i] = word_copy;
	sih_add(arpa_lm->vocab_ht,word_copy,i);
	i++;
	if (((arpa_lm->vocab_type == OPEN_VOCAB_1) && 
	     (i>arpa_lm->table_sizes[0])) || 
	    ((arpa_lm->vocab_type == CLOSED_VOCAB) &&
	     (i>arpa_lm->table_sizes[0]+1))){
	  quit(-1,"Error - Header information in ARPA format language model is incorrect.\nMore than %d unigrams needed to be stored.\n",arpa_lm->table_sizes[0]);
	}
      }
      else {
	if (strlen(in_line)>1) {
	  fprintf(stderr,"Warning, reading line -%s- gave unexpected input.\n",in_line);
	}
      }
      fgets(in_line,1024,arpa_fp);
      
    }
  }
    
  if (arpa_lm->n > 1) {

    /* Process 2, ... , n-1 grams */

    previd = -1;

    for (i=2;i<=arpa_lm->n-1;i++) {

      printf("\nReading %d-grams...\n",i);

      previd = -1;

      j=0;

      for (k=0;k<=arpa_lm->n-1;k++) {
	pos_in_list[0] = 0;
      }
				
      sprintf(temp_word[14],"\\%d-grams",i+1);
      first_one=1;
      while (strncmp(temp_word[14],temp_word[0],8)) {
      
	/* Process line into all relevant temp_words */

	num_of_args = 0;

	for (k=0;k<=i+1;k++) {
	  if (strncmp(temp_word[0],temp_word[14],8)) {
	    fscanf(arpa_fp,"%s",temp_word[k]);
	  }
	}
  
	if (strncmp(temp_word[0],temp_word[14],8)) {

	  arpa_lm->probs[i-1][j] = (prob_t) atof(temp_word[0]);
	  arpa_lm->bo_weight[i-1][j] = (bo_t) atof(temp_word[i+1]);
	
	  sih_lookup(arpa_lm->vocab_ht,temp_word[i],&temp_id);
	  arpa_lm->word_id[i-1][j] = temp_id;
	
	  if (j % 20000 == 0) {
	    if (j % 1000000 == 0) {
	      if (j != 0) {
		fprintf(stderr,".\n");
	      }
	    }
	    else {
	      fprintf(stderr,".");
	    }
	  }
	
	  j++;
	  if (j>arpa_lm->table_sizes[i-1]) {
	    quit(-1,"Error - Header information in ARPA format language model is incorrect.\nMore than %d %d-grams needed to be stored.\n",arpa_lm->table_sizes[i-1],i);
	  }

	  /* Make sure that indexes in previous table point to 
	     the right thing. */

	  for (k=0;k<=i-1;k++) {
	    previous_ngram[k] = current_ngram[k];
	    sih_lookup(arpa_lm->vocab_ht,temp_word[k+1],&temp_id);
	    if (temp_id == 0 && strcmp(temp_word[k+1],"<UNK>")) {
	      quit(-1,"Error - found unknown word in n-gram file : %s\n",
		   temp_word[k+1]);
	    }
	    current_ngram[k] = temp_id;
	  }

	  /* Find position of novelty */

	  if (first_one) {
	    pos_of_novelty = 0;
	    first_one = 0;
	  }
	  else {
      
	    pos_of_novelty = i;


	    for (k=0;k<=i-1;k++) {
	      if (current_ngram[k] > previous_ngram[k]) {
		pos_of_novelty = k;
		k = arpa_lm->n;
	      }
	      else {
		if ((current_ngram[k] > previous_ngram[k]) && (j > 0)) {
		  quit(-1,"Error : n-grams are not correctly ordered.\n");
		}
	      }
	    }

	    if (pos_of_novelty > i) {
	      fprintf(stderr,"pos of novelty 2 = %d\n",pos_of_novelty);
	    }


	    if (pos_of_novelty == i && j != 1) {
	      quit(-1,"Error - Repeated %d-gram in ARPA format language model.\n",
		   i);
	    }
	  }

	  /* If pos of novelty = i-1 then we are at the same i-1 gram
             as before, and so it will be pointing to the right
             thing. */
	
	  if (pos_of_novelty != i-1) {
	    if (i==2) {
	      /* Deal with unigram pointers */

	      for (k = previd + 1; k <= current_ngram[0]; k++) {
		arpa_lm->ind[0][k] = new_index(j-1,
					       arpa_lm->ptr_table[0],
					       &(arpa_lm->ptr_table_size[0]),
					       k);
	      }
	      previd = current_ngram[0];
	    }
	    else {


	      /* Find the appropriate place in the (i-2) table */

	      /*	      for (k=pos_of_novelty;k<=i-2;k++) { */
	      for (k=0;k<=i-2;k++) {

		/* Find appropriate place in the kth table */

		if (k == 0) {
		  pos_in_list[0] = current_ngram[0];
		}
		else {
		  pos_in_list[k] = get_full_index(arpa_lm->ind[k-1][pos_in_list[k-1]],
						  arpa_lm->ptr_table[k-1],   
						  arpa_lm->ptr_table_size[k-1],   
						  pos_in_list[k-1]);
		  while (arpa_lm->word_id[k][pos_in_list[k]] < 
			 current_ngram[k]) {
		    pos_in_list[k]++;

		  }
		  if (arpa_lm->word_id[k][pos_in_list[k]] != 
		      current_ngram[k]) {

		    quit(-1,"Error in the ARPA format language model. \nA %d-gram exists, but not the stem %d-gram.",k+2,k+1);
		  }
		}
	      }
	      for (k = previd + 1; k <= pos_in_list[i-2]; k++) {
		
		arpa_lm->ind[i-2][k] = 
		  new_index(j-1,
			    arpa_lm->ptr_table[i-2],
			    &(arpa_lm->ptr_table_size[i-2]),
			    k);

	      }
	      previd = pos_in_list[i-2];	
	    }
	  }
	}
      }

      /* Now need to tidy up pointers for bottom section of (i-1)-grams */

      if (i==2) {

	for (k = previd + 1; k <= arpa_lm->vocab_size; k++) {
	  
	  arpa_lm->ind[0][k] = new_index(arpa_lm->num_kgrams[1],
					 arpa_lm->ptr_table[0],
					 &(arpa_lm->ptr_table_size[0]),
					 k);
	}      
      }
      else {
	for (k = previd + 1; k <= arpa_lm->num_kgrams[i-2]-1;k++) {
	  arpa_lm->ind[i-2][k] = new_index(j,
					   arpa_lm->ptr_table[i-2],
					   &(arpa_lm->ptr_table_size[i-2]),
					   k);
	}
      }
					   
					   
    }



    printf("\nReading %d-grams...\n",arpa_lm->n);
    
    first_one = 1;
    j = 0;
    previd = 0;
    
    arpa_lm->ind[arpa_lm->n-2][0] = 0;

    for (k=0;k<=arpa_lm->n-1;k++) {
      pos_in_list[0] = 0;
    }
  
    while (strncmp("\\end\\",temp_word[0],5)) {
    
      /* Process line into all relevant temp_words */

      for (k=0;k<=arpa_lm->n;k++) {
	if (strncmp(temp_word[0],"\\end\\",5)) {
	  fscanf(arpa_fp,"%s",temp_word[k]);
	}
      }
    
      if (strncmp(temp_word[0],"\\end\\",5)) {
      
	if (j % 20000 == 0) {
	  if (j % 1000000 == 0) {
	    if (j != 0) {
	      fprintf(stderr,".\n");
	    }
	  }
	  else {
	    fprintf(stderr,".");
	  }
	}
      
	arpa_lm->probs[arpa_lm->n-1][j] = atof(temp_word[0]);
	sih_lookup(arpa_lm->vocab_ht,temp_word[arpa_lm->n],&temp_id);
      
	arpa_lm->word_id[arpa_lm->n-1][j] = temp_id;
      
	j++;
      
	for (k=0;k<=arpa_lm->n-1;k++) {
	  previous_ngram[k] = current_ngram[k];
	  sih_lookup(arpa_lm->vocab_ht,temp_word[k+1],&temp_id);
	  if (temp_id == 0 && strcmp(temp_word[k+1],"<UNK>")) {
	    quit(-1,"Error - found unknown word in n-gram file : %s\n",
		 temp_word[k+1]);
	  }
	  current_ngram[k] = temp_id;
	}
      
	/* Find position of novelty */
	
	if (first_one) {
	  pos_of_novelty = 0;
	  first_one = 0;
	}
	else {
      
	  pos_of_novelty = arpa_lm->n+1;

	  for (k=0;k<=arpa_lm->n-1;k++) {
	    if (current_ngram[k] > previous_ngram[k]) {
	      pos_of_novelty = k;
	      k = arpa_lm->n;
	    }
	    else {
	      if ((current_ngram[k] > previous_ngram[k]) && (j>0)) {
		quit(-1,"Error : n-grams are not correctly ordered.\n");
	      }
	    }
	  }
      
	  if ( pos_of_novelty == arpa_lm->n+1 && j != 1 ) {
	    quit(-1,"Error : Same %d-gram occurs twice in ARPA format LM.\n",
		 arpa_lm->n);
	  }
	}
	if (pos_of_novelty != arpa_lm->n-1) {
	
	  /*	  for (k=pos_of_novelty;k<=arpa_lm->n-2;k++) { */

	  for (k=0;k<=arpa_lm->n-2;k++) {

	    if (k == 0) {
	      pos_in_list[0] = current_ngram[0];
	    }
	    else {
	      pos_in_list[k] = get_full_index(arpa_lm->ind[k-1][pos_in_list[k-1]],
					      arpa_lm->ptr_table[k-1],   
					      arpa_lm->ptr_table_size[k-1],   
					      pos_in_list[k-1]);
	      while (arpa_lm->word_id[k][pos_in_list[k]] < 
		     current_ngram[k]) {
		pos_in_list[k]++;
	      }
	      
	      if (arpa_lm->word_id[k][pos_in_list[k]] != current_ngram[k]) {
		quit(-1,"Error in the ARPA format language model. \nA %d-gram exists, but not the stem %d-gram.",k+2,k+1);
	      }
	    }
	  }
	  for (k = previd + 1; k <= pos_in_list[arpa_lm->n-2]; k++) {

	    arpa_lm->ind[arpa_lm->n-2][k] = 
	      new_index(j-1,
			arpa_lm->ptr_table[arpa_lm->n-2],
			&(arpa_lm->ptr_table_size[arpa_lm->n-2]),
			k);
	  }
	  previd = pos_in_list[arpa_lm->n-2];
	}
	
	if (j>arpa_lm->table_sizes[arpa_lm->n-1]) {
	  quit(-1,"Error - Header information in ARPA format language model is incorrect.\nMore than %d %d-grams needed to be stored.\n",arpa_lm->table_sizes[arpa_lm->n-1],arpa_lm->n-1);
	}
      }
    }

    /* Tidy up bottom section */

    for (k = previd + 1; k <= arpa_lm->num_kgrams[arpa_lm->n-2]; k++) {
      arpa_lm->ind[arpa_lm->n-2][k] = 
	new_index(j,
		  arpa_lm->ptr_table[i-2],
		  &(arpa_lm->ptr_table_size[i-2]),
		  k);
    }
  
  }

  /* Tidy up */


  free(previous_ngram);
  free(current_ngram);
  free(in_line);
  free(input_line);
  rr_iclose(arpa_fp);

}










