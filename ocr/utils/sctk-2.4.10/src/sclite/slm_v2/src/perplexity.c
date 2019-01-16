
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
#include <math.h>
#include <stdlib.h>
#include <string.h>

void compute_perplexity(ng_t *ng,
			arpa_lm_t *arpa_ng,
			char *text_stream_filename,
			char *probs_stream_filename,
			char *annotation_filename,
			char *oov_filename,
			char *fb_list_filename,
			flag backoff_from_unk_inc,
			flag backoff_from_unk_exc,
			flag backoff_from_ccs_inc,
			flag backoff_from_ccs_exc,
			flag arpa_lm,
			flag include_unks,
			double log_base) {

  fb_info *fb_list;
  FILE *temp_fp;
  FILE *text_stream_fp;
  FILE *probs_stream_fp;
  FILE *annotation_fp;
  FILE *oov_fp;
  flag out_probs;
  flag annotate;
  flag out_oovs;
  flag found_unk_wrongly;
  double prob;
  double sum_log_prob;
  int total_words;
  int excluded_unks;
  int excluded_ccs;
  char current_word[1000];  /* Hope that's big enough */
  char **prev_words;
  int current_id;
  id__t short_current_id;
  id__t *context;
  int context_length;
  int i;
  int bo_case;
  int actual_context_length;
  int *ngrams_hit;
  int n;

  /* Initialise file pointers to prevent warnings from the compiler. */

  probs_stream_fp = NULL;
  annotation_fp = NULL;
  oov_fp = NULL;

  short_current_id = 0;

  found_unk_wrongly = 0;

  annotate = 0;

  bo_case = 0;

  if (arpa_lm) {
    n = arpa_ng->n;
    fb_list = gen_fb_list(arpa_ng->vocab_ht,
			  arpa_ng->vocab_size,
			  arpa_ng->vocab,
			  arpa_ng->context_cue,
			  backoff_from_unk_inc,
			  backoff_from_unk_exc,
			  backoff_from_ccs_inc,
			  backoff_from_ccs_exc,
			  fb_list_filename);
  }
  else {
    n = ng->n;
    fb_list = gen_fb_list(ng->vocab_ht,
			  ng->vocab_size,
			  ng->vocab,
			  ng->context_cue,
			  backoff_from_unk_inc,
			  backoff_from_unk_exc,
			  backoff_from_ccs_inc,
			  backoff_from_ccs_exc,
			  fb_list_filename);
  }
  
  ngrams_hit = (int *) rr_calloc(n,sizeof(int));
  prev_words = (char **) rr_malloc(sizeof(char *)*n);
  for (i=0;i<=n-1;i++) {
    prev_words[i] = (char *) rr_malloc(sizeof(char)*1000);
  }
  
  /* Check that text_stream_filename and probs_stream_filename (if
     specified) are valid. Note that the checks employed by the
     standard rr_fopen tools are not suitable here, since we don't
     want the program to terminate if the paths are not found. */

  if (!strcmp(text_stream_filename,"")) {
    printf("Error : Must specify a text file. Use the -text switch.\n");
    return;
  }

  if (!rr_fexists(text_stream_filename) && strcmp(text_stream_filename,"-")) {
    printf("Error : Can't open file %s for reading.\n",text_stream_filename);
    return;
  }

  out_probs = strcmp(probs_stream_filename,"");
  annotate = strcmp(annotation_filename,"");
  out_oovs = strcmp(oov_filename,"");

  printf("Computing perplexity of the language model with respect\n");
  printf("   to the text %s\n",text_stream_filename);
  if (out_probs) {
    printf("Probability stream will be written to file %s\n",
	    probs_stream_filename);
  }
  if (annotate) {
    printf("Annotation will be written to file %s\n",
	    annotation_filename);
  }
  if (out_oovs) {
    printf("Out of vocabulary words will be written to file %s\n",
	    oov_filename);
  }

  if (backoff_from_unk_inc) {
    printf("Will force inclusive back-off from OOVs.\n");
  }

  if (backoff_from_unk_exc) {
    printf("Will force exclusive back-off from OOVs.\n");
  }

  if (backoff_from_ccs_inc) {
    printf("Will force inclusive back-off from context cues.\n");
  }

  if (backoff_from_ccs_exc) {
    printf("Will force exclusive back-off from context cues.\n");
  }

  if (strcmp(fb_list_filename,"")) {
    printf("Will force back-off according to the contents of %s\n",
	    fb_list_filename);
  }

  if (include_unks) {
    printf("Perplexity calculation will include OOVs.\n");
  }

  /* Check for existance of files, as rr functions will quit, which isn't
     what we want */

  if (out_probs && strcmp(probs_stream_filename,"-")) {
    if ((temp_fp = fopen(probs_stream_filename,"w")) == NULL) {
      printf("Error : Can't open file %s for writing.\n",probs_stream_filename);
      return;
    }
    fclose(temp_fp);
  }

  if (annotate && strcmp(annotation_filename,"-")) {
    if ((temp_fp = fopen(annotation_filename,"w")) == NULL) {
      printf("Error : Can't open file %s for writing.\n",annotation_filename);
      return;
    }
    fclose(temp_fp);
  }
    
  if (out_oovs && strcmp(oov_filename,"-")) {
    if ((temp_fp = fopen(oov_filename,"w")) == NULL) {
      printf("Error : Can't open file %s for writing.\n",oov_filename);
      return;
    }
    fclose(temp_fp);
  }

  text_stream_fp = rr_iopen(text_stream_filename);
  if (out_probs) {
    probs_stream_fp = rr_oopen(probs_stream_filename);
  }

  if (annotate) {
    annotation_fp = rr_oopen(annotation_filename);
  }

  if (out_oovs) {
    oov_fp = rr_oopen(oov_filename);
  }

  context = (id__t *) rr_malloc(sizeof(id__t)*(n-1));

  sum_log_prob = 0.0;
  total_words = 0;
  excluded_unks = 0;
  excluded_ccs = 0;

  while (!rr_feof(text_stream_fp)) {

    if (total_words > 0) {
      if (total_words < n) {
	strcpy(prev_words[total_words-1],current_word);
      }
      else {
	for (i=0;i<=n-3;i++) {
	  strcpy(prev_words[i],prev_words[i+1]);
	}
	if (n>1) {
	  strcpy(prev_words[n-2],current_word);
	}
      }
    }

    if (total_words < (n-1)) {
      context_length = total_words;
    }
    else {
      context_length = n-1;
    }

    /* Fill context with right stuff */

    if (total_words > (n-1)) {

      for (i=0;i<=context_length-2;i++) {
	context[i] = context[i+1];
      }
      
    }

    if (context_length != 0){
      context[context_length-1] = short_current_id;
    }

    if (fscanf(text_stream_fp,"%s",current_word) != 1) {
      if (!rr_feof(text_stream_fp)) {
	printf("Error reading text file.\n");
	return;
      }
    }

    if (!rr_feof(text_stream_fp)) {

      if (arpa_lm) {
	sih_lookup(arpa_ng->vocab_ht,current_word,&current_id);
	if (arpa_ng->vocab_type == CLOSED_VOCAB && current_id == 0) {
	  found_unk_wrongly = 1;
	  printf("Error : %s is not in the vocabulary, and this is a closed \nvocabulary model.\n",current_word);
	}
	if (current_id > arpa_ng->vocab_size) {
	  quit(-1,"Error : returned value from sih_lookup (%d) is too high.\n",
	       context[i]); 
	}
	else {
	  short_current_id = current_id;
	}
      }
      else {
	sih_lookup(ng->vocab_ht,current_word,&current_id);
	if (ng->vocab_type == CLOSED_VOCAB && current_id == 0) {
	  found_unk_wrongly = 1;
	  printf("Error : %s is not in the vocabulary, and this is a closed \nvocabulary model.\n",current_word);
	}
	if (current_id > ng->vocab_size) {
	  quit(-1,"Error : returned value from sih_lookup (%d) is too high.\n",context[i]); 
	}
	else {
	  short_current_id = current_id;
	}
      }
    
      if (!found_unk_wrongly) {

	if (current_id == 0 && out_oovs) {
	  fprintf(oov_fp,"%s\n",current_word);
	}

	if ((arpa_lm && (!(arpa_ng->context_cue[current_id])))
	    || ((!arpa_lm) && (!(ng->context_cue[current_id])))) {

	  if (include_unks || current_id != 0) {

	    prob = calc_prob_of(short_current_id,
				context,
				context_length,
				ng,
				arpa_ng,
				fb_list,
				&bo_case,
				&actual_context_length,
				arpa_lm);


	    if (prob<= 0.0 || prob > 1.0) {
	      fprintf(stderr,"Warning : ");
	      if (short_current_id == 0){
		fprintf(stderr,"P( <UNK> | ");
	      }
	      else {
		fprintf(stderr,"P( %s | ",current_word);
	      }
	  
	      for (i=0;i<=actual_context_length-1;i++) {
		if (context[i+context_length-actual_context_length] == 0) {
		  fprintf(stderr,"<UNK> ");
		}
		else {
		  fprintf(stderr,"%s ",prev_words[i]);
		}
	      }
	      fprintf(stderr,") = %g logprob = %g \n ",prob,log(prob)/log(log_base));
	      fprintf(stderr,"bo_case == 0x%dx, actual_context_length == %d\n",
		      bo_case, actual_context_length);
	    }
	  
	    if (annotate) {
	      if (short_current_id == 0){
		fprintf(annotation_fp,"P( <UNK> | ");
	      }
	      else {
		fprintf(annotation_fp,"P( %s | ",current_word);
	      }
	  
	      for (i=0;i<=actual_context_length-1;i++) {
		if (context[i+context_length-actual_context_length] == 0) {
		  fprintf(annotation_fp,"<UNK> ");
		}
		else {
		  if (arpa_lm) {
		    fprintf(annotation_fp,"%s ",arpa_ng->vocab[context[i+context_length-actual_context_length]]);
		  }
		  else {
		    fprintf(annotation_fp,"%s ",ng->vocab[context[i+context_length-actual_context_length]]);
		  }
		}
	      }
	      fprintf(annotation_fp,") = %g logprob = %f bo_case = ",prob,log(prob)/log(log_base));
	      decode_bo_case(bo_case,actual_context_length,annotation_fp);
	    }

	    /* Calculate level to which we backed off */


  
	    for (i=actual_context_length-1;i>=0;i--) {
 	      int four_raise_i = 1<<(2*i);  /* PWP */
 
 	      /*
 	       * PWP: This was "if ((bo_case / (int) pow(3,i)) == 0)"
 	       * but was getting a divide-by-zero error on an Alpha
 	       * (it isn't clear to me why it should ever have done so)
 	       * Anyway, it is much faster to do in base-4.
 	       */

	      if ((bo_case == 0) || ((bo_case / four_raise_i) == 0)) {
		ngrams_hit[i+1]++;
		i = -2;
	      }
	      else {
		bo_case -= ((bo_case / four_raise_i) * four_raise_i);
	      }
	    }
  
	    if (i != -3) { 
	      ngrams_hit[0]++;
	    }

	    if (out_probs) {
	      fprintf(probs_stream_fp,"%g\n",prob);
	    }
      
	    sum_log_prob += log10(prob);
			  
	  }

          if (current_id == 0 && !include_unks) {
            excluded_unks++;
          }


	}       
	else {
	  if (((!arpa_lm) && ng->context_cue[current_id]) || 
	      (arpa_lm && arpa_ng->context_cue[current_id])) {
	    excluded_ccs++;
	  }
	}
	total_words++;
      }
    }

  }
  if (!found_unk_wrongly) {

     /*  pow(x,y) = e**(y  ln(x)) */
     printf("Perplexity = %.2f, Entropy = %.2f bits\n", 
	    exp(-sum_log_prob/(total_words-excluded_ccs-excluded_unks) * 
		log(10.0)),
	    (-sum_log_prob/(total_words-excluded_ccs-excluded_unks) * 
	     log(10.0) / log(2.0)));

    
    printf("Computation based on %d words.\n",
	   total_words-excluded_ccs-excluded_unks);
    for(i=n;i>=1;i--) {
      printf("Number of %d-grams hit = %d  (%.2f%%)\n",i,ngrams_hit[i-1],
	     (float) 100*ngrams_hit[i-1]/(total_words-excluded_ccs-excluded_unks) );
    }
    printf("%d OOVs (%.2f%%) and %d context cues were removed from the calculation.\n",
	   excluded_unks,
	   (float) 100*excluded_unks/(total_words-excluded_ccs),excluded_ccs);
    
  }

  rr_iclose(text_stream_fp);

  if (out_probs) {
    rr_oclose(probs_stream_fp);
  }
  if (annotate) {
    rr_oclose(annotation_fp);
  }
  if (out_oovs) {
    rr_oclose(oov_fp);
  }

  free (fb_list);
  free (context);
  free (ngrams_hit);
}
