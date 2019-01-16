
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

#include "toolkit.h"
#include "ngram.h"
#include "pc_libs/pc_general.h"
#include "idngram2lm.h"
#include "rr_libs/sih.h"
#include "rr_libs/general.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void main (int argc, char **argv) {

  int i,j;
  ng_t ng;
  int verbosity;
  int mem_alloc_method; /* Method used to decide how much memory to 
			   allocate for count tables */
  int buffer_size;
  flag is_ascii;
  flag ascii_flag;
  flag bin_flag;
  char current_cc[200];
  int current_cc_id;
  int test_cc_id;
  ngram current_ngram;
  ngram previous_ngram;
  int *ng_count; /* Array indicating the number of occurrances of 
		    the current 1-gram, 2-gram, ... ,n-gram */  
  int nlines;
  int pos_of_novelty;
  int prev_id1;
  flag contains_unks;
  int mem_alloced;
  char wlist_entry[1024];
  int end_size;
  int middle_size;

  flag displayed_oov_warning;

  flag context_set;
  flag oov_frac_set;
  flag disc_range_set;

  /*  ------------------  Process command line --------------------- */

  report_version(&argc,argv);

  if (argc == 1 || pc_flagarg(&argc, argv,"-help")) {
    
    /* Display help message */

    fprintf(stderr,"idngram2lm : Convert an idngram file to a language model file.\n");
    fprintf(stderr,"Usage : \n");
    fprintf(stderr,"idngram2lm -idngram .idngram\n");
    fprintf(stderr,"           -vocab .vocab\n");
    fprintf(stderr,"           -arpa .arpa | -binary .binlm\n");
    fprintf(stderr,"         [ -context .ccs ]\n");
    fprintf(stderr,"         [ -calc_mem | -buffer 100 | -spec_num y ... z ]\n");
    fprintf(stderr,"         [ -vocab_type 1 ]\n");
    fprintf(stderr,"         [ -oov_fraction 0.5 ]\n");
    fprintf(stderr,"         [ -two_byte_bo_weights   \n              [ -min_bo_weight nnnnn] [ -max_bo_weight nnnnn] [ -out_of_range_bo_weights] ]\n");
    fprintf(stderr,"         [ -four_byte_counts ]\n");
    fprintf(stderr,"         [ -linear | -absolute | -good_turing | -witten_bell ]\n");
    fprintf(stderr,"         [ -disc_ranges 1 7 7 ]\n");
    fprintf(stderr,"         [ -cutoffs 0 ... 0 ]\n");
    fprintf(stderr,"         [ -min_unicount 0 ]\n");
    fprintf(stderr,"         [ -zeroton_fraction ]\n");
    fprintf(stderr,"         [ -ascii_input | -bin_input ]\n");
    fprintf(stderr,"         [ -n 3 ]  \n");
    fprintf(stderr,"         [ -verbosity %d ]\n",DEFAULT_VERBOSITY);
    exit(1);
  }

  verbosity = pc_intarg(&argc, argv,"-verbosity",DEFAULT_VERBOSITY);
  ng.n = pc_intarg(&argc, argv,"-n",DEFAULT_N);

  if (ng.n<1) {
    quit(-1,"Error: Value of n must be positive.\n");
  }

  ng.cutoffs = (cutoff_t *) pc_shortarrayarg(&argc, argv, "-cutoffs",ng.n-1,ng.n-1);

  if (ng.cutoffs == NULL) {
    ng.cutoffs = (cutoff_t *) rr_calloc(ng.n-1,sizeof(cutoff_t));
  }

  for (i=0;i<=ng.n-3;i++) {
    if (ng.cutoffs[i+1] < ng.cutoffs[i]) {
      quit(-1,"Error - cutoffs for (n+1)-gram must be greater than or equal to those for \nn-gram. You have %d-gram cutoff = %d > %d-gram cutoff = %d.\n",i+2,ng.cutoffs[i],i+3,ng.cutoffs[i+1]);
    }
  }

  mem_alloc_method = 0;

  if (pc_flagarg(&argc, argv,"-calc_mem")) {
    mem_alloc_method = TWO_PASSES;
  }
  
  buffer_size = pc_intarg(&argc, argv,"-buffer",-1);

  if (buffer_size != -1) {
    if (mem_alloc_method != 0) {
      quit(-1,"Assigned two contradictory methods of memory allocation.\n Use one of -calc_mem, -buffer, or -spec_num.\n");
    }
    mem_alloc_method = BUFFER;
  }
 
  ng.table_sizes = pc_intarrayarg(&argc, argv, "-spec_num",ng.n-1,ng.n);

  if (ng.table_sizes != NULL) {

    if (mem_alloc_method != 0) {
      quit(-1,"Assigned two contradictory methods of memory allocation.\n Use one of -calc_mem, -guess, or -spec_num.\n");
    }
    mem_alloc_method = SPECIFIED;
    for (i=ng.n-1;i>=1;i--) {
      ng.table_sizes[i] = ng.table_sizes[i-1];
    }
  }

  if (mem_alloc_method == 0) {
    mem_alloc_method = BUFFER;
    buffer_size = STD_MEM;
  }
  
  ng.min_unicount = pc_intarg(&argc, argv, "-min_unicount",0);

  ng.id_gram_filename = salloc(pc_stringarg(&argc, argv,"-idngram",""));

  if (!strcmp(ng.id_gram_filename,"")) {
    quit(-1,"Error: id ngram file not specified. Use the -idngram flag.\n");
  }

  if (!strcmp(ng.id_gram_filename,"-") && mem_alloc_method == TWO_PASSES) {
    quit(-1,"Error: If idngram is read from stdin, then cannot use -calc_mem option.\n");
  }

  is_ascii = 0;

  ascii_flag = pc_flagarg(&argc,argv,"-ascii_input");
  bin_flag = pc_flagarg(&argc,argv,"-bin_input");

  if (ascii_flag || 
      !strcmp(&ng.id_gram_filename[strlen(ng.id_gram_filename)-6],".ascii")) {
    is_ascii = 1;
  }
  else {
  }
  
  if (ascii_flag) {
    
    if (bin_flag) {
      quit(-1,"Error : Specify only one of -bin_input and -ascii_input\n");
    }
    
    if (!strcmp(&ng.id_gram_filename[strlen(ng.id_gram_filename)-4],".bin")) {
      quit(-1,"Error : -ascii_input flag specified, but input file has .bin extention.\n");
    }
    
  }
  
  if (bin_flag && 
      !strcmp(&ng.id_gram_filename[strlen(ng.id_gram_filename)-6],".ascii") ) {
    quit(-1,"Error : -bin_input flag specified, but input file has .ascii extention.\n");
  }
  
  ng.arpa_filename = salloc(pc_stringarg(&argc, argv,"-arpa",""));
  ng.bin_filename = salloc(pc_stringarg(&argc, argv,"-binary",""));
  
  ng.write_arpa = strcmp("",ng.arpa_filename);
  ng.write_bin = strcmp("",ng.bin_filename);
  
  if (!(ng.write_arpa || ng.write_bin)) {
    quit(-1,"Error : must specify either an arpa, or a binary output file.\n");
  }
  
  ng.count_table_size = DEFAULT_COUNT_TABLE_SIZE;
  
  ng.vocab_filename = salloc(pc_stringarg(&argc,argv,"-vocab",""));
  


  if (!strcmp("",ng.vocab_filename)) {
    quit(-1,"Error : vocabulary file not specified. Use the -vocab option.\n");
  }


  ng.context_cues_filename = salloc(pc_stringarg(&argc,argv,"-context",""));

  context_set = strcmp("", ng.context_cues_filename);

  ng.vocab_type = pc_intarg(&argc,argv,"-vocab_type",1);
  
  ng.oov_fraction = pc_doublearg(&argc, argv,"-oov_fraction",-1.0);

  if (ng.oov_fraction == -1.0) {
    oov_frac_set = 0;    
    ng.oov_fraction=DEFAULT_OOV_FRACTION;
  }
  else {
    oov_frac_set = 1;
    if (ng.vocab_type != 2) {
      pc_message(verbosity,1,"Warning : OOV fraction specified, but will not be used, since vocab type is not 2.\n");
    }
  }

  if (ng.vocab_type == 0) {
    ng.first_id = 1;
  }
  else {
    ng.first_id = 0;
  }

  /* Allow both "min_alpha" etc and "min_bo_weight" etc as valid
     syntax. The "bo_weight" form is preferred, but the "alpha" form is
     maintained as it was present in version 2.00 */

  ng.min_alpha = pc_doublearg(&argc,argv,"-min_alpha",DEFAULT_MIN_ALPHA);
  ng.max_alpha = pc_doublearg(&argc,argv,"-max_alpha",DEFAULT_MAX_ALPHA);
  ng.out_of_range_alphas = pc_intarg(&argc,argv,"-out_of_range_alphas",
				     DEFAULT_OUT_OF_RANGE_ALPHAS);

  ng.min_alpha = pc_doublearg(&argc,argv,"-min_bo_weight",ng.min_alpha);
  ng.max_alpha = pc_doublearg(&argc,argv,"-max_bo_weight",ng.max_alpha);
  ng.out_of_range_alphas = pc_intarg(&argc,argv,"-out_of_range_bo_weights",
				     ng.out_of_range_alphas);


  
  if (ng.min_alpha >= ng.max_alpha) {
    quit(-1,"Error : Minimum of alpha range must be less than the maximum.\n");
  }

  ng.discounting_method = 0;
  
  if (pc_flagarg(&argc, argv,"-linear")) {
    ng.discounting_method = LINEAR;
  }

  if (pc_flagarg(&argc,argv,"-absolute")) {
    if (ng.discounting_method != 0) {
      quit(-1,"Error : Assigned two contradictory discounting methods.\nSpecify one of -linear, -absolute, -good_turing or -witten_bell.\n");
    }
    ng.discounting_method = ABSOLUTE;
  }

  if (pc_flagarg(&argc,argv,"-witten_bell")) {
    if (ng.discounting_method != 0) {
      quit(-1,"Error : Assigned two contradictory discounting methods.\nSpecify one of -linear, -absolute, -good_turing or -witten_bell.\n");
    }
    ng.discounting_method = WITTEN_BELL;
  }

  if (pc_flagarg(&argc,argv,"-good_turing")) {
    if (ng.discounting_method != 0) {
      quit(-1,"Error : Assigned two contradictory discounting methods.\nSpecify one of -linear, -absolute, -good_turing or -witten_bell.\n");
    }
    ng.discounting_method = GOOD_TURING;
  }

  if (ng.discounting_method == 0) {
    ng.discounting_method = GOOD_TURING;
  }

  ng.disc_range = (unsigned short *) pc_shortarrayarg(&argc, argv, "-disc_ranges",ng.n,ng.n);

  disc_range_set = (ng.disc_range != NULL);


  if (ng.discounting_method == GOOD_TURING) {
    if (!disc_range_set) {
      ng.disc_range = (unsigned short *) rr_malloc(sizeof(unsigned short) * ng.n);
      ng.disc_range[0] = DEFAULT_DISC_RANGE_1;
      for (i=1;i<=ng.n-1;i++) {
	ng.disc_range[i] = DEFAULT_DISC_RANGE_REST;
      }
    }
    ng.fof_size = (unsigned short *) rr_malloc(sizeof(unsigned short) * ng.n);
    for (i=0;i<=ng.n-1;i++) {
      ng.fof_size[i] = ng.disc_range[i]+1;
    }
  }
  else {
    if (disc_range_set) {
      pc_message(verbosity,2,"Warning : discount ranges specified will be ignored, since they only apply\nto Good Turing discounting.\n");
    }
  }

  ng.four_byte_alphas = !(pc_flagarg(&argc, argv, "-two_byte_alphas") || 
			  pc_flagarg(&argc, argv, "-two_byte_bo_weights"));

  ng.four_byte_counts = pc_flagarg(&argc, argv, "-four_byte_counts");

  ng.zeroton_fraction = pc_doublearg(&argc,argv,"-zeroton_fraction",1.0);

  /* Report parameters */

  pc_message(verbosity,2,"  n : %d\n",ng.n);
  pc_message(verbosity,2,"  Input file : %s",ng.id_gram_filename);
  if (is_ascii) {
    pc_message(verbosity,2,"     (ascii format)\n");
  }
  else {
    pc_message(verbosity,2,"     (binary format)\n");
  }
  pc_message(verbosity,2,"  Output files :\n");
  if (ng.write_arpa) {
    pc_message(verbosity,2,"     ARPA format   : %s\n",ng.arpa_filename);
  }
  if (ng.write_bin) {
    pc_message(verbosity,2,"     Binary format : %s\n",ng.bin_filename);
  }

  pc_message(verbosity,2,"  Vocabulary file : %s\n",ng.vocab_filename);
  if (context_set) {
    pc_message(verbosity,2,"  Context cues file : %s\n",ng.context_cues_filename);
  }
  pc_message(verbosity,2,"  Cutoffs :\n     ");
  for (i=0;i<=ng.n-2;i++) {
    pc_message(verbosity,2,"%d-gram : %d     ",i+2,ng.cutoffs[i]);
  }
  pc_message(verbosity,2,"\n");

  switch (ng.vocab_type) {
  case CLOSED_VOCAB:
    pc_message(verbosity,2,"  Vocabulary type : Closed\n");
    break;
  case OPEN_VOCAB_1:
    pc_message(verbosity,2,"  Vocabulary type : Open - type 1\n");
    break;
  case OPEN_VOCAB_2:
    pc_message(verbosity,2,"  Vocabulary type : Open - type 2\n");
    pc_message(verbosity,2,"     OOV fraction = %g\n",ng.oov_fraction);
    break;
  }
  pc_message(verbosity,2,"  Minimum unigram count : %d\n",ng.min_unicount);
  pc_message(verbosity,2,"  Zeroton fraction : %g\n",ng.zeroton_fraction);
  if (ng.four_byte_counts) { 
    pc_message(verbosity,2,"  Counts will be stored in four bytes.\n");
  }
  else {
    pc_message(verbosity,2,"  Counts will be stored in two bytes.\n");
    pc_message(verbosity,2,"  Count table size : %d\n",ng.count_table_size);
  }
  pc_message(verbosity,2,"  Discounting method : ");
  switch (ng.discounting_method) {
  case GOOD_TURING:
    pc_message(verbosity,2,"Good-Turing\n");
    pc_message(verbosity,2,"     Discounting ranges :\n        ");
    for (i=0;i<=ng.n-1;i++) {
      pc_message(verbosity,2,"%d-gram : %d     ",i+1,ng.disc_range[i]);
    }
    pc_message(verbosity,2,"\n");
    break;
  case ABSOLUTE:
    pc_message(verbosity,2,"Absolute\n");
    break;
  case LINEAR:
    pc_message(verbosity,2,"Linear\n");
    break;
  case WITTEN_BELL:
    pc_message(verbosity,2,"Witten-Bell\n");
    break;
  }
  pc_message(verbosity,2,"  Memory allocation for tree structure : \n");
  switch(mem_alloc_method) {
  case TWO_PASSES:
    pc_message(verbosity,2,"     Perform a preliminary pass over the id n-gram file to determine \n     the amount of memory to allocate\n");
    break;
  case BUFFER:
    pc_message(verbosity,2,"     Allocate %d MB of memory, shared equally between all n-gram tables.\n",buffer_size);
    break;
  case SPECIFIED:
    pc_message(verbosity,2,"     Memory requirement specified.\n          ");
    for (i=0;i<=ng.n-2;i++) {
      pc_message(verbosity,2,"%d-gram : %d     ",i+2,ng.table_sizes[i+1]);
    }
    pc_message(verbosity,2,"\n");
    break;
  }
  pc_message(verbosity,2,"  Back-off weight storage : \n");
  if (ng.four_byte_alphas) {
    pc_message(verbosity,2,"     Back-off weights will be stored in four bytes.\n");
  }
  else {
    pc_message(verbosity,2,"     Back-off weights will be stored in two bytes.\n");
    pc_message(verbosity,2,"        Minimum back-off weight : %g\n",ng.min_alpha);
    pc_message(verbosity,2,"        Maximum back-off weight : %g\n",ng.max_alpha);
    pc_message(verbosity,2,"        Maximum number of out of range back-off weights : %d\n",ng.out_of_range_alphas);
  }

  pc_report_unk_args(&argc,argv,verbosity);


  /* Attempt to open all the files that we will need for input and
     output. It is better to do it here than to spend a few hours of
     CPU processing id-gram counts, only to find that the output path
     is invalid. */

  ng.id_gram_fp = rr_iopen(ng.id_gram_filename);

  /* Vocab is read by Roni's function which does the file opening for
     us, so no need to do it here. Don't need to worry about time
     being lost if file doesn't exist, since vocab is first thing to
     be read anyway. */

  if (context_set) {
    ng.context_cues_fp = rr_iopen(ng.context_cues_filename);
  }

  if (ng.write_arpa) {
    ng.arpa_fp = rr_oopen(ng.arpa_filename);
  }

  if (ng.write_bin) {
    ng.bin_fp = rr_oopen(ng.bin_filename);
  }


  /* --------------- Read in the vocabulary -------------- */

  pc_message(verbosity,2,"Reading vocabulary.\n");

  ng.vocab_ht = sih_create(1000,0.5,2.0,1);

  read_voc(ng.vocab_filename,verbosity,ng.vocab_ht,&ng.vocab,&(ng.vocab_size));
  
  /* Determine which of the vocabulary words are context cues */

  ng.no_of_ccs = 0;
  ng.context_cue = (flag *) rr_calloc(ng.vocab_size+1,sizeof(flag));

  if (context_set) {

    while (fgets (wlist_entry, sizeof (wlist_entry),ng.context_cues_fp)) {
      if (strncmp(wlist_entry,"##",2)==0) continue;
      sscanf (wlist_entry, "%s ",current_cc);
      if (strncmp(wlist_entry,"#",1)==0) {
	fprintf(stderr,"\n\n===========================================================\n");
        fprintf(stderr,":\nWARNING: line assumed NOT a comment:\n");
	fprintf(stderr,     ">>> %s <<<\n",wlist_entry);
        fprintf(stderr,     "         '%s' will be included in the context cues list\n",current_cc);
        fprintf(stderr,     "         (comments must start with '##')\n");
	fprintf(stderr,"===========================================================\n\n");
      }
      

      if (sih_lookup(ng.vocab_ht,current_cc,&current_cc_id) == 0) {
	pc_message(verbosity,1,"Warning : %s in the context cues file does not appear in the vocabulary.\n",current_cc);
      }
      else {
	ng.context_cue[(unsigned short) current_cc_id] = 1;
	pc_message(verbosity,2,"Context cue word : %s id = %d\n",current_cc,current_cc_id);
	ng.no_of_ccs++;
      }
    }
    rr_iclose(ng.context_cues_fp);
  }

  if ((sih_lookup(ng.vocab_ht,"<s>",&test_cc_id) != 0)) {
    if (ng.context_cue[(unsigned short) test_cc_id] == 0) {
      fprintf(stderr,"WARNING: <s> appears as a vocabulary item, but is not labelled as a\ncontext cue.\n");
    }
  }

  if ((sih_lookup(ng.vocab_ht,"<p>",&test_cc_id) != 0)) {
    if (ng.context_cue[(unsigned short) test_cc_id] == 0) {
      fprintf(stderr,"WARNING: <p> appears as a vocabulary item, but is not labelled as a\ncontext cue.\n");
    }
  }

  if ((sih_lookup(ng.vocab_ht,"<art>",&test_cc_id) != 0)) {
    if (ng.context_cue[(unsigned short) test_cc_id] == 0) {
      fprintf(stderr,"WARNING: <art> appears as a vocabulary item, but is not labelled as a\ncontext cue.\n");
    }
  }
       
		     
  /* Allocate space for the table_size array */

  if (ng.n>1) {

    switch(mem_alloc_method) {

    case TWO_PASSES: 
      ng.table_sizes = (table_size_t *) rr_calloc(ng.n,sizeof(table_size_t));
      pc_message(verbosity,2,"Calculating memory requirement.\n");
      calc_mem_req(&ng,is_ascii);
      break;
    case BUFFER:
      ng.table_sizes = (table_size_t *) rr_malloc(ng.n*sizeof(table_size_t));
      middle_size = sizeof(count_ind_t) + sizeof(bo_weight_t) + 
	sizeof(index__t) + sizeof(id__t);
      end_size = sizeof(count_ind_t) + sizeof(id__t);
      if (ng.four_byte_alphas) {
	middle_size += 2;
      }
      if (ng.four_byte_counts) {
	middle_size += 2;
	end_size += 2;
      }
   

      guess_mem(buffer_size,
		middle_size,
		end_size,
		ng.n,
		ng.table_sizes,
		verbosity);
      break;
    case SPECIFIED:
      break;

    }
  
  }
  else {

    ng.table_sizes = (table_size_t *) rr_calloc(1,sizeof(table_size_t));

  }



  ng.table_sizes[0] = ng.vocab_size+1;

  /* ----------- Allocate memory for tree structure -------------- */

  ng.count = (count_ind_t **) rr_malloc(sizeof(count_ind_t *)*ng.n);
  ng.count4 = (int **) rr_malloc(sizeof(int *)*ng.n);


  ng.count_table = (count_t **) rr_malloc(sizeof(count_t *)*ng.n);

  if (!ng.four_byte_counts) {
    for (i=0;i<=ng.n-1;i++) {
      ng.count_table[i] = (count_t *) rr_calloc(ng.count_table_size,
						sizeof(count_t));
    }
    ng.marg_counts = (count_ind_t *) rr_malloc(sizeof(count_ind_t)*
					       ng.table_sizes[0]);
  }
  else {
    ng.marg_counts4 = (int *) rr_malloc(sizeof(int)*ng.table_sizes[0]);
  }

  ng.word_id = (id__t **) rr_malloc(sizeof(id__t *)*ng.n);
  if (ng.four_byte_alphas) {
    ng.bo_weight4 = (four_byte_t **) rr_malloc(sizeof(four_byte_t *)*ng.n);
  }
  else {
    ng.bo_weight = (bo_weight_t **) rr_malloc(sizeof(bo_weight_t *)*ng.n);
  }

  ng.ind = (index__t **)  rr_malloc(sizeof(index__t *)*ng.n);

  /* First table */

  if (ng.four_byte_counts) {
    ng.count4[0] = (int *) rr_calloc(ng.table_sizes[0],sizeof(int));
  }
  else {
    ng.count[0] = (count_ind_t *) rr_calloc(ng.table_sizes[0],
					    sizeof(count_ind_t));
  }
  ng.uni_probs = (uni_probs_t *) rr_malloc(sizeof(uni_probs_t)*
					   ng.table_sizes[0]);
  ng.uni_log_probs = (uni_probs_t *) rr_malloc(sizeof(uni_probs_t)*
					       ng.table_sizes[0]);
  if (ng.four_byte_alphas) {
    ng.bo_weight4[0] = (four_byte_t *) rr_malloc(sizeof(four_byte_t)*
						ng.table_sizes[0]);
  }
  else {
    ng.bo_weight[0] = (bo_weight_t *) rr_malloc(sizeof(bo_weight_t)*
						ng.table_sizes[0]);
  }

  if (ng.n >=2 ) {
    ng.ind[0] = (index__t *) rr_calloc(ng.table_sizes[0],sizeof(index__t));
  }

  for (i=1;i<=ng.n-2;i++) {
    
    ng.word_id[i] = (id__t *) rr_malloc(sizeof(id__t)*ng.table_sizes[i]);
    if (ng.four_byte_counts) {
      ng.count4[i] = (int *) rr_malloc(sizeof(int)*ng.table_sizes[i]);
    }
    else {
      ng.count[i] = (count_ind_t *) 
	rr_malloc(sizeof(count_ind_t)*ng.table_sizes[i]);
    }
    if (ng.four_byte_alphas) {
      ng.bo_weight4[i] = (four_byte_t *) 
	rr_malloc(sizeof(four_byte_t)*ng.table_sizes[i]);
    }
    else {
      ng.bo_weight[i] = (bo_weight_t *) 
	rr_malloc(sizeof(bo_weight_t)*ng.table_sizes[i]);
    }
    
    ng.ind[i] = (index__t *) rr_malloc(sizeof(index__t)*ng.table_sizes[i]);

    mem_alloced = sizeof(count_ind_t) + sizeof(bo_weight_t) + 
		sizeof(index__t) + sizeof(id__t);
    
    if (ng.four_byte_alphas) {
      mem_alloced += 2;
    }

    if (ng.four_byte_counts) {
      mem_alloced += 2;
    }
   
    mem_alloced *= ng.table_sizes[i];
    
    pc_message(verbosity,2,"Allocated %d bytes to table for %d-grams.\n",
	       mem_alloced,i+1);
    
  }

  ng.word_id[ng.n-1] = (id__t *) 
    rr_malloc(sizeof(id__t)*ng.table_sizes[ng.n-1]);
  if (ng.four_byte_counts) {
    ng.count4[ng.n-1] = (int *) rr_malloc(sizeof(int)*ng.table_sizes[ng.n-1]);
    pc_message(verbosity,2,"Allocated %d bytes to table for %d-grams.\n",
	       (sizeof(int) + 
		sizeof(id__t))*ng.table_sizes[ng.n-1],ng.n);
    
  }
  else {
    ng.count[ng.n-1] = (count_ind_t *) 
      rr_malloc(sizeof(count_ind_t)*ng.table_sizes[ng.n-1]);
    pc_message(verbosity,2,"Allocated %d bytes to table for %d-grams.\n",
	       (sizeof(count_ind_t) + 
		sizeof(id__t))*ng.table_sizes[ng.n-1],ng.n);

  }



  
  /* Allocate memory for table for first-byte of indices */

  ng.ptr_table = (int **) rr_malloc(sizeof(int *)*ng.n);
  ng.ptr_table_size = (unsigned short *) 
    rr_calloc(ng.n,sizeof(unsigned short));
  for (i=0;i<=ng.n-1;i++) {
    ng.ptr_table[i] = (int *) rr_calloc(65535,sizeof(int));
  }

  /* Allocate memory for alpha array */

  ng.alpha_array = (double *) rr_malloc(sizeof(double)*ng.out_of_range_alphas);
  ng.size_of_alpha_array = 0;

  /* Allocate memory for frequency of frequency information */

  ng.freq_of_freq = (int **) rr_malloc(sizeof(int *)*ng.n);

  switch(ng.discounting_method) {
  case LINEAR:
    for (i=0;i<=ng.n-1;i++) {
      ng.freq_of_freq[i] = (int *) rr_calloc(2,sizeof(int));
    }
    break;
  case GOOD_TURING:
    for (i=0;i<=ng.n-1;i++) {
      ng.freq_of_freq[i] = (int *) rr_calloc(ng.fof_size[i]+1,sizeof(int));
    }
    break;
  case ABSOLUTE:
    for (i=0;i<=ng.n-1;i++) {
      ng.freq_of_freq[i] = (int *) rr_calloc(3,sizeof(int));
    }
    ng.abs_disc_const = (double *) rr_malloc(sizeof(double)*ng.n);

    break;
  case WITTEN_BELL:
    ng.freq_of_freq[0] = (int *) rr_calloc(1,sizeof(int));
    break;

  }


  /* Read n-grams into the tree */


  pc_message(verbosity,2,"Processing id n-gram file.\n");
  pc_message(verbosity,2,"20,000 n-grams processed for each \".\", 1,000,000 for each line.\n");

  /* Allocate space for ngrams id arrays */

  current_ngram.id_array = (id__t *) rr_calloc(ng.n,sizeof(id__t));
  previous_ngram.id_array = (id__t *) rr_calloc(ng.n,sizeof(id__t));
  current_ngram.n = ng.n;
  previous_ngram.n = ng.n;
  
  ng.num_kgrams = (int *) rr_calloc(ng.n,sizeof(int));
  ng_count = (count_t *) rr_calloc(ng.n,sizeof(count_t));
  nlines = 1;
  ng.n_unigrams = 0;
  
  /* Process first n-gram */
  
  get_ngram(ng.id_gram_fp,&current_ngram,is_ascii);

  contains_unks = 0;
  for (i=0;i<=ng.n-1;i++) {
    if (current_ngram.id_array[i] == 0) {
      contains_unks = 1;
    }
  }

  while (ng.vocab_type == CLOSED_VOCAB && contains_unks){

    get_ngram(ng.id_gram_fp,&current_ngram,is_ascii);
    contains_unks = 0;
    for (i=0;i<=ng.n-1;i++) {
      if (current_ngram.id_array[i] == 0) {
	contains_unks = 1;
      }
    }
  }

  for (i=0;i<=ng.n-2;i++) {
    ng.ind[i][0] = new_index(0,ng.ptr_table[i],&(ng.ptr_table_size[i]),0);
    ng.word_id[i+1][0] = current_ngram.id_array[i+1];
    ng.num_kgrams[i+1]++;
    ng_count[i] = current_ngram.count;
  }
  ng_count[0] = current_ngram.count;
  
  if (ng.discounting_method == GOOD_TURING && 
      current_ngram.count <= ng.fof_size[ng.n-1]) {
    
    if (current_ngram.count <= 0) {
      quit(-1,"Error in idngram stream. This is most likely to be caused by trying to read\na gzipped file as if it were uncompressed. Ensure that all gzipped files have\na .gz extension. Other causes might be confusion over whether the file is in\nascii or binary format.\n");
    }

    ng.freq_of_freq[ng.n-1][current_ngram.count]++;
  }
  
  if (ng.discounting_method == LINEAR && current_ngram.count == 1) {
    ng.freq_of_freq[ng.n-1][1]++;
  }
	  
  if (ng.discounting_method == ABSOLUTE && current_ngram.count <= 2) {

    if (current_ngram.count <= 0) {
      quit(-1,"Error in idngram stream. This is most likely to be caused by trying to read\na gzipped file as if it were uncompressed. Ensure that all gzipped files have\na .gz extension. Other causes might be confusion over whether the file is in\nascii or binary format.\n");
    }

    ng.freq_of_freq[ng.n-1][current_ngram.count]++;
  }

    store_count(ng.four_byte_counts,
		ng.count_table[ng.n-1],
		ng.count_table_size,
		ng.count[ng.n-1],
		ng.count4[ng.n-1],
		0,
		current_ngram.count); 
  
  if (current_ngram.count <= ng.cutoffs[ng.n-2]) {
    ng.num_kgrams[ng.n-1]--;
  }
  prev_id1 = current_ngram.id_array[0];
    
  displayed_oov_warning = 0;

  for (i=0;i<=ng.n-1;i++) {
    previous_ngram.id_array[i] = current_ngram.id_array[i];
  }
  previous_ngram.count = current_ngram.count;


  while (!rr_feof(ng.id_gram_fp)) {


    if (get_ngram(ng.id_gram_fp,&current_ngram,is_ascii)) {
    
      if (ng.vocab_type == CLOSED_VOCAB) {
	contains_unks = 0;
	for (i=0;i<=ng.n-1;i++) {
	  if (current_ngram.id_array[i] == 0) {
	    contains_unks = 1;
	  }
	}
      }
    
      if (!contains_unks || ng.vocab_type != CLOSED_VOCAB) {


	

  
	/* Test for where this ngram differs from last - do we have an
	   out-of-order ngram? */
      
	pos_of_novelty = ng.n;

	for (i=0;i<=ng.n-1;i++) {
	  if (current_ngram.id_array[i] > previous_ngram.id_array[i]) {
	    pos_of_novelty = i;
	    i=ng.n;
	  }
	  else {
	    if (current_ngram.id_array[i] < previous_ngram.id_array[i]) {
	      if (nlines < 5) { /* Error ocurred early - file format? */
		quit(-1,"Error : n-gram ordering problem - could be due to using wrong file format.\nCheck whether id n-gram file is in ascii or binary format.\n");
	      }
	      else {
		quit(-1,"Error : n-grams are not correctly ordered. Error occurred at ngram %d.\n",nlines);
	      }
	    }
	  }
	}

	if (pos_of_novelty == ng.n) {
	  if (nlines > 3) {
	    quit(-1,"Error - same n-gram appears twice in idngram stream.\n");
	  }
	  else {
	    quit(-1,"Error in the idngram stream. It appears that the same n-gram occurs twice\n in the stream. Check that text2idngram exited successfully, and the \nformat (binary/ascii) of the idngram file.\n");
	  }
	}
    
	nlines++;
    
	if (nlines % 20000 == 0) {
	  if (nlines % 1000000 == 0) {
	    pc_message(verbosity,2,".\n");
	  }
	  else {
	    pc_message(verbosity,2,".");
	  }
	}
    
	/* Add new n-gram as soon as it is encountered */
    
	/* If all of the positions 2,3,...,n of the n-gram are context
	   cues then ignore the n-gram. */
    
	if (ng.n > 1) {
      
	  store_count(ng.four_byte_counts,
		      ng.count_table[ng.n-1],
		      ng.count_table_size,
		      ng.count[ng.n-1],
		      ng.count4[ng.n-1],
		      ng.num_kgrams[ng.n-1],
		      current_ngram.count);

	  
	  if (ng.discounting_method == GOOD_TURING && 
	      current_ngram.count <= ng.fof_size[ng.n-1]) {

	    if (current_ngram.count <= 0) {
	      quit(-1,"Error in idngram stream. This is most likely to be caused by trying to read\na gzipped file as if it were uncompressed. Ensure that all gzipped files have\na .gz extension. Other causes might be confusion over whether the file is in\nascii or binary format.\n");
	    }
	    
	    ng.freq_of_freq[ng.n-1][current_ngram.count]++;
	  }
	  
	  if (ng.discounting_method == LINEAR && current_ngram.count == 1) {
	    ng.freq_of_freq[ng.n-1][1]++;
	  }
	  
	  if (ng.discounting_method == ABSOLUTE && current_ngram.count <= 2) {

	    if (current_ngram.count <= 0) {
	      quit(-1,"Error in idngram stream. This is most likely to be caused by trying to read\na gzipped file as if it were uncompressed. Ensure that all gzipped files have\na .gz extension. Other causes might be confusion over whether the file is in\nascii or binary format.\n");
	    }
	    
	    ng.freq_of_freq[ng.n-1][current_ngram.count]++;
	  }
	  
	  ng.word_id[ng.n-1][ng.num_kgrams[ng.n-1]] = 
	    current_ngram.id_array[ng.n-1];
	  
	  ng.num_kgrams[ng.n-1]++;
	  
	  
	  if (ng.num_kgrams[ng.n-1] >= ng.table_sizes[ng.n-1]) {
	    quit(-1,"\nMore than %d %d-grams needed to be stored. Rerun with a higher table size.\n",ng.table_sizes[ng.n-1],ng.n);
	  }

	}
	/* Deal with new 2,3,...,(n-1)-grams */
      
	for (i=ng.n-2;i>=MAX(1,pos_of_novelty);i--) {

	  if (ng.discounting_method == GOOD_TURING && 
	      ng_count[i] <= ng.fof_size[i]) {
	    ng.freq_of_freq[i][ng_count[i]]++;
	  }

	  if (ng.discounting_method == LINEAR && ng_count[i] == 1) {
	    ng.freq_of_freq[i][1]++;
	  }

	  if (ng.discounting_method == ABSOLUTE && ng_count[i] <= 2) {
	    ng.freq_of_freq[i][ng_count[i]]++;
	  }
	  
	  if (ng_count[i] <= ng.cutoffs[i-1]) {
	    ng.num_kgrams[i]--;
	  }
	  else {
	    store_count(ng.four_byte_counts,
			ng.count_table[i],
			ng.count_table_size,
			ng.count[i],
			ng.count4[i],
			ng.num_kgrams[i]-1,
			ng_count[i]);
	  }
	  ng_count[i] = current_ngram.count;
	  ng.word_id[i][ng.num_kgrams[i]] = current_ngram.id_array[i];
	  ng.ind[i][ng.num_kgrams[i]] = new_index(ng.num_kgrams[i+1]-1,
						  ng.ptr_table[i],
						  &(ng.ptr_table_size[i]),
						  ng.num_kgrams[i]);

	  ng.num_kgrams[i]++;
	
	  if (ng.num_kgrams[i] >= ng.table_sizes[i]) {
	    quit(-1,"More than %d %d-grams needed to be stored. Rerun with a higher table size.\n",ng.table_sizes[i],i+1);
	  }
  
	}

	/* this was original place - messes up for bigram models */

	/*	if (current_ngram.count <= ng.cutoffs[ng.n-2]) {
		ng.num_kgrams[ng.n-1]--;
		} */
      
	for (i=0;i<=pos_of_novelty-1;i++) {
	  ng_count[i] += current_ngram.count;
	}
      
	/* Deal with new 1-grams */
      
	if (pos_of_novelty == 0) {
	  if (ng.n>1) {

	    for (i = prev_id1 + 1; i <= current_ngram.id_array[0]; i++) {
	      ng.ind[0][i] = new_index(ng.num_kgrams[1]-1,
				       ng.ptr_table[0],
				       &(ng.ptr_table_size[0]),
				       i);
	    }
	    prev_id1 = current_ngram.id_array[0];

	  }

	  if (ng.discounting_method == GOOD_TURING && 
	      ng_count[0] <= ng.fof_size[0]) {
	    ng.freq_of_freq[0][ng_count[0]]++;
	  }

	  if (ng.discounting_method == LINEAR && ng_count[0] == 1) {
	    ng.freq_of_freq[0][1]++;
	  }

	  if (ng.discounting_method == ABSOLUTE && ng_count[0] <= 2) {
	    ng.freq_of_freq[0][ng_count[0]]++;
	  }

	  if (!ng.context_cue[previous_ngram.id_array[0]]) {
	    ng.n_unigrams += ng_count[0];

	    store_count(ng.four_byte_counts,
			ng.count_table[0],
			ng.count_table_size,
			ng.count[0],
			ng.count4[0],
			previous_ngram.id_array[0],
			ng_count[0]); 

	  }

	  store_count(ng.four_byte_counts,
		      ng.count_table[0],
		      ng.count_table_size,
		      ng.marg_counts,
		      ng.marg_counts4,
		      previous_ngram.id_array[0],
		      ng_count[0]);
		      
	  ng_count[0] = current_ngram.count;
	}

	if (current_ngram.count <= ng.cutoffs[ng.n-2]) {
	  ng.num_kgrams[ng.n-1]--;
	}

	for (i=0;i<=ng.n-1;i++) {
	  previous_ngram.id_array[i] = current_ngram.id_array[i];
	}
	previous_ngram.count = current_ngram.count;
	

      }
      else {
	if (!displayed_oov_warning){
	  pc_message(verbosity,2,"Warning : id n-gram stream contains OOV's (n-grams will be ignored).\n");
	  displayed_oov_warning = 1;
	}
      }
    }
  }

  rr_iclose(ng.id_gram_fp);

  for (i=ng.n-2;i>=1;i--) {
    if (ng.discounting_method == GOOD_TURING && 
	ng_count[i] <= ng.fof_size[i]) {
      ng.freq_of_freq[i][ng_count[i]]++;
    }
    
    if (ng.discounting_method == LINEAR && ng_count[i] == 1) {
      ng.freq_of_freq[i][1]++;
    }

    if (ng.discounting_method == ABSOLUTE && ng_count[i] <= 2) {
      ng.freq_of_freq[i][ng_count[i]]++;
    }

    if (ng_count[i] <= ng.cutoffs[i-1]) {
      ng.num_kgrams[i]--;
    }
    else {

      store_count(ng.four_byte_counts,
		  ng.count_table[i],
		  ng.count_table_size,
		  ng.count[i],
		  ng.count4[i],
		  ng.num_kgrams[i]-1,
		  ng_count[i]);

    }
  }
  
  if (ng.discounting_method == GOOD_TURING && ng_count[0] <= ng.fof_size[0]) {
    ng.freq_of_freq[0][ng_count[0]]++;
  }


  if (ng.discounting_method == LINEAR && ng_count[0] == 1) {
    ng.freq_of_freq[0][1]++;
  }

  if (ng.discounting_method == ABSOLUTE && ng_count[0] <= 2) {
    ng.freq_of_freq[0][ng_count[0]]++;
  }

  if (!ng.context_cue[current_ngram.id_array[0]]) {
    ng.n_unigrams += ng_count[0];

    store_count(ng.four_byte_counts,
		ng.count_table[0],
		ng.count_table_size,
		ng.count[0],
		ng.count4[0],
		current_ngram.id_array[0],
		ng_count[0]);
    
  }

  store_count(ng.four_byte_counts,
	      ng.count_table[0],
	      ng.count_table_size,
	      ng.marg_counts,
	      ng.marg_counts4,
	      current_ngram.id_array[0],
	      ng_count[0]);

  if (ng.n>1) {

    for (i=current_ngram.id_array[0]+1;i<=ng.vocab_size;i++) {
      ng.ind[0][i] = new_index(ng.num_kgrams[1],
			       ng.ptr_table[0],
			       &(ng.ptr_table_size[0]),
			       current_ngram.id_array[0]);
    }
  }

  pc_message(verbosity,2,"\n");

  /* Impose a minimum unigram count, if required */

  if (ng.min_unicount > 0) {

    int nchanged;
    
    nchanged = 0;

    for (i=ng.first_id;i<=ng.vocab_size;i++) {
      if ((return_count(ng.four_byte_counts,
			ng.count_table[0],
			ng.count[0],
			ng.count4[0],
			i) < ng.min_unicount) && !ng.context_cue[i]) {

	switch(ng.discounting_method) {
	case LINEAR:
	  if (ng.count[0][i] <= 1) {
	    ng.freq_of_freq[0][ng.count[0][i]]--;
	  }
	  break;
	case ABSOLUTE:
	  if (ng.count[0][i] <= 2) {
	    ng.freq_of_freq[0][ng.count[0][i]]--;
	  }
	case GOOD_TURING:
	  if (ng.count[0][i] <= ng.fof_size[0]) {
	    ng.freq_of_freq[0][ng.count[0][i]]--;
	  }
	  break;
	case WITTEN_BELL:
	  if (ng.count[0][i] == 0) {
	    ng.freq_of_freq[0][ng.count[0][i]]--;
	  }
	  break;
	}
	ng.n_unigrams += (ng.min_unicount - ng.count[0][i]);

	store_count(ng.four_byte_counts,
		    ng.count_table[0],
		    ng.count_table_size,
		    ng.count[0],
		    ng.count4[0],
		    i,
		    ng.min_unicount);

	nchanged++;
      }
    }

    if (nchanged > 0) {
      pc_message(verbosity,2,
		 "Unigram counts of %d words were bumped up to %d.\n",
		 nchanged,ng.min_unicount);
    }

  }

  /* Count zeroton information for unigrams */

  ng.freq_of_freq[0][0] = 0;
  
  for (i=ng.first_id;i<=ng.vocab_size;i++) {
    if (return_count(ng.four_byte_counts,
		     ng.count_table[0],
		     ng.count[0],
		     ng.count4[0],
		     i) == 0) {
      ng.freq_of_freq[0][0]++;
    }
  }
  

  if (ng.discounting_method == GOOD_TURING) {
    for (i=0;i<=ng.n-1;i++) {
      for (j=1;j<=ng.fof_size[i];j++) {
	pc_message(verbosity,3,"fof[%d][%d] = %d\n",i,j,ng.freq_of_freq[i][j]);
      }
    }
  }

  


  /* Calculate discounted counts */

  pc_message(verbosity,2,"Calculating discounted counts.\n");

  switch(ng.discounting_method) {
  case GOOD_TURING:

    ng.gt_disc_ratio = (disc_val_t **) rr_malloc(sizeof(disc_val_t *)*ng.n);
    
    for (i=0;i<=ng.n-1;i++) {
      ng.gt_disc_ratio[i] = (disc_val_t *) 
	rr_malloc(sizeof(disc_val_t)*ng.fof_size[i]);
    }
    
    for (i=0;i<=ng.n-1;i++) {
      if (i==0) {
	compute_gt_discount(i+1,
			    ng.freq_of_freq[0],
			    ng.fof_size[0],
			    &ng.disc_range[0],
			    0,
			    verbosity,
			    &ng.gt_disc_ratio[0]);
      }
      else {
	compute_gt_discount(i+1,
			    ng.freq_of_freq[i],
			    ng.fof_size[i],
			    &ng.disc_range[i],
			    ng.cutoffs[i-1],
			    verbosity,
			    &ng.gt_disc_ratio[i]);
      }
    }
    break;
  case WITTEN_BELL:
    break;
  case LINEAR:
    ng.lin_disc_ratio = (disc_val_t *) rr_malloc(sizeof(disc_val_t)*ng.n);
    pc_message(verbosity,1,"Linear discounting ratios :\n");
    for (i=0;i<=ng.n-1;i++) {
      ng.lin_disc_ratio[i] = 1 - ( (float) ng.freq_of_freq[i][1]/
				   (float) ng.n_unigrams);
      pc_message(verbosity,1,"%d-gram : %g\n",i+1,ng.lin_disc_ratio[i]);
    }

    break;
  case ABSOLUTE:
    pc_message(verbosity,1,"Absolute discounting ratios :\n");
    for (i=0;i<=ng.n-1;i++) {
      ng.abs_disc_const[i] = ((float) ng.freq_of_freq[i][1] ) /
	((float) ng.freq_of_freq[i][1] + (2*ng.freq_of_freq[i][2]) );
      pc_message(verbosity,1,"%d-gram : ",i+1);
      for (j=1;j<=5;j++) {
	pc_message(verbosity,1,"%g ",(j-ng.abs_disc_const[i])/j);
      }
      pc_message(verbosity,1," ... \n");
    }
    break;
  }
     
     
  /* Smooth unigram distribution, to give some mass to zerotons */
     
  compute_unigram(&ng,verbosity);

  /* Increment Contexts if using Good-Turing discounting. No need otherwise,
     since all values are discounted anyway. */

  if (ng.discounting_method == GOOD_TURING) {
    pc_message(verbosity,2,"Incrementing contexts...\n");  

    for (i=ng.n-1;i>=1;i--) {
      
      increment_context(&ng,i,verbosity);
      
    }
  }


  /* Calculate back-off weights */

  pc_message(verbosity,2,"Calculating back-off weights...\n");

  for (i=1;i<=ng.n-1;i++) {
    compute_back_off(&ng,i,verbosity);
  }

  if (!ng.four_byte_alphas) {
    pc_message(verbosity,3,"Number of out of range alphas = %d\n",
	       ng.size_of_alpha_array);
  }

  /* Write out LM */

  pc_message(verbosity,2,"Writing out language model...\n");

  if (ng.write_arpa) {

    write_arpa_lm(&ng,verbosity);

  }

  if (ng.write_bin) {
    
    write_bin_lm(&ng,verbosity);

  }

  pc_message(verbosity,0,"idngram2lm : Done.\n");

  exit(0);
    
}

	      
