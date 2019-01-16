
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

#define MAX_ARGS 200

#include "evallm.h"
#include "pc_libs/pc_general.h"
#include "rr_libs/general.h"
#include <stdio.h>
#include <string.h>

void main (int argc, char **argv) {

  ng_t ng;
  arpa_lm_t arpa_ng;
  char input_string[500];
  int num_of_args;
  char *args[MAX_ARGS];
  char *lm_filename_arpa;
  char *lm_filename_binary;
  flag told_to_quit;
  flag inconsistant_parameters;
  flag backoff_from_unk_inc;
  flag backoff_from_unk_exc;
  flag backoff_from_ccs_inc;
  flag backoff_from_ccs_exc;
  flag arpa_lm;
  flag binary_lm;
  flag include_unks;
  char *fb_list_filename;
  char *probs_stream_filename;
  char *annotation_filename;
  char *text_stream_filename;
  char *oov_filename;
  char *ccs_filename;
  double log_base;
  char wlist_entry[1024];
  char current_cc[200];
  int current_cc_id;
  FILE *context_cues_fp;
  int n;

  /* Process command line */

  report_version(&argc,argv);

  if (pc_flagarg(&argc, argv,"-help") || 
      argc == 1 || 
      (strcmp(argv[1],"-binary") && strcmp(argv[1],"-arpa"))) {
   fprintf(stderr,"evallm : Evaluate a language model.\n");
   fprintf(stderr,"Usage : evallm [ -binary .binlm | \n");
   fprintf(stderr,"                 -arpa .arpa [ -context .ccs ] ]\n");
   exit(1);
  }
  

  lm_filename_arpa = salloc(pc_stringarg(&argc, argv,"-arpa",""));

  if (strcmp(lm_filename_arpa,"")) {
    arpa_lm = 1;
  }
  else {
    arpa_lm = 0;
  }

  lm_filename_binary = salloc(pc_stringarg(&argc, argv,"-binary",""));

  if (strcmp(lm_filename_binary,"")) {
    binary_lm = 1;
  }
  else {
    binary_lm = 0;
  }

  if (arpa_lm && binary_lm) {
    quit(-1,"Error : Can't use both -arpa and -binary flags.\n");
  }
  
  if (!arpa_lm && !binary_lm) {
    quit(-1,"Error : Must specify either a binary or an arpa format language model.\n");
  }

  ccs_filename = salloc(pc_stringarg(&argc, argv,"-context",""));

  if (binary_lm && strcmp(ccs_filename,"")) {
    fprintf(stderr,"Warning - context cues file not needed with binary language model file.\nWill ignore it.\n");
  }

  pc_report_unk_args(&argc,argv,2);
 
  /* Load language model */

  if (arpa_lm) {
    fprintf(stderr,"Reading in language model from file %s\n",
	    lm_filename_arpa);
    load_arpa_lm(&arpa_ng,lm_filename_arpa);
  }
  else {
    fprintf(stderr,"Reading in language model from file %s\n",
	    lm_filename_binary);
    load_lm(&ng,lm_filename_binary); 
  }

  fprintf(stderr,"\nDone.\n");

  if (!arpa_lm) {
    n=ng.n;
  }
  else {
    n=arpa_ng.n;
  }

  if (arpa_lm) {
    arpa_ng.context_cue = 
      (flag *) rr_calloc(arpa_ng.table_sizes[0],sizeof(flag));    
    arpa_ng.no_of_ccs = 0;
    if (strcmp(ccs_filename,"")) {
      context_cues_fp = rr_iopen(ccs_filename);
      while (fgets (wlist_entry, sizeof (wlist_entry),context_cues_fp)) {
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
	
	
	if (sih_lookup(arpa_ng.vocab_ht,current_cc,&current_cc_id) == 0) {
	  quit(-1,"Error : %s in the context cues file does not appear in the vocabulary.\n",current_cc);
	}
	
	arpa_ng.context_cue[(unsigned short) current_cc_id] = 1;
	arpa_ng.no_of_ccs++;
	fprintf(stderr,"Context cue word : %s id = %d\n",current_cc,current_cc_id);
      }
      rr_iclose(context_cues_fp);
    }
  }

  /* Process commands */
  
  told_to_quit = 0;
  num_of_args = 0;

  while (!feof(stdin) && !told_to_quit) {
    printf("evallm : ");
    gets(input_string);

    if (!feof(stdin)) {
      parse_comline(input_string,&num_of_args,args);

      log_base = pc_doublearg(&num_of_args,args,"-log_base",10.0);

      backoff_from_unk_inc = pc_flagarg(&num_of_args,args,
					"-backoff_from_unk_inc");
      backoff_from_ccs_inc = pc_flagarg(&num_of_args,args,
					"-backoff_from_ccs_inc");
      backoff_from_unk_exc = pc_flagarg(&num_of_args,args,
					"-backoff_from_unk_exc");
      backoff_from_ccs_exc = pc_flagarg(&num_of_args,args,
					"-backoff_from_ccs_exc");
      include_unks = pc_flagarg(&num_of_args,args,"-include_unks");
      fb_list_filename = salloc(pc_stringarg(&num_of_args,args,
					     "-backoff_from_list",""));
    
      text_stream_filename = 
	salloc(pc_stringarg(&num_of_args,args,"-text",""));
      probs_stream_filename = 
	salloc(pc_stringarg(&num_of_args,args,"-probs",""));
      annotation_filename = 
	salloc(pc_stringarg(&num_of_args,args,"-annotate",""));
      oov_filename = salloc(pc_stringarg(&num_of_args,args,"-oovs",""));


      inconsistant_parameters = 0;
    
      if (backoff_from_unk_inc && backoff_from_unk_exc) {
	fprintf(stderr,"Error : Cannot specify both exclusive and inclusive forced backoff.\n");
	fprintf(stderr,"Use only one of -backoff_from_unk_exc and -backoff_from_unk_inc\n");
	inconsistant_parameters = 1;
      }

      if (backoff_from_ccs_inc && backoff_from_ccs_exc) {
	fprintf(stderr,"Error : Cannot specify both exclusive and inclusive forced backoff.\n");
	fprintf(stderr,"Use only one of -backoff_from_ccs_exc and -backoff_from_ccs_inc\n");
	inconsistant_parameters = 1;
      }

      if (num_of_args > 0) {
      
	if (!inconsistant_parameters) {
	  if (!strcmp(args[0],"perplexity")) {
	    compute_perplexity(&ng,
			       &arpa_ng,
			       text_stream_filename,
			       probs_stream_filename,
			       annotation_filename,
			       oov_filename,
			       fb_list_filename,
			       backoff_from_unk_inc,
			       backoff_from_unk_exc,
			       backoff_from_ccs_inc,
			       backoff_from_ccs_exc,
			       arpa_lm,
			       include_unks,
			       log_base);
	  }
	  else {
	    if (!strcmp(args[0],"validate")) {

	      if (num_of_args != n) {
		fprintf(stderr,"Error : must specify %d words of context.\n",
			n-1);
	      }
	      else {
	      
		/* Assume last n-1 parameters form context */
	      
		validate(&ng,
			 &arpa_ng,
			 &(args[num_of_args-n+1]),
			 backoff_from_unk_inc,
			 backoff_from_unk_exc,
			 backoff_from_ccs_inc,
			 backoff_from_ccs_exc,
			 arpa_lm,
			 fb_list_filename);
	      }
	    }
	    else {
	      if (!strcmp(args[0],"stats")) {
		if (arpa_lm) {
		  display_arpa_stats(&arpa_ng);
		}
		else {
		  display_stats(&ng);
		}
	      }
	      else {
		if (!strcmp(args[0],"quit")) {
		  told_to_quit=1;
		}
		else {
		  if (!strcmp(args[0],"help")) {

		    printf("The user may specify one of the following commands: \n");
		    printf("\n");
		    printf(" - perplexity\n");
		    printf("\n");
		    printf("Computes the perplexity of a given text. May optionally specify words\n");
		    printf("from which to force back-off.\n");
		    printf("\n");
		    printf("Syntax: \n");
		    printf("\n");
		    printf("perplexity -text .text\n");
		    printf("         [ -probs .fprobs ]\n");
		    printf("         [ -oovs .oov_file ]\n");
		    printf("         [ -annotate .annotation_file ]         \n");
		    printf("         [ -backoff_from_unk_inc | -backoff_from_unk_exc ]\n");
		    printf("         [ -backoff_from_ccs_inc | -backoff_from_ccs_exc ] \n");
		    printf("         [ -backoff_from_list .fblist ]\n");
		    printf("         [ -include_unks ]\n");
		    printf("\n");
		    printf(" - validate\n");
		    printf("       \n");
		    printf("Calculate the sum of the probabilities of all the words in the\n");
		    printf("vocabulary given the context specified by the user.\n");
		    printf("\n");
		    printf("Syntax: \n");
		    printf("\n");
		    printf("validate [ -backoff_from_unk -backoff_from_ccs |\n");
		    printf("           -backoff_from_list .fblist ]\n");
		    printf("         [ -forced_backoff_inc | -forced_back_off_exc ]      \n");
		    printf("           word1 word2 ... word_(n-1)\n");
		    printf("\n");
		    printf("Where n is the n in n-gram. \n");
		    printf("\n");
		    printf(" - help\n");
		    printf("\n");
		    printf("Displays this help message.\n");
		    printf("\n");
		    printf("Syntax: \n");
		    printf("\n");
		    printf("help\n");
		    printf("\n");
		    printf(" - quit\n");
		    printf("\n");
		    printf("Exits the program.\n");
		    printf("\n");
		    printf("Syntax: \n");
		    printf("\n");
		    printf("quit\n");	

		  } 
	      
		  else {
		    fprintf(stderr,"Unknown command : %s\nType \'help\'\n",
			    args[0]);
		  }
		}
	      }
	    }
	  }
	}
      }
    }    
  }

  fprintf(stderr,"evallm : Done.\n");

  exit(0);
  
}

    
    
    

