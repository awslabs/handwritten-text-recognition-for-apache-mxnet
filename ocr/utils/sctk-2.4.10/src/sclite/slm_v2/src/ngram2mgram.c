
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

/* 

   Converts a n-gram stream in to an m-gram stream, where n > m 

   Input (from the standard input) can either be :
   
   a binary id n-gram file
   an ascii id n-gram file
   a word n-gram file
   
   Output (at the standard output) will be the same format of the input.

   Note: Program is not intelligent enough to be able to tell value of n from 
   the input, so user must specify it on the command line.

*/

#define MAX_WORD_LENGTH 150

#define BINARY 1
#define ASCII 2
#define WORDS 3

#define NUMERIC 1
#define ALPHA 2

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toolkit.h"
#include "ngram.h"
#include "pc_libs/pc_general.h"
#include "rr_libs/general.h"

/***************************
      MAIN FUNCTION
 ***************************/

void main(int argc, char *argv[]) {

  int verbosity;
  int n;
  int m;
  int i;
  int input_type;
  int storage_type;
  unsigned short *current_ngram_int;
  unsigned short *previous_ngram_int;
  char **current_ngram_text;
  char **previous_ngram_text;
  int current_count;
  int running_total;
  flag same;
  flag first_one;
  flag got_to_eof;
   
  running_total = 0;

  report_version(&argc,argv);

  if (pc_flagarg( &argc, argv,"-help") || argc==1) {
    fprintf(stderr,"ngram2mgram - Convert an n-gram file to an m-gram file, where m<n\n");
    fprintf(stderr,"Usage : ngram2mgram   -n N -m M\n");
    fprintf(stderr,"                    [ -binary | -ascii | -words ]\n");
    fprintf(stderr,"                    < .ngram > .mgram\n");
    exit(1);
  }
 
  n = pc_intarg( &argc, argv,"-n",0);
  m = pc_intarg( &argc, argv,"-m",0);
  verbosity = pc_intarg(&argc,argv,"-verbosity",DEFAULT_VERBOSITY);
  

  input_type = 0;
  
  if (pc_flagarg( &argc, argv,"-binary")) {
    input_type = BINARY;
  }

  if (pc_flagarg( &argc, argv,"-ascii")) {
    if (input_type != 0) {
      quit(-1,"Error : more than one file format specified.\n");
    }
    input_type = ASCII;
  }

  if (pc_flagarg( &argc, argv,"-words")) {  
    if (input_type != 0) {
      quit(-1,"Error : more than one file format specified.\n");
    }
    input_type = WORDS;
  }    

  if (input_type == 0) {
    pc_message(verbosity,2,"Warning : no input type specified. Defaulting to binary.\n");
    input_type = BINARY;
  }

  if (n == 0) {
    quit(-1,"Must specify a value for n. Use the -n switch.\n");
  }

  if (m == 0) {
    quit(-1,"Must specify a value for m. Use the -m switch.\n");
  }
  
  if (n<=m) {
    quit(-1,"n must be greater than m.\n");
  }

  pc_report_unk_args(&argc,argv,verbosity);

  if (input_type == BINARY || input_type == ASCII) {
    storage_type = NUMERIC;
  }
  else {
    storage_type = ALPHA;
  }

  if (storage_type == NUMERIC) {
    current_ngram_int = (unsigned short *) 
      rr_malloc(n*sizeof(unsigned short));
    previous_ngram_int = (unsigned short *) 
      rr_malloc(n*sizeof(unsigned short));

    /* And to prevent compiler warnings ... */

    current_ngram_text = NULL;
    previous_ngram_text = NULL;
  }
  else {
    current_ngram_text = (char **) rr_malloc(n*sizeof(char *));
    previous_ngram_text = (char **) rr_malloc(n*sizeof(char *));
    for (i=0;i<=n-1;i++) {
      current_ngram_text[i] = (char *) rr_malloc(MAX_WORD_LENGTH*sizeof(char));
      previous_ngram_text[i] = (char *) rr_malloc(MAX_WORD_LENGTH*sizeof(char));
    }

    /* And to prevent compiler warnings ... */

    current_ngram_int = NULL;
    previous_ngram_int = NULL;

  }

  got_to_eof = 0;
  first_one = 1;

  while (!rr_feof(stdin)) {

    /* Store previous n-gram */

    if (!first_one) {

      if (storage_type == NUMERIC) {
	for (i=0;i<=n-1;i++) {
	  previous_ngram_int[i] = current_ngram_int[i];
	}
      }
      else {
	for (i=0;i<=n-1;i++) {
	  strcpy(previous_ngram_text[i],current_ngram_text[i]);
	}
      }

    }

    /* Read new n-gram */

    switch(input_type) {
    case BINARY:
      for (i=0;i<=n-1;i++) {
	rr_fread(&current_ngram_int[i],sizeof(id__t),1,stdin,
		 "from id_ngrams at stdin",0);
      }
      rr_fread(&current_count,sizeof(count_t),1,stdin,
	       "from id_ngrams file at stdin",0);
      break;
    case ASCII:
      for (i=0;i<=n-1;i++) {
	if (fscanf(stdin,"%hu",&current_ngram_int[i]) != 1) {
	  if (!rr_feof(stdin)) {
	    quit(-1,"Error reading id_ngram.\n");
	  }
	  else {
	    got_to_eof = 1;
	  }
	}
      }
      if (fscanf(stdin,"%d",&current_count) != 1) {
	if (!rr_feof(stdin)) {
	  quit(-1,"Error reading id_ngram.\n");
	}
	else {
	  got_to_eof = 1;
	}
      }
      break;
    case WORDS:
      for (i=0;i<=n-1;i++) {
	if (fscanf(stdin,"%s",current_ngram_text[i]) != 1) {
	  if (!rr_feof(stdin)) {
	    quit(-1,"Error reading id_ngram.\n");
	  }
	  else {
	    got_to_eof = 1;
	  }
	}
      }
      if (fscanf(stdin,"%d",&current_count) != 1) {
	if (!rr_feof(stdin)) {
	  quit(-1,"Error reading id_ngram.\n");
	}
	else {
	  got_to_eof = 1;
	}
      }
      break;
    }

    if (!got_to_eof) {

      /* Check for correct sorting */

      if (!first_one) {

	switch(storage_type) {
	case NUMERIC:
	  for (i=0;i<=n-1;i++) {
	    if (current_ngram_int[i]<previous_ngram_int[i]) {
	      quit(-1,"Error : ngrams not correctly sorted.\n");
	    }
	    else {
	      if (current_ngram_int[i]>previous_ngram_int[i]) {
		i=n;
	      }
	    }
	  }
	  break;
	case ALPHA:
	  for (i=0;i<=n-1;i++) {
	    if (strcmp(current_ngram_text[i],previous_ngram_text[i])<0) {
	      quit(-1,"Error : ngrams not correctly sorted.\n");
	    }
	    else {
	      if (strcmp(current_ngram_text[i],previous_ngram_text[i])>0) {
		i=n;
	      }
	    }
	  }
	  break;
	}
      }

      /* Compare this m-gram with previous m-gram */

      if (!first_one) {

	switch(storage_type) {
	case NUMERIC:
	  same = 1;
	  for (i=0;i<=m-1;i++) {
	    if (current_ngram_int[i] != previous_ngram_int[i]) {
	      same = 0;
	    }
	  }
	  if (same) {
	    running_total += current_count;
	  }
	  else {
	    if (input_type == ASCII) {
	      for (i=0;i<=m-1;i++) {
		printf("%d ",previous_ngram_int[i]);
	      }
	      printf("%d\n",running_total);
	    }
	    else {
	      for (i=0;i<=m-1;i++) {
		rr_fwrite(&previous_ngram_int[i],sizeof(id__t),1,stdout,
			  "to id_ngrams at stdout");
	      }
	      rr_fwrite(&running_total,sizeof(count_t),1,stdout,
			"to id n-grams at stdout");
	    }
	    running_total = current_count;
	  }
	  break;
	case ALPHA:
	  same = 1;
	  for (i=0;i<=m-1;i++) {
	    if (strcmp(current_ngram_text[i],previous_ngram_text[i])) {
	      same = 0;
	    }
	  }
	  if (same) {
	    running_total += current_count;
	  }
	  else {
	    for (i=0;i<=m-1;i++) {
	      printf("%s ",previous_ngram_text[i]);
	    }
	    printf("%d\n",running_total);
	    running_total = current_count;
	  
	  }
	  break;
	}
      
      }
      else {
	running_total = current_count;
      } 
    
      first_one = 0;
    
    }
  }

  /* Write out final m-gram */

  switch(input_type) {
  case BINARY:
    break;
  case ASCII:
    for (i=0;i<=m-1;i++) {
      printf("%d ",previous_ngram_int[i]);
    }
    printf("%d\n",running_total);
    break;
  case WORDS:
    for (i=0;i<=m-1;i++) {
      printf("%s ",previous_ngram_text[i]);
    }
    printf("%d\n",running_total);
    break;
  } 

  pc_message(verbosity,0,"ngram2mgram : Done.\n");

  exit(0);

}	  
