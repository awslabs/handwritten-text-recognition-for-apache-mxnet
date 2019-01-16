
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

#define DEFAULT_HASH_SIZE 200000
#define DEFAULT_MAX_FILES 20
#define MAX_N 20
#define TEMP_FILE_ROOT "text2idngram.temp."

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/utsname.h>
#include <unistd.h>
#include "toolkit.h"
#include "rr_libs/general.h"
#include "pc_libs/pc_general.h"
#include "idngram.h"

void add_to_buffer(unsigned short word_index,
		   int ypos,
		   int xpos, 
		   unsigned short *buffer) {

  
  buffer[(n*ypos)+xpos] = word_index;

}

unsigned short buffer_contents(int ypos,
			       int xpos, 
			       unsigned short *buffer) {
  
  return (buffer[(n*ypos)+xpos]);

}



/***************************
      MAIN FUNCTION
 ***************************/

void main(int argc, char *argv[]) {

  int i,j;

  char *vocab_filename;
  FILE *tempfile;
  char tempfiles_directory[1000];
  int vocab_size;
  FILE *vocab_file;

  int verbosity;

  int buffer_size;
  int position_in_buffer;
  int number_of_tempfiles;
  int max_files;
  int fof_size;

  unsigned short *buffer;
  unsigned short *placeholder;
  unsigned short *temp_ngram;
  int temp_count;
  
  char temp_word[500];
  char temp_word2[500];

  char *temp_file_root;
  char *temp_file_ext;
  char *host_name;
  int proc_id;
  struct utsname uname_info;

  flag write_ascii;

  /* Vocab hash table things */

  struct hash_table vocabulary;
  unsigned long hash_size;
  unsigned long M;

  tempfile = NULL; /* Just to prevent compilation warnings. */

  report_version(&argc,argv);

  verbosity = pc_intarg(&argc,argv,"-verbosity",DEFAULT_VERBOSITY);

  /* Process command line */
  
  if (pc_flagarg( &argc, argv,"-help") || argc==1) {
    fprintf(stderr,"text2idngram - Convert a text stream to an id n-gram stream.\n");
    fprintf(stderr,"Usage : text2idngram  -vocab .vocab \n");
    fprintf(stderr,"                    [ -buffer 100 ]\n");
    fprintf(stderr,"                    [ -hash %d ]\n",DEFAULT_HASH_SIZE);
    fprintf(stderr,"                    [ -temp %s ]\n",DEFAULT_TEMP);
    fprintf(stderr,"                    [ -files %d ]\n",DEFAULT_MAX_FILES);
    fprintf(stderr,"                    [ -gzip | -compress ]\n");
    fprintf(stderr,"                    [ -verbosity %d ]\n",
	    DEFAULT_VERBOSITY);
    fprintf(stderr,"                    [ -n 3 ]\n");
    fprintf(stderr,"                    [ -write_ascii ]\n");
    fprintf(stderr,"                    [ -fof_size 10 ]\n");
    exit(1);
  }

  pc_message(verbosity,2,"text2idngram\n");

  n = pc_intarg( &argc, argv, "-n",DEFAULT_N);

  placeholder = (unsigned short *) rr_malloc(sizeof(unsigned short)*n);
  temp_ngram = (unsigned short *) rr_malloc(sizeof(unsigned short)*n);
  hash_size = pc_intarg( &argc, argv, "-hash",DEFAULT_HASH_SIZE);
  buffer_size = pc_intarg( &argc, argv, "-buffer",STD_MEM);

  write_ascii = pc_flagarg(&argc,argv,"-write_ascii");

  fof_size = pc_intarg(&argc,argv,"-fof_size",10);

  max_files = pc_intarg( &argc, argv, "-files",DEFAULT_MAX_FILES);

  vocab_filename = salloc(pc_stringarg( &argc, argv, "-vocab", "" ));
  
  if (!strcmp("",vocab_filename)) {
    quit(-1,"text2idngram : Error : Must specify a vocabulary file.\n");
  }
    
  strcpy(tempfiles_directory,pc_stringarg( &argc, argv, "-temp", 
					   DEFAULT_TEMP));

  if (pc_flagarg(&argc,argv,"-compress")) {
    temp_file_ext = salloc(".Z");
  }
  else {
    if (pc_flagarg(&argc,argv,"-gzip")) {
      temp_file_ext = salloc(".gz");
    }
    else {
      temp_file_ext = salloc("");
    }
  }

  uname(&uname_info);

  host_name = salloc(uname_info.nodename);

  proc_id = getpid();

  sprintf(temp_word,"%s%s.%d.",TEMP_FILE_ROOT,host_name,proc_id);

  temp_file_root = salloc(temp_word);

  pc_report_unk_args(&argc,argv,verbosity);
  
  /* If the last charactor in the directory name isn't a / then add one. */
  
  if (tempfiles_directory[strlen(tempfiles_directory)-1] != '/') {
    strcat(tempfiles_directory,"/");
  }
  
  pc_message(verbosity,2,"Vocab                  : %s\n",vocab_filename);
  pc_message(verbosity,2,"N-gram buffer size     : %d\n",buffer_size);
  pc_message(verbosity,2,"Hash table size        : %d\n",hash_size);
  pc_message(verbosity,2,"Temp directory         : %s\n",tempfiles_directory);
  pc_message(verbosity,2,"Max open files         : %d\n",max_files);
  pc_message(verbosity,2,"FOF size               : %d\n",fof_size);  
  pc_message(verbosity,2,"n                      : %d\n",n);

  buffer_size *= (1000000/(sizeof(unsigned short)*n));

  /* Allocate memory for hash table */

  fprintf(stderr,"Initialising hash table...\n");

  M = nearest_prime(hash_size);

  new_hashtable(&vocabulary,M);

  /* Read in the vocabulary */

  vocab_size = 0;

  vocab_file = rr_iopen(vocab_filename);

  pc_message(verbosity,2,"Reading vocabulary...\n");

  while (fgets (temp_word, sizeof(temp_word),vocab_file)) {
    if (strncmp(temp_word,"##",2)==0) continue;
    sscanf (temp_word, "%s ",temp_word2);

    /* Check for repeated words in the vocabulary */

    if (index2(&vocabulary,temp_word2) != 0) {
      fprintf(stderr,"======================================================\n");
      fprintf(stderr,"WARNING: word %s is repeated in the vocabulary.\n",temp_word);
      fprintf(stderr,"=======================================================\n");
    }
    if (strncmp(temp_word,"#",1)==0) {
      fprintf(stderr,"\n\n===========================================================\n");
      fprintf(stderr,":\nWARNING: line assumed NOT a comment:\n");
      fprintf(stderr,     ">>> %s <<<\n",temp_word);
      fprintf(stderr,     "         '%s' will be included in the vocabulary.\n",temp_word2);
      fprintf(stderr,     "         (comments must start with '##')\n");
      fprintf(stderr,"===========================================================\n\n");
    }
    vocab_size++;
    add_to_hashtable(&vocabulary,hash(temp_word2,M),temp_word2,vocab_size);
  }

  if (vocab_size > MAX_VOCAB_SIZE) {
    quit(-1,"text2idngram : Error : Vocabulary size exceeds maximum.\n");
  }   
  
  pc_message(verbosity,2,"Allocating memory for the n-gram buffer...\n");

  buffer=(unsigned short*) rr_malloc(n*(buffer_size+1)*sizeof(unsigned short));

  number_of_tempfiles = 0;

  /* Read text into buffer */

  /* Read in the first ngram */

  position_in_buffer = 0;

  for (i=0;i<=n-1;i++) {
    get_word(stdin,temp_word);
    add_to_buffer(index2(&vocabulary,temp_word),0,i,buffer);
  }

  while (!rr_feof(stdin)) {

    /* Fill up the buffer */

    pc_message(verbosity,2,"Reading text into the n-gram buffer...\n");
    pc_message(verbosity,2,"20,000 n-grams processed for each \".\", 1,000,000 for each line.\n");
    while ((position_in_buffer<buffer_size) && (!rr_feof(stdin))) {
      position_in_buffer++;
      if (position_in_buffer % 20000 == 0) {
	if (position_in_buffer % 1000000 == 0) {
	  pc_message(verbosity,2,".\n");
	}
	else {
	  pc_message(verbosity,2,".");
	}
      }
      for (i=1;i<=n-1;i++) {
	add_to_buffer(buffer_contents(position_in_buffer-1,i,buffer),
		      position_in_buffer,i-1,buffer);
      }
      if (get_word(stdin,temp_word) == 1) {
	add_to_buffer(index2(&vocabulary,temp_word),position_in_buffer,
		      n-1,buffer);
      }
    }

    for (i=0;i<=n-1;i++) {
      placeholder[i] = buffer_contents(position_in_buffer,i,buffer);
    }

    /* Sort buffer */
    
    pc_message(verbosity,2,"\nSorting n-grams...\n");
    
    qsort((void*) buffer,(size_t) position_in_buffer,
	  n*sizeof(unsigned short),compare_ngrams);

    /* Output the buffer to temporary BINARY file */
    
    number_of_tempfiles++;

    sprintf(temp_word,"%s%s%hu%s",tempfiles_directory,temp_file_root,
	    number_of_tempfiles,temp_file_ext);

    pc_message(verbosity,2,"Writing sorted n-grams to temporary file %s\n",
	       temp_word);

    tempfile = rr_oopen(temp_word);

    for (i=0;i<=n-1;i++) {
      temp_ngram[i] = buffer_contents(0,i,buffer);
      if (temp_ngram[i] > MAX_VOCAB_SIZE) {
	quit(-1,"Invalid trigram in buffer.\nAborting");

      }
    }
    temp_count = 1;

    for (i=1;i<=position_in_buffer;i++) {
 
      if (!compare_ngrams(temp_ngram,&buffer[i*n])) {
	temp_count++;
      }
      else {
	for (j=0;j<=n-1;j++) {
	  rr_fwrite(&temp_ngram[j],sizeof(unsigned short),1,
		    tempfile,"temporary n-gram ids");
	  temp_ngram[j] = buffer_contents(i,j,buffer);
	}
	rr_fwrite(&temp_count,sizeof(int),1,tempfile,
		  "temporary n-gram counts");
	temp_count = 1;
      }
    }
    
    rr_oclose(tempfile);

    for (i=0;i<=n-1;i++) {
      add_to_buffer(placeholder[i],0,i,buffer);
    }

    position_in_buffer = 0;

  }

  /* Merge the temporary files, and output the result to standard output */

  pc_message(verbosity,2,"Merging temporary files...\n");
  
  merge_tempfiles(1,
		  number_of_tempfiles,
		  temp_file_root,
		  temp_file_ext,
		  max_files,
		  tempfiles_directory,
		  stdout,
		  write_ascii,
		  fof_size); 

  pc_message(verbosity,0,"text2idngram : Done.\n");

  exit(0);
  
}

