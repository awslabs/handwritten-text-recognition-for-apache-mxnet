
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

#define DEFAULT_MAX_FILES 20
#define TEMP_FILE_ROOT "text2wngram.tmp"

#include <sys/types.h>
#include <unistd.h>
#include <sys/utsname.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "toolkit.h"
#include "pc_libs/pc_general.h"
#include "rr_libs/general.h"

int cmp_strings(const void *string1,const void *string2) {
  
  char *s1;
  char *s2;
  
  s1 = *((char **) string1);
  s2 = *((char **) string2);

  return (strcmp(s1,s2));

}

void merge_tempfiles (int start_file, 
		      int end_file, 
		      char *temp_file_root,
		      char *temp_file_ext,
		      int max_files,
		      char *tempfiles_directory, 
		      FILE *outfile,
		      int n,
		      int verbosity) {


  FILE *new_temp_file;
  char *new_temp_filename;
  
  FILE **temp_file;
  char **temp_filename;
  char **current_ngram;
  char smallest_ngram[1000];
  int *current_ngram_count;
  flag *finished;
  flag all_finished;
  int temp_count;
  char temp_word[500];
  int i,j;
  
  pc_message(verbosity,2,"Merging temp files %d through %d...\n", start_file,
 	  end_file);
   /*
    * If we try to do more than max_files, then merge into groups,
    * then merge groups recursively.
    */
    if (end_file-start_file+1 > max_files) {
       int new_start_file, new_end_file;
       int n_file_groups = 1 + (end_file-start_file)/max_files;
 
       fprintf(stderr, "%d files to do, in %d groups\n", end_file-start_file,
 	      n_file_groups);
 
       new_temp_filename = (char *) rr_malloc(300*sizeof(char));
 
       /*
        * These n_file_groups sets of files will be done in groups of
        * max_files batches each, as temp files numbered
        * end_file+1 ... end_file+n_file_groups,
        * and then these will be merged into the final result.
        */
 
       for (i = 0; i < n_file_groups; i++) {
 	  /* do files i*max_files through min((i+1)*max_files-1,end_file); */
 	  new_start_file = start_file + (i*max_files);
 	  new_end_file = start_file + ((i+1)*max_files) - 1;
 	  if (new_end_file > end_file) new_end_file = end_file;
 	  
 	  sprintf(new_temp_filename,
 		  "%s%s%hu%s",
 		  tempfiles_directory,
 		  temp_file_root,
 		  end_file+i+1,
 		  temp_file_ext);
 
 	  new_temp_file = rr_oopen(new_temp_filename);
 
 	  merge_tempfiles(new_start_file,
 			  new_end_file,
 			  temp_file_root,
			  temp_file_ext,
 			  max_files,
 			  tempfiles_directory,
 			  new_temp_file,
 			  n,
			  verbosity);
 
 	  rr_iclose(new_temp_file);
 
       }
 
       merge_tempfiles(end_file+1,
		       end_file+n_file_groups,
		       temp_file_root,
		       temp_file_ext,
		       max_files,
		       tempfiles_directory,
		       outfile,
		       n,
		       verbosity);
 
       return;
    }
    
   /*
    * We know we are now doing <= max_files.
    */
 
   temp_file = (FILE **) rr_malloc((end_file+1)*sizeof(FILE *));
   temp_filename = (char **) rr_malloc((end_file+1)*sizeof(char *));
   for (i=start_file;i<=end_file;i++) {
     temp_filename[i] = (char *) rr_malloc(300*sizeof(char));
   }
   current_ngram = (char **) rr_malloc((end_file+1)*sizeof(char *));
   for (i=start_file;i<=end_file;i++) {
     current_ngram[i] = (char *) rr_malloc(1000*sizeof(char));
   }
   current_ngram_count = (int *) rr_malloc((end_file+1)*sizeof(int));
   finished = (flag *) rr_malloc(sizeof(flag)*(end_file+1));
  
   /* Open all the temp files for reading */
   for (i=start_file;i<=end_file;i++) {
     sprintf(temp_filename[i],"%s%s%hu%s",tempfiles_directory,
	     temp_file_root,i,temp_file_ext);
     temp_file[i] = rr_iopen(temp_filename[i]);
   }
 
   /* Now go through the files simultaneously, and write out the appropriate
      ngram counts to the output file. */
 
   for (i=start_file;i<=end_file;i++) {
     finished[i] = 0;
     if (!rr_feof(temp_file[i])) {
       for (j=0;j<=n-1;j++) {
 	if (fscanf(temp_file[i],"%s",temp_word) != 1) {
 	  if (!rr_feof(temp_file[i])) {
 	    quit(-1,"Error reading temp file %s\n",temp_filename[i]);
 	  }
 	}
 	else {
 	  if (j==0) {
 	    strcpy(current_ngram[i],temp_word);
  	  }
  	  else {
 	    strcat(current_ngram[i]," ");
 	    strcat(current_ngram[i],temp_word);
  	  }
  	}
       }
       if (fscanf(temp_file[i],"%d",&current_ngram_count[i]) != 1) {
	 if (!rr_feof(temp_file[i])) {
	   quit(-1,"Error reading temp file %s\n",temp_filename[i]);
	 }
       }
     }
   }
   
   all_finished = 0;
   
   while (!all_finished) {
  
     /* Find the smallest current ngram */
 
     strcpy(smallest_ngram,"");
 
     for (i=start_file;i<=end_file;i++) {
       if (!finished[i]) {
	 if (strcmp(smallest_ngram,current_ngram[i]) > 0 ||
	     (smallest_ngram[0] == '\0')) {
	   strcpy(smallest_ngram,current_ngram[i]);
	 }
       }
     }
     
     /* For each of the files that are currently holding this ngram,
        add its count to the temporary count, and read in a new ngram
        from the files. */
  
     temp_count = 0;
 
     for (i=start_file;i<=end_file;i++) {
       if (!finished[i]) {
 	if (!strcmp(smallest_ngram,current_ngram[i])) {
 	  temp_count += current_ngram_count[i];
 	  if (!rr_feof(temp_file[i])) {
 	    for (j=0;j<=n-1;j++) {
 	      if (fscanf(temp_file[i],"%s",temp_word) != 1) {
 		if (!rr_feof(temp_file[i])) {
 		  quit(-1,"Error reading temp file %s\n",temp_filename[i]);
 		}
 	      }
 	      else {
 		if (j==0) {
 		  strcpy(current_ngram[i],temp_word);
  		}
  		else {
 		  strcat(current_ngram[i]," ");
 		  strcat(current_ngram[i],temp_word);
  		}
  	      }
 	    }
 	    if (fscanf(temp_file[i],"%d",&current_ngram_count[i]) != 1) {
 	      if (!rr_feof(temp_file[i])) {
 		quit(-1,"Error reading temp file count %s\n",
 		     temp_filename[i]);
  	      }
  	    }
 	  }
 
 	  /*
 	   * PWP: Note that the fscanf may have changed the state of
 	   * temp_file[i], so we re-ask the question rather than just
 	   * doing an "else".
 	   */
 	  if (rr_feof(temp_file[i])) {
 	    finished[i] = 1;
 	    all_finished = 1;
 	    for (j=start_file;j<=end_file;j++) {
 	      if (!finished[j]) {
 		all_finished = 0;
  	      }
  	    }
  	  }
  	}
        }
      }
 
     /*
      * PWP: We cannot conditionalize this on (!all_finished) because
      * if we do we may have lost the very last count.  (Consider the
      * case when several files have ran out of data, but the last
      * couple have the last count in them.)
      */
     if (fprintf(outfile,"%s %d\n",smallest_ngram,temp_count) < 0) {
       quit(-1,"Write error encountered while attempting to merge temporary files.\nAborting, but keeping temporary files.\n");
     }
   }
 
   for (i=start_file;i<=end_file;i++) {
     rr_iclose(temp_file[i]);
     remove(temp_filename[i]);
   }
    
    free(temp_file);
   for (i=start_file;i<=end_file;i++) {
      free(temp_filename[i]);
    }
    free(temp_filename);  
   for (i=start_file;i<=end_file;i++) {
      free(current_ngram[i]);
    }
    free(current_ngram);

  free(current_ngram_count);
  free(finished);
}



void main (int argc, char **argv) {

  int n;
  int verbosity;
  int max_files;
  int max_words;
  int max_chars;
  char temp_directory[1000];

  int current_word;
  int current_char;
  int start_char;		/* start boundary (possibly > than 0) */

  int no_of_spaces;
  int pos_in_string;

  int i;
  char *current_string;
  char current_temp_filename[500];
  int current_file_number;
  FILE *temp_file;

  flag text_buffer_full;

  char *text_buffer;
  char **pointers;

  char current_ngram[500];
  int current_count;

  int counter;

  struct utsname uname_info;
  char *temp_file_root;
  char *temp_file_ext;
  char *host_name;
  int proc_id;
  char temp_word[500];

  flag words_set;
  flag chars_set;

  /* Process command line */

  verbosity = pc_intarg(&argc, argv,"-verbosity",DEFAULT_VERBOSITY);
  pc_message(verbosity,2,"text2wngram\n");

  report_version(&argc,argv);

  if (pc_flagarg( &argc, argv,"-help")) {
    fprintf(stderr,"text2wngram - Convert a text stream to a word n-gram stream.\n");
    fprintf(stderr,"Usage : text2wngram [ -n 3 ]\n");
    fprintf(stderr,"                    [ -temp %s ]\n",DEFAULT_TEMP);
    fprintf(stderr,"                    [ -chars %d ]\n",STD_MEM*7000000/11);
    fprintf(stderr,"                    [ -words %d ]\n",STD_MEM*1000000/11);
    fprintf(stderr,"                    [ -gzip | -compress ]\n");
    fprintf(stderr,"                    [ -verbosity 2 ]\n");
    fprintf(stderr,"                    < .text > .wngram\n");
    exit(1);
  }

  n = pc_intarg(&argc, argv,"-n",DEFAULT_N);

  /*  max_words = pc_intarg(&argc, argv,"-words",STD_MEM*1000000/11);
  max_chars = pc_intarg(&argc, argv,"-chars",STD_MEM*7000000/11); */

  max_words = pc_intarg(&argc, argv,"-words",-1);
  max_chars = pc_intarg(&argc, argv,"-chars",-1);

  if (max_words == -1) {
    words_set = 0;
    max_words = STD_MEM*1000000/11;
  }
  else {
    words_set = 1;
  }

  if (max_chars == -1) {
    chars_set = 0;
    max_chars = STD_MEM*7000000/11; 
  }
  else {
    chars_set = 1;
  }
  
  max_files = pc_intarg(&argc, argv,"-files",DEFAULT_MAX_FILES);

  strcpy(temp_directory,pc_stringarg( &argc, argv, "-temp", DEFAULT_TEMP));


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
 
  if (words_set && !chars_set) {
    max_chars = max_words * 7;
  }

  if (!words_set && chars_set) {
    max_words = max_chars / 7;
  }

  /* If the last charactor in the directory name isn't a / then add one. */
  
  if (temp_directory[strlen(temp_directory)-1] != '/') {
    strcat(temp_directory,"/");
  }
  pc_message(verbosity,2,"n = %d\n",n);
  pc_message(verbosity,2,"Number of words in buffer = %d\n",max_words);
  pc_message(verbosity,2,"Number of chars in buffer = %d\n",max_chars);
  pc_message(verbosity,2,"Max number of files open at once = %d\n",max_files);
  pc_message(verbosity,2,"Temporary directory = %s\n",temp_directory);

  /* Allocate memory for the buffers */

  text_buffer = (char *) rr_malloc(sizeof(char)*max_chars);
  pc_message(verbosity,2,"Allocated %d bytes to text buffer.\n",
	     sizeof(char)*max_chars);

  pointers = (char **) rr_malloc(sizeof(char *)*max_words);
  pc_message(verbosity,2,"Allocated %d bytes to pointer array.\n",
	     sizeof(char *)*max_words);

  current_file_number = 0;

  current_word = 1;
  start_char = 0;
  current_char = 0;
  counter = 0;
  pointers[0] = text_buffer;
      
  while (!feof(stdin)) {

    current_file_number++;

    /* Read text into buffer */
    
    pc_message(verbosity,2,"Reading text into buffer...\n");

    pc_message(verbosity,2,"Reading text into the n-gram buffer...\n");
    pc_message(verbosity,2,"20,000 words processed for each \".\", 1,000,000 for each line.\n");
    
    pointers[0] = text_buffer;
    
    while ((!rr_feof(stdin)) && 
	   (current_word < max_words) && 
	   (current_char < max_chars)) {

      text_buffer[current_char] = getchar();
      if (text_buffer[current_char] == '\n' || 
	  text_buffer[current_char] == '\t' ) {
	text_buffer[current_char] = ' ';
      }
      if (text_buffer[current_char] == ' ') {
	if (current_char > start_char) {
	  if (text_buffer[current_char-1] == ' ') {
	    current_word--;
	    current_char--;
	  }
	  pointers[current_word] = &(text_buffer[current_char+1]);
	  current_word++; 
	  counter++;
	  if (counter % 20000 == 0) {
	    if (counter % 1000000 == 0) {
	      pc_message(verbosity,2,"\n");
	    }
	    else {
	      pc_message(verbosity,2,".");
	    }
	  }
	}
      }
      
      if (text_buffer[current_char] != ' ' ||
	  current_char > start_char) {
	current_char++;
      }
    }

    text_buffer[current_char]='\0';


    if (current_word == max_words || rr_feof(stdin)) {
      for (i=current_char+1;i<=max_chars-1;i++) {
	text_buffer[i] = ' ';
      }
      text_buffer_full = 0;
    }
    else {
      text_buffer_full = 1;
    }
    
    /* Sort buffer */

    pc_message(verbosity,2,"\nSorting pointer array...\n"); 

    qsort((void *) pointers,(size_t) current_word-n,sizeof(char *),cmp_strings);
   
    /* Write out temporary file */

    sprintf(current_temp_filename,"%s%s%hu%s",temp_directory,temp_file_root,current_file_number,temp_file_ext);

    pc_message(verbosity,2,"Writing out temporary file %s...\n",current_temp_filename);
        
    temp_file = rr_oopen(current_temp_filename);
    text_buffer[current_char] = ' ';
    
    current_count = 0;
    strcpy(current_ngram,"");
    
    for (i = 0; i <= current_word-n; i++) {
      current_string = pointers[i];
      
      /* Find the nth space */

      no_of_spaces = 0;
      pos_in_string = 0;
      while (no_of_spaces < n) {
	
	if (current_string[pos_in_string] == ' ') {
	  no_of_spaces++;
	}
	pos_in_string++;
      }
      
      if (!strncmp(current_string,current_ngram,pos_in_string)) {
	current_count++;
      }
      else {
	if (strcmp(current_ngram,"")) {
	  if (fprintf(temp_file,"%s %d\n",current_ngram,current_count) < 0)  {
	    quit(-1,"Error writing to temporary file %s\n",current_temp_filename);
	  }
	}
	current_count = 1;
	strncpy(current_ngram,current_string,pos_in_string);
	current_ngram[pos_in_string] = '\0';
      }
    }
    
    rr_oclose(temp_file);

    /* Move the last n-1 words to the beginning of the buffer, and set
       correct current_word and current_char things */

    strcpy(text_buffer,pointers[current_word-n]);
    pointers[0]=text_buffer;
   
    /* Find the (n-1)th space */

    no_of_spaces=0;
    pos_in_string=0;

    if (!text_buffer_full){ 
      while (no_of_spaces<(n-1)) {
	if (pointers[0][pos_in_string]==' ') {
	  no_of_spaces++;
	  pointers[no_of_spaces] = &pointers[0][pos_in_string+1];
	}
	pos_in_string++;
      }
    }
    else {
      while (no_of_spaces<n) {
	if (pointers[0][pos_in_string]==' ') {
	  no_of_spaces++;
	  pointers[no_of_spaces] = &pointers[0][pos_in_string+1];
	}
	pos_in_string++;
      }
      pos_in_string--;
    }

    current_char = pos_in_string;
    current_word = n;
    /* mark boundary beyond which counting pass cannot backup */
    start_char = current_char;

  }
  /* Merge temporary files */

  pc_message(verbosity,2,"Merging temporary files...\n");

  merge_tempfiles(1,
		  current_file_number,
		  temp_file_root,
		  temp_file_ext,
		  max_files,
		  temp_directory,
		  stdout,
		  n,
		  verbosity); 
  pc_message(verbosity,0,"text2wngram : Done.\n");

  exit(0);

}

