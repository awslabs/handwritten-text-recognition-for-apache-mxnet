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


/* Stuff that is common to both text2idngram and wngram2idngram */


struct node {
  char *word;
  unsigned short ind;
  struct node *next;
};

struct hash_table {

  int size;
  struct node **chain;

};


int n; /* Declare it globally, so doesn't need to be passed
		     as a parameter to compare_ngrams. which might
		     cause qsort to choke */


int compare_ngrams(const void *ngram1,
		   const void *ngram2) {

  int i;

  unsigned short *ngram_pointer1;
  unsigned short *ngram_pointer2;

  ngram_pointer1 = (unsigned short *) ngram1;
  ngram_pointer2 = (unsigned short *) ngram2;
 

  for (i=0;i<=n-1;i++) {
    if (ngram_pointer1[i]<ngram_pointer2[i]) {
      return(-1);
    }
    else {
      if (ngram_pointer1[i]>ngram_pointer2[i]) {
	return(1);
      }
    }
  }

  return(0);

}



int get_word( FILE *fp , char *word ) {

  /* read word from stream, checking for read errors and EOF */

  int nread;
  int rt_val;

  rt_val = 0;

  nread = fscanf( fp, "%s", word );
  if ( nread == 1 ) {
    rt_val = 1;
  }
  else {
    if ( nread != EOF ) {
      quit(-1, "Error reading file" );
    }
  }
  return(rt_val);
}


void merge_tempfiles (int start_file, 
		      int end_file, 
		      char *temp_file_root,
		      char *temp_file_ext,
		      int max_files,
		      char *tempfiles_directory, 
		      FILE *outfile,
		      flag write_ascii,
		      int fof_size) {

  FILE *new_temp_file;
  char temp_string[1000];
  char *new_temp_filename;
  
  FILE **temp_file;
  char **temp_filename;
  unsigned short **current_ngram;
  int *current_ngram_count;
  flag *finished;
  flag all_finished;
  unsigned short *smallest_ngram;
  int temp_count;
  int i,j,t;

  flag first_ngram;
  int **fof_array;
  int *num_kgrams;
  unsigned short *previous_ngram;
  int *ng_count;
  int pos_of_novelty;
  
  pos_of_novelty = n; /* Simply for warning-free compilation */
  num_kgrams = (int *) rr_calloc(n-1,sizeof(int));
  ng_count = (int *) rr_calloc(n-1,sizeof(int));
  first_ngram = 1;
  
  previous_ngram = (unsigned short *) rr_calloc(n,sizeof(unsigned short));
  temp_file = (FILE **) rr_malloc(sizeof(FILE *) * (end_file-start_file+1));
  temp_filename = (char **) rr_malloc(sizeof(char *) * 
				      (end_file-start_file+1));
  current_ngram = (unsigned short **) rr_malloc(sizeof(unsigned short *) * 
						(end_file-start_file+1));

  for (i=0;i<=end_file-start_file;i++) {
    current_ngram[i] = (unsigned short *) rr_malloc(sizeof(unsigned short)*n);
  }

  current_ngram_count = (int *) rr_malloc(sizeof(int)*(end_file-start_file+1));
  finished = (flag *) rr_malloc(sizeof(flag)*(end_file-start_file+1));
  smallest_ngram = (unsigned short *) rr_malloc(sizeof(unsigned short)*n);
  fof_array = (int **) rr_malloc(sizeof(int *)*(n-1));
  for (i=0;i<=n-2;i++) {
    fof_array[i] = (int *) rr_calloc(fof_size+1,sizeof(int));
  }


  if (end_file-start_file+1 > max_files) {
    sprintf(temp_string,"%s%s%hu%s",tempfiles_directory,temp_file_root,
	    end_file+1,temp_file_ext);
    new_temp_filename = salloc(temp_string);
    new_temp_file = rr_oopen(new_temp_filename);
    merge_tempfiles(start_file,start_file+max_files-1,
		    temp_file_root,temp_file_ext,max_files,
		    tempfiles_directory,new_temp_file,write_ascii,0);
    merge_tempfiles(start_file+max_files,end_file+1,
		    temp_file_root,temp_file_ext,max_files,
		    tempfiles_directory,outfile,write_ascii,0);
  }
  
  else {

    /* Open all the temp files for reading */
    for (i=0;i<=end_file-start_file;i++) {
      sprintf(temp_string,"%s%s%hu%s",tempfiles_directory,temp_file_root,
	      i+start_file,temp_file_ext);
      temp_filename[i] = salloc(temp_string);
      temp_file[i] = rr_iopen(temp_filename[i]);
    }
    
    /* Now go through the files simultaneously, and write out the appropriate
       ngram counts to the output file. */

    for (i=end_file-start_file;i>=0;i--) {
      finished[i] = 0;
      if (!rr_feof(temp_file[i])) {
	for (j=0;j<=n-1;j++) {
	  rr_fread(&current_ngram[i][j], sizeof(unsigned short),1,
		   temp_file[i],"temporary n-gram ids",0);
	}    
	rr_fread(&current_ngram_count[i], sizeof(int),1,
		 temp_file[i],"temporary n-gram counts",0);
      }
    }
    
    all_finished = 0;

    while (!all_finished) {
 
      /* Find the smallest current ngram */

      for (i=0;i<=n-1;i++) {
	smallest_ngram[i] = MAX_VOCAB_SIZE;
      }

      for (i=0;i<=end_file-start_file;i++) {
	if (!finished[i]) {
	  if (compare_ngrams(smallest_ngram,current_ngram[i]) > 0) {
	    for (j=0;j<=n-1;j++) {
	      smallest_ngram[j] = current_ngram[i][j];
	    }
	  }
	}
      }

      for (i=0;i<=n-1;i++) {
	if (smallest_ngram[i] > MAX_VOCAB_SIZE) {
	  quit(-1,"Error : Temporary files corrupted, invalid n-gram found.\n");
	}
      }
	  
      /* For each of the files that are currently holding this ngram,
	 add its count to the temporary count, and read in a new ngram
	 from the files. */

      temp_count = 0;

      for (i=0;i<=end_file-start_file;i++) {
	if (!finished[i]) {
	  if (compare_ngrams(smallest_ngram,current_ngram[i]) == 0) {
	    temp_count = temp_count + current_ngram_count[i];
	    if (!rr_feof(temp_file[i])) {
	      for (j=0;j<=n-1;j++) {
		rr_fread(&current_ngram[i][j],sizeof(unsigned short),1,
			 temp_file[i],"temporary n-gram ids",0);
	      }
	      rr_fread(&current_ngram_count[i],sizeof(int),1,
		       temp_file[i],"temporary n-gram count",0);
	    }
	    else {
	      finished[i] = 1;
	      all_finished = 1;
	      for (j=0;j<=end_file-start_file;j++) {
		if (!finished[j]) {
		  all_finished = 0;
		}
	      }
	    }
	  }
	}
      }
      
      if (write_ascii) {
	for (i=0;i<=n-1;i++) {
	  if (fprintf(outfile,"%hu ",smallest_ngram[i]) < 0) {
	    quit(-1,"Write error encountered while attempting to merge temporary files.\nAborting, but keeping temporary files.\n");
	  }
	}
	if (fprintf(outfile,"%d\n",temp_count) < 0)  {
	  quit(-1,"Write error encountered while attempting to merge temporary files.\nAborting, but keeping temporary files.\n");
	}
      }
      else {
	for (i=0;i<=n-1;i++) {
	  rr_fwrite(&smallest_ngram[i],sizeof(unsigned short),1,
		    outfile,"n-gram ids");
	}
	rr_fwrite(&temp_count,sizeof(int),1,outfile,"n-gram counts");
	
	   
      }

      if (fof_size > 0 && n>1) { /* Add stuff to fof arrays */
	
	/* Code from idngram2stats */
	

	pos_of_novelty = n;
	  
	for (i=0;i<=n-1;i++) {
	  if (smallest_ngram[i] > previous_ngram[i]) {
	    pos_of_novelty = i;
	    i=n;
	  }
	}
	  
	/* Add new N-gram */
	  
	num_kgrams[n-2]++;
	if (temp_count <= fof_size) {
	  fof_array[n-2][temp_count]++;
	}

	if (!first_ngram) {
	  for (i=n-2;i>=MAX(1,pos_of_novelty);i--) {
	    num_kgrams[i-1]++;
	    if (ng_count[i-1] <= fof_size) {
	      fof_array[i-1][ng_count[i-1]]++;
	    }
	    ng_count[i-1] = temp_count;
	  }
	}
	else {
	  for (i=n-2;i>=MAX(1,pos_of_novelty);i--) {
	    ng_count[i-1] = temp_count;
	  }
	  first_ngram = 0;
	}
	  
	for (i=0;i<=pos_of_novelty-2;i++) {
	  ng_count[i] += temp_count;
	}
	for (i=0;i<=n-1;i++) {
	  previous_ngram[i]=smallest_ngram[i];
	}
	 
      }
    }
    
    for (i=0;i<=end_file-start_file;i++) {
      fclose(temp_file[i]);
      remove(temp_filename[i]); 
    }
    
  }    



  if (fof_size > 0 && n>1) { /* Display fof arrays */

    /* Process last ngram */

    for (i=n-2;i>=MAX(1,pos_of_novelty);i--) {
      num_kgrams[i-1]++;
      if (ng_count[i-1] <= fof_size) {
	fof_array[i-1][ng_count[i-1]]++;
      }
      ng_count[i-1] = temp_count;
    }
    
    for (i=0;i<=pos_of_novelty-2;i++) {
      ng_count[i] += temp_count;
    }

    for (i=0;i<=n-2;i++) {
      fprintf(stderr,"\n%d-grams occurring:\tN times\t\t> N times\tSug. -spec_num value\n",i+2);
      fprintf(stderr,"%7d\t\t\t\t\t\t%7d\t\t%7d\n",0,num_kgrams[i],((int)(num_kgrams[i]*1.01))+10);
      t = num_kgrams[i];
      for (j=1;j<=fof_size;j++) {
	t -= fof_array[i][j];
	fprintf(stderr,"%7d\t\t\t\t%7d\t\t%7d\t\t%7d\n",j,
		fof_array[i][j],t,((int)(t*1.01))+10);
      }
    }

  }

}


/* Hashing functions, by Gary Cook (gdc@eng.cam.ac.uk).  Could use the
   sih functions used in idngrma2lm, but these are much faster. */

/* return the nearest prime not smaller than 'num' */
int nearest_prime(int num)
{
  int div;
  int num_has_divisor = 1;
  
  if ( num / 2 * 2 == num ) num++; 
  for (; num_has_divisor; num += 2 ) {
     num_has_divisor=0;
     for ( div = 3; div <= num / 3; div++ ) {
        if ( ( num / div) * div == num ) {
           num_has_divisor = 1;
           break;
        }
     }
  }
  num -= 2;
  return( num );
}

/* generate a hash table address from a variable length character */
/* string - from R. Sedgewick, "Algorithms in C++". */
int hash( char *key, int M )
{
  unsigned int h; 
  char *t = key;

  for( h = 0; *t; t++ )
    h = ( 256 * h + *t ) % M;
  return h;
}

/* create a new node, and sets the index to ind */
struct node *new_node( char *key ,unsigned short ind)
{
  struct node *x;

  x = (struct node *) rr_malloc( sizeof( struct node ) );
  x->word = (char *) rr_malloc( (strlen( key ) + 1) * sizeof( char ) );
  strcpy( x->word, key );
  x->ind = ind;
  return x;
}

/* create hash table */
void new_hashtable( struct hash_table *table, int M )
{
  int i;
  
  table->size = M;
  table->chain = (struct node **) rr_malloc( M * sizeof( struct node *) );
  for( i = 0; i < M; i++ ) {
    table->chain[i] = new_node( "HEAD_NODE" , 0);
    table->chain[i]->next = (struct node *) NULL;
  }
}

/* update linked list */
int update_chain( struct node *t, char *key ,unsigned short ind)
{
  struct node *x;

  /* Move to end of list */ 

  while( t->next != NULL ) {
    t = t->next;
  }

  /* add node at end */
  x = new_node( key,ind );
  x->next = (struct node *) NULL;
  t->next = x;
  return 0;
}

void add_to_hashtable( struct hash_table *table,
		       unsigned long position,
		       char *vocab_item,
		       unsigned short ind) {

  update_chain( table->chain[position], vocab_item,ind );
}

unsigned short index2(struct hash_table *vocab,
		      char *word) {
  
  unsigned long chain;
  struct node *chain_pos;

  chain = hash( word, vocab->size );
  if ( chain < 0 || chain >= vocab->size ) {
    fprintf( stderr, "WARNING : invalid hash address\n" );
    fprintf( stderr, "%s ignored\n", word );
    return(0);
  }

  chain_pos = vocab->chain[chain];
  while (chain_pos->next != NULL) {
    if (strcmp(word,chain_pos->next->word) ) {
      chain_pos = chain_pos->next;
    }
    else {
      return (chain_pos->next->ind);
    }
  }
  return (0);

}

