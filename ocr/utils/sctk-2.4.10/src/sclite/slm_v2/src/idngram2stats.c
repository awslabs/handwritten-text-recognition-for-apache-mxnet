
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

#include <stdlib.h>
#include <stdio.h>
#include "pc_libs/pc_general.h"
#include "toolkit.h"
#include "rr_libs/general.h"

typedef unsigned short id__t;

typedef struct {
  unsigned short n;
  id__t          *id_array;
  int            count;
} ngram;



int get_ngram(FILE *id_ngram_fp, ngram *ng, flag ascii);

void main (int argc, char **argv) {

  flag first_ngram;
  int n;
  int fof_size;
  flag is_ascii;
  int verbosity;
  int **fof_array;
  int *num_kgrams;
  ngram current_ngram;
  ngram previous_ngram;
  int *ng_count;
  int pos_of_novelty;
  int nlines;
  int i;
  int j;
  int t;

  pos_of_novelty = n; /* Simply for warning-free compilation */

  report_version(&argc,argv);

  if (argc == 1 || pc_flagarg(&argc, argv,"-help")) {
    fprintf(stderr,"indngram2stats : Report statistics for an id n-gram file.\n");
    fprintf(stderr,"Usage : idngram2stats [ -n 3 ] \n");
    fprintf(stderr,"                      [ -fof_size 50 ]\n");
    fprintf(stderr,"                      [ -verbosity %d ]\n",
	    DEFAULT_VERBOSITY);
    fprintf(stderr,"                      [ ascii_input ] \n");
    fprintf(stderr,"                      < .idngram > .stats\n");
    exit(1);
  }

  is_ascii = pc_flagarg(&argc, argv,"-ascii_input");
  n = pc_intarg(&argc, argv,"-n",3);
  fof_size = pc_intarg(&argc, argv,"-fof_size",50);
  verbosity = pc_intarg(&argc, argv,"-verbosity",DEFAULT_VERBOSITY);

  pc_report_unk_args(&argc,argv,verbosity);

  pc_message(verbosity,2,"n        = %d\n",n);
  pc_message(verbosity,2,"fof_size = %d\n",fof_size);

  current_ngram.n = n;
  previous_ngram.n = n;
  
  pos_of_novelty = n; /* Simply for warning-free compilation */

  fof_array = (int **) rr_malloc(sizeof(int *)*(n-1));
  for (i=0;i<=n-2;i++) {
    fof_array[i] = (int *) rr_calloc(fof_size+1,sizeof(int));
  }

  num_kgrams = (int *) rr_calloc(n-1,sizeof(int));
  ng_count = (int *) rr_calloc(n-1,sizeof(int));

  current_ngram.id_array = (id__t *) rr_calloc(n,sizeof(id__t));
  previous_ngram.id_array = (id__t *) rr_calloc(n,sizeof(id__t));

  pc_message(verbosity,2,"Processing id n-gram file.\n");
  pc_message(verbosity,2,"20,000 n-grams processed for each \".\", 1,000,000 for each line.\n");

  nlines = 0;
  first_ngram = 1;
  
  while (!rr_feof(stdin)) {
    for (i=0;i<=n-1;i++) {
      previous_ngram.id_array[i]=current_ngram.id_array[i];
    }

    if (get_ngram(stdin,&current_ngram,is_ascii)) {

      nlines++;
      if (nlines % 20000 == 0) {
	if (nlines % 1000000 == 0) {
	  pc_message(verbosity,2,".\n");
	}
	else {
	  pc_message(verbosity,2,".");
	}
      }
    
      /* Test for where this ngram differs from last - do we have an
	 out-of-order ngram? */
      
      pos_of_novelty = n;

      for (i=0;i<=n-1;i++) {
	if (current_ngram.id_array[i] > previous_ngram.id_array[i]) {
	  pos_of_novelty = i;
	  i=n;
	}
	else {
	  if (current_ngram.id_array[i] < previous_ngram.id_array[i]) {
	    quit(-1,"Error : n-grams are not correctly ordered.\n");
	  }
	}
      }
      
      if (pos_of_novelty == n && nlines != 1) {
	quit(-1,"Error : Repeated ngram in idngram stream.\n");
      }

      /* Add new N-gram */
     
      num_kgrams[n-2]++;
      if (current_ngram.count <= fof_size) {
	fof_array[n-2][current_ngram.count]++;
      }

      if (!first_ngram) {
	for (i=n-2;i>=MAX(1,pos_of_novelty);i--) {
	  num_kgrams[i-1]++;
	  if (ng_count[i-1] <= fof_size) {
	    fof_array[i-1][ng_count[i-1]]++;
	  }
	  ng_count[i-1] = current_ngram.count;
	}
      }
      else {
	for (i=n-2;i>=MAX(1,pos_of_novelty);i--) {
	  ng_count[i-1] = current_ngram.count;
	}
	first_ngram = 0;
      }
	
      for (i=0;i<=pos_of_novelty-2;i++) {
	ng_count[i] += current_ngram.count;
      }
    }
  }

  /* Process last ngram */

  for (i=n-2;i>=MAX(1,pos_of_novelty);i--) {
    num_kgrams[i-1]++;
    if (ng_count[i-1] <= fof_size) {
      fof_array[i-1][ng_count[i-1]]++;
    }
    ng_count[i-1] = current_ngram.count;
  }
  
  for (i=0;i<=pos_of_novelty-2;i++) {
    ng_count[i] += current_ngram.count;
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

  pc_message(verbosity,0,"idngram2stats : Done.\n");

  exit(0);
  
}
