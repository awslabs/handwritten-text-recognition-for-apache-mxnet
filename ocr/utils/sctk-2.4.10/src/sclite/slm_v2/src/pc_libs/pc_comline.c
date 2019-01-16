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




#include <string.h>
#include <stdlib.h>
#include "../rr_libs/general.h"
#include "pc_general.h"
#include "../toolkit.h"

/* update the command line argument sequence */
static void updateArgs( int *pargc, char **argv, int rm_cnt ) 
{
  int i ;                      
  /* update the argument count */
  (*pargc)-- ;
 
  /* update the command line */
  for( i = rm_cnt ; i < *pargc ; i++ ) argv[i] = argv[i+1] ;
}

int pc_flagarg(int *argc, char **argv, char *flag) {

  int i;
  
  for(i = 1; i < *argc; i++){
    if (!strcmp(argv[i], flag)) {
      updateArgs(argc, argv, i);
      return(1);
    }
  }
  return(0);
}

char *pc_stringarg(int *argc, char **argv, char *flag, char *value) {
  
  int i;
  
  for(i = 1; i < *argc -1; i++){
    if (!strcmp(argv[i], flag)) {
      value = argv[i+1];
      updateArgs(argc, argv, i+1);
      updateArgs(argc, argv, i);
      return(value);
    }
  }
  return(value);
}

int pc_intarg(int *argc, char **argv, char *flag, int value) {

  int i;

  for(i = 1; i < *argc - 1; i++){
    if (!strcmp(argv[i], flag)) {
      value = atoi(argv[i+1]);
      updateArgs(argc, argv, i+1);
      updateArgs(argc, argv, i);
      return(value);
    }
  }
  return(value);
}

double pc_doublearg(int *argc, char **argv, char *flag, double value) {

  int i;

  for(i = 1; i < *argc -1; i++){
    if (!strcmp(argv[i], flag)) {
      value = atof(argv[i+1]);
      updateArgs(argc, argv, i+1);
      updateArgs(argc, argv, i);
      return(value);
    }
  }
  return(value);
}
  
short *pc_shortarrayarg(int *argc, char **argv, char *flag, int elements, int size) {
 
  short *array;
  int i,j;

  array = NULL;

  if (size < elements) {
    quit(-1,"pc_shortarrayarg Error : Size of array is less than number of elements\nto be read.\n");
  }

  for(i = 1; i < *argc-elements; i++){
    if (!strcmp(argv[i], flag)) {

      array = (short *) rr_malloc(size * sizeof(int));

      for (j=0;j<=elements-1;j++) {
	array[j] = atoi(argv[i+1+j]);
      }
      for (j=i+elements;j>=i;j--) {
	updateArgs(argc, argv, j);
      }
      return(array);
    }
  }

  return(array);
}

int *pc_intarrayarg(int *argc, char **argv, char *flag, int elements, int size) {
 
  int *array;
  int i,j;

  array = NULL;

  if (size < elements) {
    quit(-1,"pc_shortarrayarg Error : Size of array is less than number of elements\nto be read.\n");
  }

  for(i = 1; i < *argc-elements; i++){
    if (!strcmp(argv[i], flag)) {

      array = (int *) rr_malloc(size * sizeof(int));

      for (j=0;j<=elements-1;j++) {
	array[j] = atoi(argv[i+1+j]);
      }
      for (j=i+elements;j>=i;j--) {
	updateArgs(argc, argv, j);
      }
      return(array);
    }
  }

  return(array);
}

void pc_report_unk_args(int *argc, char **argv, int verbosity) {
 
  int i;
 
  if (*argc > 1) {
    fprintf(stderr,"Error : Unknown (or unprocessed) command line options:\n");
    for (i = 1; i< *argc; i++) {
      fprintf(stderr,"%s ",argv[i]);
    }
    quit(-1,"\nRerun with the -help option for more information.\n");
  }
}

void report_version(int *argc, char **argv) {

  if (pc_flagarg(argc,argv,"-version")) {
    quit(-1,"%s from the CMU-Cambridge SLM Toolkit V%3.2f\n",argv[0],VERSION);
    
  }
}

