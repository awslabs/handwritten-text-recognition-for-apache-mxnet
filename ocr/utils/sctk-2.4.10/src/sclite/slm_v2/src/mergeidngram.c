
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

#include <stdio.h>
#include <stdlib.h>
#include "pc_libs/pc_general.h"
#include "toolkit.h"
#include "rr_libs/general.h"

typedef unsigned short id__t;

typedef struct {
  unsigned short n;
  id__t          *id_array;
  int            count;
} ngram;

int n;
flag ascii_in;
flag ascii_out;

void updateArgs( int *pargc, char **argv, int rm_cnt );
void procComLine( int *argc, char **argv );
void printUsage( char *name );
int cmp_ngram( ngram *ng1, ngram *ng2 );
extern int get_ngram(FILE *id_ngram_fp, ngram *ng, flag ascii);

/* write ngram in either ascii or binary */
void write_ngram( FILE *id_ngram_fp, ngram *ng, flag ascii )
{
  int i;
  
  if ( ascii ) {
    for( i = 0; i < n; i++ ) {
      if ( fprintf( stdout, "%hu ", ng->id_array[i] ) < 0 ) {
	quit( -1, "error writing ascii ngram\n" );
      }
    }
    if ( fprintf( stdout, "%d\n", ng->count ) < 0 ) {
      quit( -1, "error writing ascii ngram\n" );
    }
  }
  else {

    for ( i = 0; i < n; i++ ) {
      rr_fwrite( &ng->id_array[i], sizeof( id__t ), 1, id_ngram_fp,
		 "binary ngram" );
    }
    rr_fwrite( &ng->count, sizeof( int ), 1, id_ngram_fp,
	       "binary ngram" );
  }
}

/* update command line argument sequence */
void updateArgs( int *pargc, char **argv, int rm_cnt )
{
  int i ;             

  /* update the argument count */
  (*pargc)-- ;

  /* update the command line */
  for( i = rm_cnt ; i < *pargc ; i++ ) argv[i] = argv[i+1] ;
}       

/* process the command line */
void procComLine( int *argc, char **argv ) 
{
  int i;

  n = 3;
  ascii_in = 0;
  ascii_out = 0;

  i = *argc - 1 ;
  while( i > 0 ) {

    /* handle a request for help */
    if( !strcmp( argv[i], "-h" ) || !strcmp( argv[i], "-help" ) ) {
      printUsage( argv[0] ) ;
      exit( 1 ) ;
    }

    /* specify n */
    if( !strcmp( argv[i], "-n" ) ) {
      n = atoi( argv[i+1] ) ;
      updateArgs( argc, argv, i+1 ) ;
      updateArgs( argc, argv, i ) ;
    }
    
    /* input files in ascii */
    if( !strcmp( argv[i], "-ascii_input" ) ) {
      ascii_in = 1;
      updateArgs( argc, argv, i ) ;
    }

    /* input files in ascii */
    if( !strcmp( argv[i], "-ascii_output" ) ) {
      ascii_out = 1;
      updateArgs( argc, argv, i ) ;
    }

    i--;
  }
}
   
/* show command line usage */ 
void printUsage( char *name )
{
  fprintf( stderr, "%s: merge idngram files.\n", name );
  fprintf( stderr, "Usage:\n%s [options] .idngram_1 ... .idngram_N > .idngram\n", name );
  fprintf( stderr, "  -n 3           \tn in n-gram \n" );
  fprintf( stderr, "  -ascii_input   \tinput files are ascii\n" );
  fprintf( stderr, "  -ascii_output  \toutput files are ascii\n" );
  exit(1);
}

/* compare two ngrams */
int cmp_ngram( ngram *ng1, ngram *ng2 )
{
  int i;

  if ( ng1->n != ng2->n ) {
    quit( -1, "Error: n-grams have different n!\n" );
  }

  for( i = 0; i < ng1->n; i++ ) {
    if ( ng1->id_array[i] < ng2->id_array[i] ) return( -1 );
    if ( ng1->id_array[i] > ng2->id_array[i] ) return( 1 );
  }
  return( 0 );
}
    
int main( int argc, char **argv )
{
  FILE **fin;
  ngram *ng;
  ngram outng;
  flag *done, finished;
  int i, j, nfiles;

  /* Process the command line */
  report_version(&argc,argv);
  procComLine( &argc, argv ) ;
  if( argc < 2 ) {
    printUsage( argv[0] ) ;
    exit( 1 ) ;
  }
  nfiles = argc - 1;

  /* allocate memory */
  fin = (FILE **) rr_malloc( sizeof( FILE *) * nfiles );
  done = (flag *) rr_malloc( sizeof( flag ) * nfiles );
  ng = (ngram *) rr_malloc( sizeof( ngram ) * nfiles );
  for( i = 0; i < nfiles; i++ ) {
    ng[i].id_array = (id__t *) rr_calloc( n, sizeof( id__t ) );
    ng[i].n = n;
  }
  outng.id_array = (id__t *) rr_calloc( n, sizeof( id__t ) );
  outng.n = n;

  /* open the input files */
  for( i = 0; i < nfiles; i++ ) {
    fin[i] = rr_iopen( argv[i+1] );
  }

  /* read first ngram from each file */
  for( i = 0; i < nfiles; i++ ) {
    done[i] = 0;
    if ( !get_ngram( fin[i], &ng[i], ascii_in ) ) {
      done[i] = 1;
    }
  }

  finished = 0;
  while ( !finished ) {

  /* set outng to max possible */
  for( i = 0; i < n; i++ )
    outng.id_array[i] = MAX_VOCAB_SIZE;
    
    /* find smallest ngram */
    for( i = 0; i < nfiles; i++ ) {
      if ( !done[i] ) {
	if ( cmp_ngram( &outng, &ng[i] ) > 0 ) {
	  for( j = 0; j < n; j++ ) outng.id_array[j] = ng[i].id_array[j];
	}
      }
    }
    
    outng.count = 0;
    for( i = 0; i < nfiles; i++ ) {
      if ( !done[i] ) {
	/* add counts of equal ngrams */
	if ( cmp_ngram( &outng, &ng[i] ) == 0 ) {
	  outng.count += ng[i].count;
	  if ( !get_ngram( fin[i], &ng[i], ascii_in ) ) {
	    /* check if all files done */
	    done[i] = 1;
	    finished = 1;
	    for( j = 0; j < nfiles; j++ ) {
	      if ( ! done[j] ) finished = 0;
	    }
	  }
	}
      }
    }

    write_ngram( stdout, &outng, ascii_out );

  }
  for( i = 0; i < nfiles; i++ )
    rr_iclose( fin[i] );

  fprintf(stderr,"mergeidngram : Done.\n");

  return( 0 );
}

