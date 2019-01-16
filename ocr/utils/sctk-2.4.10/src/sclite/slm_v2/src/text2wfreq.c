
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

/* Very strongly based on the program wordfreq, by Gary Cook
   (gdc@eng.cam.ac.uk), adapted (with permission) for the sake of
   consistency with the rest of the toolkit by Philip Clarkson,
   27/9/96 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "toolkit.h"
#include "rr_libs/general.h"
#include "pc_libs/pc_general.h"

#define MAX_STRING_LENGTH 501
#define DEFAULT_HASH 1000000

struct node {

  char *word;
  int count;

  struct node *next;

};

struct hash_table {

  int size;
  struct node **chain;

};

/* create a new node, and sets the count to 1 */
struct node *new_node( char *key )
{
  struct node *x;

  x = (struct node *) rr_malloc( sizeof( struct node ) );
  x->word = (char *) rr_malloc( (strlen( key ) + 1) * sizeof( char ) );
  strcpy( x->word, key );
  x->count = 1;
  return x;
}

/* create hash table */
void new_hashtable( struct hash_table *table, int M )
{
  int i;
  
  table->size = M;
  table->chain = (struct node **) rr_malloc( M * sizeof( struct node *) );
  for( i = 0; i < M; i++ ) {
    table->chain[i] = new_node( "HEAD_NODE" );
    table->chain[i]->next = (struct node *) NULL;
  }
}

/* update linked list */
int update_chain( struct node *t, char *key )
{
  struct node *x;
  int score;

  while( t->next != NULL ) {
    score = strcmp( key, t->next->word ); 
    /* move to next node */
    if ( score > 0 ) t = t->next;
    /* update node */
    else if ( score == 0 ) {
      t->next->count++;
      return 1;
    }
    /* add new node */
    else {
      x = new_node( key );
      x->next = t->next;
      t->next = x;
      return 0;
    }
  }
  /* add node at end */
  x = new_node( key );
  x->next = (struct node *) NULL;
  t->next = x;
  return 0;
}

/* print contents of linked list */
void print_chain( struct node *t )
{
  t = t->next;  /* don't print head node */
  while ( t != NULL ) {
    printf( "%s %d\n", t->word, t->count );
    t = t->next;
  }
}

/* generate a hash table address from a variable length character */
/* string - from R. Sedgewick, "Algorithms in C++". */
int hash( char *key, int M )
{
  unsigned int h; 
  char *t = key;

  for( h = 0; *t; t++ )
    h = ( 64 * h + *t ) % M;
  return h;
}

/* print hash table contents */
void print( struct hash_table *table )
{
  int i;
  for( i = 0; i < table->size; i++ )
    print_chain( table->chain[i] );
}

/* update hash table contents */
void update( struct hash_table *table, char *key, int verbosity )
{
  int chain;
  
  chain = hash( key, table->size );
  if ( chain < 0 || chain >= table->size ) {
    pc_message(verbosity,1,"WARNING : invalid hash address.\n");
    pc_message(verbosity,1,"%s ignored\n", key );
    return;
  }
  update_chain( table->chain[ chain ], key );
}

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

int main( int argc, char **argv )
{
  int init_nwords, hash_size, scanrc;
  struct hash_table vocab;
  char word[MAX_STRING_LENGTH];
  int verbosity;

  if (pc_flagarg( &argc, argv,"-help")) {
    fprintf(stderr,"text2wfreq : Generate a word frequency list for text.\n");
    fprintf(stderr,"Usage : text2freq [ -hash %d ]\n",DEFAULT_HASH);
    fprintf(stderr,"                  [ -verbosity 2 ]\n");
    fprintf(stderr,"                  < .text > .wfreq\n");
    exit(1);
  }

  /* process command line */

  report_version(&argc,argv);
  init_nwords = pc_intarg( &argc, argv, "-hash", DEFAULT_HASH );

  verbosity = pc_intarg(&argc,argv,"-verbosity",DEFAULT_VERBOSITY);

  pc_report_unk_args(&argc,argv,verbosity);

  pc_message(verbosity,2,"text2wfreq : Reading text from standard input...\n");

  hash_size = nearest_prime( init_nwords );
  new_hashtable( &vocab, hash_size );
  while( (scanrc = scanf( "%500s", word )) == 1 ) {
    if ( strlen( word ) >= 500 ) {
      pc_message(verbosity,1,"text2wfreq : WARNING: word too long, will be split: %s...\n",word);
    }
    if (strlen(word)) {
      update( &vocab, word ,verbosity);
    }
  }
  if ( scanrc != EOF ) {
    quit(-1,"Error reading input\n");
  }
  print( &vocab );
  pc_message(verbosity,0,"text2wfreq : Done.\n");
  return 0;
}


