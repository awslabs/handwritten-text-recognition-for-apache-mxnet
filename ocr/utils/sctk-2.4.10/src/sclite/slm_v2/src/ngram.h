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


/* Type and function definitions for general n_gram models */

#ifndef _NGRAM_H_
#define _NGRAM_H_

#include "rr_libs/sih.h"
#include "toolkit.h"

#define DEFAULT_COUNT_TABLE_SIZE 65535
#define DEFAULT_OOV_FRACTION 0.5
#define DEFAULT_DISC_RANGE_1 1
#define DEFAULT_DISC_RANGE_REST 7
#define DEFAULT_MIN_ALPHA -3.2
#define DEFAULT_MAX_ALPHA 2.5
#define DEFAULT_OUT_OF_RANGE_ALPHAS 10000

#define GOOD_TURING 1
#define ABSOLUTE 2
#define LINEAR 3
#define WITTEN_BELL 4

#define SPECIFIED 1
#define BUFFER 2
#define TWO_PASSES 3

#define KEY 65000

#define CLOSED_VOCAB 0
#define OPEN_VOCAB_1 1
#define OPEN_VOCAB_2 2

typedef unsigned short id__t; /* Double underscore, since id_t is
				 already defined on some platforms */
typedef int count_t;   /* The count as read in, rather than its index 
			  in the count table. */
typedef unsigned short count_ind_t; /* The count's index in the count 
				       table. */
typedef unsigned short bo_weight_t;
typedef unsigned short cutoff_t;
typedef int table_size_t;
typedef unsigned short index__t;
typedef double disc_val_t;
typedef double uni_probs_t;
typedef int ptr_tab_t;
typedef float four_byte_t;


typedef struct {
  unsigned short n;
  id__t          *id_array;
  count_t        count;
} ngram;

typedef struct {
  unsigned short count_table_size;
  int            *counts_array;
} count_table_t;

typedef struct {

  /* Language model type */

  unsigned short n;                /* n=3 for trigram, n=4 for 4-gram etc. */
  int            version;

  /* Vocabulary stuff */

  sih_t          *vocab_ht;      /* Vocabulary hash table */
  unsigned short vocab_size;     /* Vocabulary size */
  char           **vocab;        /* Array of vocabulary words */
  unsigned short no_of_ccs;      /* Number of context cues */

  /* Tree */

  table_size_t   *table_sizes;   /* Pointer to table size array */
  id__t          **word_id;      /* Pointer to array of id lists */
  count_ind_t    **count;        /* Pointer to array of count lists 
				    (actually indices in a count table) */
  count_ind_t    *marg_counts;   /* Array of marginal counts for the 
				    unigrams. The normal unigram counts
				    differ in that context cues have
				    zero counts there, but not here */
  int            **count4;       /* Alternative method of storing the counts,
				    using 4 bytes. Not normally allocated */
  int            *marg_counts4;  /* Ditto */
  bo_weight_t    **bo_weight;    /* Pointer to array of back-off weights */
  four_byte_t    **bo_weight4;   /* Pointer to array of 4 byte
				    back_off weights. Only one of
				    these arrays will be allocated */
  index__t       **ind;          /* Pointer to array of index lists */
  

  /* Two-byte alpha stuff */

  double         min_alpha;      /* The minimum alpha in the table */
  double         max_alpha;      /* The maximum alpha in the table */
  unsigned short out_of_range_alphas;  /* The maximum number of out of range 
					  alphas that we are going to allow. */
  double         *alpha_array;
  unsigned short size_of_alpha_array;

  /* Count table */

  count_ind_t    count_table_size; /* Have same size for each count table */
  count_t        **count_table;    /* Pointer to array of count tables */

  /* Index lookup tables */

  ptr_tab_t      **ptr_table;     /* Pointer to the tables used for compact 
				     representation of the indices */
  unsigned short *ptr_table_size; /* Pointer to array of pointer tables */

  /* Discounting and cutoffs - note: some of these may not used,
     depending on the discounting techinque used. */

  unsigned short discounting_method;     /* See #define stuff at the top of 
					    this file */
  cutoff_t       *cutoffs;               /* Array of cutoffs */
  int            **freq_of_freq;         /* Array of frequency of frequency 
					    information  */
  unsigned short *fof_size;              /* The sizes of the above arrays */
  unsigned short *disc_range;            /* Pointer to array of discounting 
					    ranges - typically will be 
					    fof_size - 1, but can be reduced
					    further if stats are anomolous */
  disc_val_t     **gt_disc_ratio;        /* The discounted values of the 
					    counts */
  disc_val_t     *lin_disc_ratio;        /* The linear discounting ratio */
  double         *abs_disc_const;        /* The constant required for
					    absolute discounting */

  /* Unigram statistics */

  uni_probs_t    *uni_probs;             /* Probs for each unigram */
  uni_probs_t    *uni_log_probs;         /* Log probs for each unigram */
  flag           *context_cue;           /* True if word with this id is
					    a context cue */
  int            n_unigrams;             /* Total number of unigrams in
					    the training data */
  int            min_unicount;           /* Count to which infrequent unigrams
					    will be bumped up */
  /* Input files */

  char           *id_gram_filename;  /* The filename of the id-gram file */
  FILE           *id_gram_fp;        /* The file pointer of the id-gram file */
  char           *vocab_filename;    /* The filename of the vocabulary file */
  char           *context_cues_filename; /* The filename of the context cues 
					    file */
  FILE           *context_cues_fp;       /* The file pointer of the context 
					    cues file */

  /* Output files */

  flag           write_arpa;      /* True if the language model is to be 
				     written out in arpa format */
  char           *arpa_filename;  /* The filaname of the arpa format LM */
  FILE           *arpa_fp;        /* The file of the arpa format LM */
  flag           write_bin;       /* True if the language model is to be 
				     written out in binary format */
  char           *bin_filename;   /* The filaname of the bin format LM */
  FILE           *bin_fp;         /* The file of the bin format LM */

  /* Misc */

  int            *num_kgrams;     /* Array indicating how many 
				     2-grams, ... ,n-grams, have been 
				     processed so far */

  unsigned short vocab_type;      /* see #define stuff at the top */

  unsigned short first_id;        /* 0 if we have open vocab, 1 if we have
				     a closed vocab. */

  /* Once the tree has been constructed, the tables are indexed from 0
     to (num_kgrams[i]-1). */

  /* 1-gram tables are indexed from 0 to ng.vocab_size. */

  double         zeroton_fraction; /* cap on prob(zeroton) as fraction of 
				      P(singleton) */
  double         oov_fraction;
  flag           four_byte_alphas;
  flag           four_byte_counts;

} ng_t;

#endif


