
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


/* Patched together from other header files. Ugly (and not used in
   compiling the toolkit), but makes using toolkit functions in other
   code simpler. */

#include <stdio.h>
#include <stdlib.h>

#ifndef _TOOLKIT_H_
#define _TOOLKIT_H_

#define DEFAULT_N 3
#define DEFAULT_VERBOSITY 2
#define MAX_VOCAB_SIZE 65535

/* The following gives the amount of memory (in MB) which the toolkit
   will assign when allocating big chunks of memory for buffers. Note
   that the more memory that can be allocated, the faster things will
   run, so if you are running these tools on machines with 400 MB of
   RAM, then you could safely triple this figure. */

#define STD_MEM 100
#define DEFAULT_TEMP "/usr/tmp/"

#define VERSION 2.0

typedef unsigned short flag;

#endif
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



/* Function prototypes for pc library */

#ifndef _PCGEN_H_
#define _PCGEN_H_

int pc_flagarg(int *argc, char **argv, char *flag);

char *pc_stringarg(int *argc, char **argv, char *flag, char *value);

int pc_intarg(int *argc, char **argv, char *flag, int value);

double pc_doublearg(int *argc, char **argv, char *flag, double value);

short *pc_shortarrayarg(int *argc, char **argv, char *flag, int elements, 
			int size);

int *pc_intarrayarg(int *argc, char **argv, char *flag, int elements, 
		    int size);

void pc_message(unsigned short verbosity, 
	       unsigned short priority, 
	       char *msg, ...);

void pc_report_unk_args(int *argc, char **argv, int verbosity);

void report_version(int *argc, char **argv);

#endif


/* GENERAL.H  */
/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1994, Carnegie Mellon University and Ronald Rosenfeld.
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

#ifndef _GENERAL_H_
#define _GENERAL_H_


#define CMU_SLM_VERSION  "CMU SLM Toolkit, Version for internal CMU use"

/* the following should be made machine-dependent */
typedef int   int32;
typedef short int16;

FILE *rr_fopen(char *filename, char *mode);
void *rr_fseek(FILE *fp, int offset, int mode, char *description);
void *rr_fread(char *ptr, int elsize, int n_elem, FILE *fp,
	       char *header, int not_more);
void *rr_fwrite(char *ptr, int elsize, int n_elem, FILE *fp, char *header);
char *rr_malloc(size_t n_bytes);
char *rr_calloc(size_t nelem, size_t elsize);
int  rr_filesize(int fd);
int  rr_feof(FILE *fp);
char *salloc(char *str);
int  rr_fexists(char *path);
FILE *rr_iopen(char *path);
void *rr_iclose(FILE *fp);
FILE *rr_oopen(char *path);
void *rr_oclose(FILE *fp);
void parse_line(char *line, int mwords, int canonize,
    char **pword_begin, char **pword_end, int *p_nwords, int *p_overflow);
int quit(int rc, char *msg, ...);

typedef char Boolean;
typedef unsigned short wordid_t;
typedef int    cluster_t;

#ifndef MIN
#define MIN(X,Y)  ( ((X)<(Y)) ? (X) : (Y))
#endif
#ifndef MAX
#define MAX(X,Y)  ( ((X)>(Y)) ? (X) : (Y))
#endif

#define LOG_BASE	9.9995e-5
#define MIN_LOG		-690810000
#define LOG(x) ((x == 0.0) ? MIN_LOG : ((x > 1.0) ?			    \
				 	(int) ((log (x) / LOG_BASE) + 0.5) :\
				 	(int) ((log (x) / LOG_BASE) - 0.5)))
#define EXP(x)  (exp ((double) (x) * LOG_BASE))

#ifdef __alpha
#define SLM_SWAP_BYTES  1    /* reverse byteorder */
#endif

/* the following are for the benefit of vararg-less environments */

#define quit0(rc,msg) {fprintf(stderr,msg); exit(rc);}
#define quit1(rc,msg,i1) {fprintf(stderr,msg,i1); exit(rc);}
#define quit2(rc,msg,i1,i2) {fprintf(stderr,msg,i1,i2); exit(rc);}
#define quit3(rc,msg,i1,i2,i3) {fprintf(stderr,msg,i1,i2,i3); exit(rc);}
#define quit4(rc,msg,i1,i2,i3,i4) {fprintf(stderr,msg,i1,i2,i3,i4); exit(rc);}

#define  MAX_WORDS_PER_DOC 65534

#endif  
/* MIPS_SWAP.H  */
/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1994, Carnegie Mellon University and Ronald Rosenfeld.
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

#ifndef _MIPS_SWAP_H_
#define _MIPS_SWAP_H_


#ifdef SLM_SWAP_BYTES    /* reverse byteorder */

/* the following works even for badly aligned pointers */

#define SWAPFIELD(x) {if     (sizeof(*(x))==sizeof(short)) {SWAPHALF((x))}  \
		      else if (sizeof(*(x))==sizeof(int))   {SWAPWORD((x))}  \
		      else if (sizeof(*(x))==sizeof(double)){SWAPDOUBLE((x))}\
		     }

#define SWAPHALF(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+1); \
			*((char*)(x)+1) = tmp_byte;        \
		    }
#define SWAPWORD(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+3); \
			*((char*)(x)+3) = tmp_byte;  	   \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+2); \
			*((char*)(x)+2) = tmp_byte;        \
		    }

#define SWAPDOUBLE(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+7); \
			*((char*)(x)+7) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+6); \
			*((char*)(x)+6) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+2); \
			*((char*)(x)+2) = *((char*)(x)+5); \
			*((char*)(x)+5) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+3); \
			*((char*)(x)+3) = *((char*)(x)+4); \
			*((char*)(x)+4) = tmp_byte;        \
		    }

#if 0 /* old */
#define SWAPHALF(x) *(short*)(x) = ((0xff   & (*(short*)(x)) >> 8) | \
				    (0xff00 & (*(short*)(x)) << 8))
#define SWAPWORD(x) *(int*)  (x) = ((0xff       & (*(int*)(x)) >> 24) | \
				    (0xff00     & (*(int*)(x)) >>  8) | \
                        	    (0xff0000   & (*(int*)(x)) <<  8) | \
				    (0xff000000 & (*(int*)(x)) << 24))
#define SWAPDOUBLE(x) { int *low  = (int *) (x), \
		            *high = (int *) (x) + 1, temp;\
                        SWAPWORD(low);  SWAPWORD(high);\
                        temp = *low; *low = *high; *high = temp;}
#endif /* old */
#else

#define SWAPFIELD(x)
#define SWAPHALF(x)
#define SWAPWORD(x)
#define SWAPDOUBLE(x)

#endif


#define ALWAYS_SWAPFIELD(x) {\
		      if      (sizeof(*(x))==sizeof(short)) {SWAPHALF((x))}  \
		      else if (sizeof(*(x))==sizeof(int))   {SWAPWORD((x))}  \
		      else if (sizeof(*(x))==sizeof(double)){SWAPDOUBLE((x))}\
		     }

#define ALWAYS_SWAPHALF(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+1); \
			*((char*)(x)+1) = tmp_byte;        \
		    }
#define ALWAYS_SWAPWORD(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+3); \
			*((char*)(x)+3) = tmp_byte;  	   \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+2); \
			*((char*)(x)+2) = tmp_byte;        \
		    }

#define ALWAYS_SWAPDOUBLE(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+7); \
			*((char*)(x)+7) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+6); \
			*((char*)(x)+6) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+2); \
			*((char*)(x)+2) = *((char*)(x)+5); \
			*((char*)(x)+5) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+3); \
			*((char*)(x)+3) = *((char*)(x)+4); \
			*((char*)(x)+4) = tmp_byte;        \
		    }

#endif

/* SIH.H : String-to-Integer Hashing */
/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1994, Carnegie Mellon University and Ronald Rosenfeld.
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

#ifndef _SIH_H_
#define _SIH_H_


typedef struct {
	char *string;	   /* string (input to hash function) */
	int32 intval;	   /* Associated int32 value (output of hash function) */
} sih_slot_t;


typedef struct {
	double  max_occupancy;  /* max. allowed occupancy rate */
	double  growth_ratio;   /* ratio of expansion when above is violated */
	int     warn_on_update; /* print warning if same string is hashed again */
	int	nslots;    	/* # of slots in the hash table */
	int	nentries;	/* # of actual entries */
	sih_slot_t *slots;	/* array of (string,intval) pairs */
} sih_t;


sih_t *sih_create(int initial_size, double max_occupancy, double growth_ratio, int warn_on_update);

void    sih_add(sih_t *ht, char *string, int32 intval);

char sih_lookup(sih_t *ht, char *string, int32 *p_intval);

void *sih_val_write_to_file(sih_t *ht, FILE *fp, char *filename, int verbosity);

void *sih_val_read_from_file(sih_t *ht, FILE *fp, char *filename, int verbosity);

/* Moved to here from read_voc.c by Philip Clarkson March 4, 1997 */

void get_vocab_from_vocab_ht(sih_t *ht, int vocab_size, int verbosity, char ***p_vocab);

/* Added by Philip Clarkson March 4, 1997 */

void read_wlist_into_siht(char *wlist_filename, int verbosity,  
			  sih_t *p_word_id_ht, int *p_n_wlist);

void read_wlist_into_array(char *wlist_filename, int verbosity, 
			   char ***p_wlist, int *p_n_wlist);

#endif 
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

/* Function prototypes for evallm */

#ifndef _EVALLM_PROTS_
#define _EVALLM_PROTS_


/* Type specification for forced back-off list */

typedef struct {
  flag backed_off;
  flag inclusive;
} fb_info;

typedef float bo_t;
typedef float prob_t;

/* Type specification for arpa_lm type */

typedef struct {

  unsigned short n;                /* n=3 for trigram, n=4 for 4-gram etc. */

  /* Vocabulary stuff */

  sih_t          *vocab_ht;      /* Vocabulary hash table */
  unsigned short vocab_size;     /* Vocabulary size */
  char           **vocab;        /* Array of vocabulary words */
  flag           *context_cue;   /* True if word with this id is
				    a context cue */
  int            no_of_ccs;      /* The number of context cues in the LM */

  /* Tree */

  table_size_t   *table_sizes;   /* Pointer to table size array */
  id__t          **word_id;      /* Pointer to array of id lists */

  bo_t           **bo_weight;    /* Pointer to array of back-off weights */
  prob_t         **probs;        /* Pointer to array of probabilities */
  index__t       **ind;          /* Pointer to array of index lists */

  /* Index lookup tables */

  ptr_tab_t      **ptr_table;     /* Pointer to the tables used for compact 
				     representation of the indices */
  unsigned short *ptr_table_size; /* Pointer to array of pointer tables */

  /* Misc */

  int            *num_kgrams;     /* Array indicating how many 
				     2-grams, ... ,n-grams, have been 
				     processed so far */

  unsigned short vocab_type;      /* see #define stuff at the top */

  unsigned short first_id;        /* 0 if we have open vocab, 1 if we have
				     a closed vocab. */

} arpa_lm_t;

/* Function prototypes */

unsigned short num_of_types(int k,
			    int ind,
			    ng_t *ng);
void decode_bo_case(int bo_case,
		    int context_length,
		    FILE *annotation_fp);

void display_stats(ng_t *ng);

void display_arpa_stats(arpa_lm_t *arpa_ng);

void load_lm(ng_t *ng,
	     char *lm_filename);

void load_arpa_lm(arpa_lm_t *arpa_lm,
		  char *lm_filename);
		  

void parse_comline(char *input_line,
		  int *num_of_args,
		  char **args);

void compute_perplexity(ng_t *ng,
			arpa_lm_t *arpa_ng,
			char *text_stream_filename,
			char *probs_stream_filename,
			char *annotation_filename,
			char *oov_filename,
			char *fb_list_filename,
			flag backoff_from_unk_inc,
			flag backoff_from_unk_exc,
			flag backoff_from_ccs_inc,
			flag backoff_from_ccs_exc,
			flag arpa_lm,
			flag include_unks,
			double log_base);

fb_info *gen_fb_list(sih_t *vocab_ht,
		     int vocab_size,
		     char **vocab,
		     flag *context_cue,
		     flag backoff_from_unk_inc,
		     flag backoff_from_unk_exc,
		     flag backoff_from_ccs_inc,
		     flag backoff_from_ccs_exc,
		     char *fb_list_filename);

void validate(ng_t *ng,
	      arpa_lm_t *arpa_ng,
	      char **words,
	      flag backoff_from_unk_inc,
	      flag backoff_from_unk_exc,
	      flag backoff_from_ccs_inc,
	      flag backoff_from_ccs_exc,
	      flag arpa_lm,
	      char *fb_list_filename);

double calc_prob_of(id__t sought_word,
		    id__t *context,
		    int context_length,
		    ng_t *ng,
		    arpa_lm_t *arpa_ng,
		    fb_info *fb_list,
		    int *bo_case,
		    int *actual_context_length,
		    flag arpa_lm);

void arpa_bo_ng_prob(int context_length,
		     id__t *sought_ngram,
		     arpa_lm_t *arpa_ng,
		     int verbosity,
		     double *p_prob,
		     int *bo_case);


#endif
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


/* Function prototypes */

#ifndef _IDNGRAM2LM_H_
#define _IDNGRAM2LM_H_

unsigned short num_of_types(int k,
			    int ind,
			    ng_t *ng);
int get_ngram(FILE *id_ngram_fp,ngram *ng,flag is_ascii);
void calc_mem_req(ng_t *ng,flag is_ascii);
void write_arpa_lm(ng_t *ng,int verbosity);
void write_bin_lm(ng_t *ng,int verbosity);
unsigned short new_index(int full_index,
			 int *ind_table,
			 unsigned short *ind_table_size,
			 int position_in_list);
int get_full_index(unsigned short short_index,
		   int *ind_table,
		   int ind_table_size,
		   int position_in_list);
void compute_gt_discount(int            n,
			 int            *freq_of_freq,
			 int            fof_size,
			 unsigned short *disc_range,
			 int cutoff,
			 int verbosity,
			 disc_val_t **discounted_values);
int lookup_index_of(int *lookup_table, 
		    int lookup_table_size, 
		    int intintval);
void compute_unigram(ng_t *ng,int verbosity);
void compute_back_off(ng_t *ng,int n,int verbosity);
void bo_ng_prob(int context_length,
		id__t *sought_ngram,
		ng_t *ng,
		int verbosity,
		double *p_prob,
		int *bo_case);
void increment_context(ng_t *ng, int k, int verbosity);
unsigned short short_alpha(double long_alpha,
			   double *alpha_array,
			   unsigned short *size_of_alpha_array,
			   int elements_in_range,
			   double min_range,
			   double max_range);

double double_alpha(unsigned short short_alpha,
		    double *alpha_array,
		    int size_of_alpha_array,
		    int elements_in_range,
		    double min_range,
		    double max_range);

void guess_mem(int total_mem,
	       int middle_size,
	       int end_size,
	       int n,
	       table_size_t *table_sizes,
	       int verbosity);

void read_voc(char *filename, int verbosity,   
	      sih_t *p_vocab_ht, char ***p_vocab, 
	      unsigned short *p_vocab_size);

void store_count(flag four_byte_counts,
		 int *count_table,
		 int count_table_size,
		 unsigned short *short_counts,
		 int *long_counts,
		 int position,
		 int count);

int return_count(flag four_byte_counts,
		 int *count_table,
		 unsigned short *short_counts,
		 int *long_counts,
		 int position);

#endif
