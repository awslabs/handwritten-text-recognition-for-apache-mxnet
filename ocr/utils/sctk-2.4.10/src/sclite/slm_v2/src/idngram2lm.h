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

#include "ngram.h"
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
