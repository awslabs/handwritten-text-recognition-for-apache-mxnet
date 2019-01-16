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

#include <sys/types.h>
#include <stdio.h>
#include "general.h"

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
