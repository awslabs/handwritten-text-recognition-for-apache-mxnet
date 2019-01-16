/* sih:  string-to-int32 hashing */
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
/* The string: any non-null character string.
   The int32 value: any int32 value.
   Only addition, update and lookup are currently allowed.
   If the hash table was not meant for update, a warning will be printed during update.
   The hash table grows automatically to preserve the 'max_occupancy' condition.
*/
/* 94/08/18, RR: sih_lookup(): if not found, also put 0 in 'intval'  */

/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */

#include <stdio.h>
#include <sys/types.h>
#include "general.h"
#include "sih.h"

/* need to include stdlib to prevent warnings at compilation time,
   Philip Clarkson, March 1997 */

#include <stdlib.h>

#define HASH_VERSION 940728 /* change this if you change the hash function below */
#define SIH_KEY(STRING,KEY) {\
    int i = 0; \
    char *pstr = STRING; \
    KEY = 0; \
    do {KEY += *pstr++ << (0xF & --i);} while (*pstr);  /* Fil's hash function */ \
}

int nearest_prime_up(int num)
/* return the nearest prime not smaller than 'num' */
{
  int num_has_divisor=1;
  if (num/2*2 == num) num++; /* start w/ an odd number */
  for (; num_has_divisor; num+=2) {
     int div;
     num_has_divisor=0;
     for (div=3; div<=num/3; div++) {
        if ((num/div) * div == num) {
	   num_has_divisor=1;
	   break;
	}
     }
  }
  num -= 2;
  return(num);
}


sih_t *sih_create(int initial_size, double max_occupancy, double growth_ratio, int warn_on_update)
{
  static char rname[]="sih_create";
  sih_t *ht = (sih_t *) rr_malloc(sizeof(sih_t));

  initial_size = nearest_prime_up(MAX(initial_size,11));
  if (max_occupancy<0.01 || max_occupancy>0.99) 
    quit(-1,"%s ERROR: max_occupancy (%.3f) must be in the range 0.01-0.99\n",rname,max_occupancy);
  if (growth_ratio<1.10 || growth_ratio>100) 
    quit(-1,"%s ERROR: growth_ratio (%.3f) must be in the range 1.1-100\n",rname,growth_ratio);

  ht->max_occupancy = max_occupancy;
  ht->growth_ratio = growth_ratio;
  ht->warn_on_update = warn_on_update;
  ht->nslots = initial_size;
  ht->nentries = 0;
  ht->slots = (sih_slot_t *) rr_calloc(ht->nslots,sizeof(sih_slot_t));
  return(ht);
}



void sih_add(sih_t *ht, char *string, int32 intval)
{
  static char *rname = "sih_add";
  unsigned int  key;

  if (*string==0) quit(-1,"%s ERROR: cannot hash the null string\n",rname);

  /* if the "occupancy rate" exceeded its limit, expand the hash table */
  if (((ht->nentries+1) / (double)ht->nslots) > ht->max_occupancy) {
     sih_slot_t *old_slots = ht->slots, 
		*end_old_slots = (old_slots + ht->nslots),
		*pslot;

     /* allocate the new hash table */
     ht->nslots = (int)(ht->nslots * ht->growth_ratio) + 3;
     if ((ht->nentries / (double)ht->nslots) > ht->max_occupancy)
	ht->nslots = ht->nslots * (int)(ht->max_occupancy+1) + 3;
     ht->nslots = nearest_prime_up(ht->nslots);
     ht->nentries = 0;
     ht->slots = (sih_slot_t *) rr_calloc(ht->nslots,sizeof(sih_slot_t));

     /* copy all entries from old hash table to new one */
     for (pslot = old_slots; pslot<end_old_slots; pslot++) {
	if (pslot->string) sih_add(ht, pslot->string, pslot->intval);
     }
     free (old_slots);
  }

  SIH_KEY(string,key);

  for (; ; key++) {
     char *pstr;
     key %= ht->nslots;
     pstr = ht->slots[key].string;
     if (!pstr) {
	ht->slots[key].string = string;
	ht->slots[key].intval = intval;
	ht->nentries++;
	break;
     }
     else if (strcmp(pstr,string) == 0) {  /* updating an existing entry*/
	if (ht->warn_on_update) {
 	   fprintf(stderr,"%s WARNING: repeated hashing of '%s'",rname,string);
	   if (ht->slots[key].intval != intval) 
	     fprintf(stderr,", older value will be overridden.\n");
	   else fprintf(stderr,".\n");
	}
	ht->slots[key].intval = intval;
	break;
     }
  }
}


char sih_lookup(sih_t *ht, char *string, int32 *p_intval)
{
  static char *rname = "sih_lookup";
  unsigned int  key;

  if (*string == 0) quit(-1,"%s ERROR: cannot hash the null string\n",rname);

  SIH_KEY(string,key);

  for (; ; key++) {
     char *pstr;

     key %= ht->nslots;
     pstr = ht->slots[key].string;
     if (!pstr) {
	*p_intval = 0;
	return(0);
     }
     else if (strcmp(pstr, string) == 0) {
	*p_intval = (int) ht->slots[key].intval;
	return (1);
     }
  }
}



/* ======================================================================== */


/* Read/Write from/to a file an (almost) ready-to-use hash table.
   The hash-table is a string->int mapping.
   All intvals are written out, whether or not they belong to active entries.
   All strings are writen out too, where an empty entry is represented in the
   file as a null string (null strings as entries are not allowed).
*/
void *sih_val_write_to_file(sih_t *ht, FILE *fp, char *filename, int verbosity)
{

  static char rname[] = "sih_val_wrt_to_file";
  int nstrings=0, total_string_space=0, islot, version=HASH_VERSION;
  char null_char = '\0';

  /* write out the header */
  rr_fwrite(&version,sizeof(int),1,fp,"version");
  rr_fwrite(&ht->max_occupancy,sizeof(double),1,fp,"ht->max_occupancy");
  rr_fwrite(&ht->growth_ratio,sizeof(double),1,fp,"ht->growth_ratio");
  rr_fwrite(&ht->warn_on_update,sizeof(int),1,fp,"ht->warn_on_update");
  rr_fwrite(&ht->nslots,sizeof(int),1,fp,"ht->nslots");
  rr_fwrite(&ht->nentries,sizeof(int),1,fp,"ht->nentries");

  /* Write out the array of 'intval's.  */
  /* Also, compute and write out the total space taken by the strings */
  for (islot=0; islot<ht->nslots; islot++) {
    rr_fwrite(&(ht->slots[islot].intval),sizeof(int32),1,fp,"ht->slots[islot].intval");
    if (ht->slots[islot].string) {
       total_string_space += strlen(ht->slots[islot].string)+1;
       nstrings++;
    }
    else total_string_space++;
  }
  if (nstrings!=ht->nentries)
     quit(-1,"%s: nentries=%d, but there are actually %d non-empty entries\n",
	      rname, ht->nentries, nstrings);

  /* Write out the strings, with the trailing null, preceded by their length */
  rr_fwrite(&total_string_space,sizeof(int),1,fp,"total_string_space");
  for (islot=0; islot<ht->nslots; islot++) {
     if (ht->slots[islot].string)
       rr_fwrite(ht->slots[islot].string,sizeof(char),strlen(ht->slots[islot].string)+1,fp,"str");
     else 
       rr_fwrite(&null_char,sizeof(char),1,fp,"null");
  }
  if (verbosity) fprintf(stderr,
     "%s: a hash table of %d entries (%d non-empty) was written to '%s'\n",
      rname, ht->nslots, ht->nentries, filename);

  return(0); /* Not relevant, but stops compilation warnings. */

}


void *sih_val_read_from_file(sih_t *ht, FILE *fp, char *filename, int verbosity)
{
  static char rname[] = "sih_val_rd_fm_file";
  int total_string_space=0, islot, version;
  char *string_block, *pstring,  *past_end_of_block;

  /* read in the header and allocate a zeroed table */
  rr_fread(&version,sizeof(int),1,fp,"version",0);
  if (version!=HASH_VERSION)
    quit(-1,"%s ERROR: version of '%s' is %d, current version is %d\n",
	     rname, filename, version, HASH_VERSION);
  rr_fread(&ht->max_occupancy,sizeof(double),1,fp,"ht->max_occupancy",0);
  rr_fread(&ht->growth_ratio,sizeof(double),1,fp,"ht->growth_ratio",0);
  rr_fread(&ht->warn_on_update,sizeof(int),1,fp,"ht->warn_on_update",0);
  rr_fread(&ht->nslots,sizeof(int),1,fp,"ht->nslots",0);
  rr_fread(&ht->nentries,sizeof(int),1,fp,"ht->nentries",0);
  ht->slots = (sih_slot_t *) rr_calloc(ht->nslots,sizeof(sih_slot_t));

  /* Read in the array of 'intval's  */
  for (islot=0; islot<ht->nslots; islot++) {
     rr_fread(&(ht->slots[islot].intval),sizeof(int32),1,fp,"intv",0);
  }

  /* Read in the block of strings */
  rr_fread(&total_string_space,sizeof(int),1,fp,"total_string_space",0);
  string_block = (char *) rr_malloc(total_string_space*sizeof(char));
  rr_fread(string_block,sizeof(char),total_string_space,fp,"string_block",0);

  /* make 'string's of non-empty entries point to their corresponding string */
  pstring = string_block;
  past_end_of_block = string_block + total_string_space;
  for (islot=0; islot<ht->nslots; islot++) {
    if (*pstring == (char)NULL) {
	/* an empty entry: */
	ht->slots[islot].string = NULL;
	pstring++;
     }
     else {
        /* a real entry: find the trailing null */
        ht->slots[islot].string = pstring;
        while (*pstring && (pstring < past_end_of_block)) pstring++;
        if (pstring >= past_end_of_block) 
          quit(-1,"%s ERROR: in '%s', string block ended prematurely\n",
		    rname, filename);
        pstring++; /* point to beginning of next string */
     }
  }
  if (pstring!=past_end_of_block) 
    quit(-1,"%s ERROR: some strings remained unaccounted for in %s\n",
	     rname, filename);
  if (verbosity) fprintf(stderr,
     "%s: a hash table of %d entries (%d non-empty) was read from '%s'\n",
      rname, ht->nslots, ht->nentries, filename);

  return(0); /* Not relevant, but stops compilation warnings. */
}




