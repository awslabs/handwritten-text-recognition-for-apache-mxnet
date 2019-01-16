/* read_vocab: create a vocabulary hash table and optionally a vocabulary direct-access table */
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

/*  If the file's extension is "vocab_ht" (or the file's name is "vocab_ht"),
       it is assumed to be a pre-compiled vocabulary hash table.  Otherwise
       it is assumed to be a regular .vocab file.
*/

/* Edited by Philip Clarkson, March 1997 to prevent compilation warnings */


#include <stdio.h>
#include <strings.h>
#include "general.h"
#include "sih.h"

void read_voc(char *filename, int verbosity,   
	      sih_t *p_vocab_ht, char ***p_vocab, 
	      unsigned short *p_vocab_size)
/* p_vocab==NULL means: build only a hash table */
{

  /*  static char rname[] = "rd_voc"; */  /* Never used anyway! */
  char *pperiod;
  int   vocab_size;

  pperiod = rindex(filename,'.');
  if (pperiod==NULL) pperiod = filename-1;

  if (strcmp(pperiod+1,"vocab_ht")==0) { 	     /* file == hash_table */
     FILE *fp=rr_iopen(filename);
     sih_val_read_from_file(p_vocab_ht, fp, filename, verbosity);
     rr_iclose(fp);
     vocab_size = p_vocab_ht->nentries;
     if (p_vocab!=NULL) {
        get_vocab_from_vocab_ht(p_vocab_ht, vocab_size, verbosity, p_vocab);
	*p_vocab[0] = salloc("<UNK>");
     }
  }
  else {					     /* file == vocab(ascii) */
     read_wlist_into_siht(filename, verbosity, p_vocab_ht, &vocab_size);
     if (p_vocab!=NULL) {
        read_wlist_into_array(filename, verbosity, p_vocab, &vocab_size);
        *p_vocab[0] = salloc("<UNK>");
     }
  }

  if (p_vocab_size) {
    *p_vocab_size = vocab_size;
  }
  
}


/* derive the vocab from the vocab hash table */

void get_vocab_from_vocab_ht(sih_t *ht, int vocab_size, int verbosity, char ***p_vocab)
{
  static char rname[]="get_vocab_fm_ht";
  char   **wlist;
  int    islot, wordid;

  wlist = (char **) rr_malloc((vocab_size+1)*sizeof(char *));

  for (islot=0; islot<ht->nslots; islot++) {
     wordid = (int) ht->slots[islot].intval;
     if (wordid>0) wlist[wordid] = ht->slots[islot].string;
  }

  for (wordid=1; wordid<=vocab_size; wordid++)
    if (wlist[wordid]==NULL)
      quit(-1,"%s ERROR: the hash table does not contain wordid %d\n",
	       rname, wordid);

  if (verbosity) fprintf(stderr,
     "%s: vocabulary was constructed from the vocab hash table\n",rname);
  *p_vocab = wlist;
}
