
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


/* Generates a pointer to an array of size (vocab_size+1) of flags
   indicating whether or not there should be forced backing off from
   the vocab item, and returns a pointer to it. */

#include "evallm.h"
#include <stdio.h>
#include <string.h>

fb_info *gen_fb_list(sih_t *vocab_ht,
		     int vocab_size,
		     char **vocab,
		     flag *context_cue,
		     flag backoff_from_unk_inc,
		     flag backoff_from_unk_exc,
		     flag backoff_from_ccs_inc,
		     flag backoff_from_ccs_exc,
		     char *fb_list_filename) {

  fb_info *fb_list;
  int i;
  FILE *fb_list_file;
  char current_fb_word[500];
  char inc_or_exc[500];
  int current_fb_id;
  char wlist_entry[1024];

  fb_list = (fb_info *) rr_calloc(vocab_size+1,sizeof(fb_info));

  if (backoff_from_unk_inc) {
    fb_list[0].backed_off = 1;
    fb_list[0].inclusive = 1;
  }

  if (backoff_from_unk_exc) {
    fb_list[0].backed_off = 1;
    fb_list[0].inclusive = 0;
  }

  if (backoff_from_ccs_inc || backoff_from_ccs_exc) {
    for (i=0;i<=vocab_size;i++) {
      if (context_cue[i]) {
	fb_list[i].backed_off = 1;
	if (backoff_from_ccs_inc) {
	  fb_list[i].inclusive = 1;
	}
	else {
	  fb_list[i].inclusive = 0;
	}
      }
    }
  }

  if (strcmp(fb_list_filename,"")) {
    fb_list_file = rr_iopen(fb_list_filename);
    while (fgets (wlist_entry, sizeof (wlist_entry),fb_list_file)) {
      if (strncmp(wlist_entry,"##",2)==0) continue;
      sscanf (wlist_entry, "%s %s",current_fb_word,inc_or_exc);
      if (strncmp(wlist_entry,"#",1)==0) {
	fprintf(stderr,"\n\n===========================================================\n");
        fprintf(stderr,":\nWARNING: line assumed NOT a comment:\n");
	fprintf(stderr,     ">>> %s <<<\n",wlist_entry);
        fprintf(stderr,     "         '%s' will be included in the forced backoff list\n",current_fb_word);
        fprintf(stderr,     "         (comments must start with '##')\n");
	fprintf(stderr,"===========================================================\n\n");
      }
      

      if (sih_lookup(vocab_ht,current_fb_word,&current_fb_id) == 0) {
	fprintf(stderr,"Error : %s in the forced backoff list does not appear in the vocabulary.",current_fb_word);
      }
      


      if (inc_or_exc[0] == 'i' || inc_or_exc[0] == 'I') {
	fb_list[current_fb_id].inclusive = 1;
	fb_list[current_fb_id].backed_off = 1;
      }
      else {
	if (inc_or_exc[0] == 'e' || inc_or_exc[0] == 'E') {
	  fb_list[current_fb_id].inclusive = 0;
	  fb_list[current_fb_id].backed_off = 1;
	}
	else {
	  fprintf(stderr,"Error in line of forced back-off list file.\nLine is : %s\n",wlist_entry);
	}
      }
    }
    rr_iclose(fb_list_file);
  }

  return (fb_list);
}


