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
#include <string.h>

#include "ngram.h"
#include "toolkit.h"
#include "pc_libs/pc_general.h"
#include "rr_libs/general.h"
#include "idngram2lm.h"
#include "evallm.h"


void main (int argc,char **argv) {

  char *bin_path;
  int verbosity;
  ng_t ng;

  if (pc_flagarg(&argc,argv,"-help") || argc == 1) {
    fprintf(stderr,"binlm2arpa : Convert a binary format language model to ARPA format.\n");
    fprintf(stderr,"Usage : binlm2arpa -binary .binlm\n");
    fprintf(stderr,"                   -arpa .arpa\n");
    fprintf(stderr,"                 [ -verbosity n ]\n");
    exit(1);
  }

  report_version(&argc,argv);

  verbosity = pc_intarg(&argc,argv,"-verbosity",DEFAULT_VERBOSITY);

  bin_path = salloc(pc_stringarg(&argc,argv,"-binary",""));

  if (!strcmp(bin_path,"")) {
    quit(-1,"Error : must specify a binary language model file.\n");
  }

  ng.arpa_filename = salloc(pc_stringarg(&argc,argv,"-arpa",""));

  if (!strcmp(ng.arpa_filename,"")) {
    quit(-1,"Error : must specify an ARPA language model file.\n");
  }

  ng.arpa_fp = rr_oopen(ng.arpa_filename);

  pc_report_unk_args(&argc,argv,verbosity);

  pc_message(verbosity,1,"Reading binary language model from %s...",bin_path);

  load_lm(&ng,bin_path);

  if (verbosity>=2) {
    display_stats(&ng);
  }

  pc_message(verbosity,1,"Done\n");

  write_arpa_lm(&ng,verbosity);

  pc_message(verbosity,0,"binlm2arpa : Done.\n");

  exit(0);

}
