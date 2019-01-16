
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

/* Fill array with pointer to arguments (in the same format as argv),
   and store the number of arguments (like argc) */

#include <string.h>
#include <stdio.h>
#include "rr_libs/general.h"

void parse_comline(char *input_line,
		   int *num_of_args,
		   char **args) {

  int next_space;
  char next_word[200];

  *num_of_args = 0;

  while (strlen(input_line) > 0) {
    
    if (input_line[0] == ' ') {
      input_line = &(input_line[1]);
    }
    else {
      next_space = strcspn(input_line," ");

      if (input_line[next_space]==' ') {
	strncpy(next_word,input_line,next_space);
	next_word[next_space] = '\0';
	input_line = &(input_line[next_space+1]);
      }
      else {
	strcpy(next_word,input_line);
	input_line[0]='\0';
      }

      args[*num_of_args] = salloc(next_word);

      (*num_of_args)++;
      
    }
  }
  
}
