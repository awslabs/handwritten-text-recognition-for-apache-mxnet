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


/* Print a message at the standard output if the message's priority is
   sufficiently high. */

#include <stdio.h>
#include <stdarg.h>
 
void pc_message(unsigned short verbosity, 
	       unsigned short priority, 
	       char *msg, ...) {

  va_list args ;

  if (priority <= verbosity) {
 
    va_start(args,msg);
    vfprintf(stderr,msg,args) ;
    va_end(msg);
     
  }

}
