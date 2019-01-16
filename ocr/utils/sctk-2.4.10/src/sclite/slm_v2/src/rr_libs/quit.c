/* quit():  print a message and exit */
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

/* Portable version courtesy of Ralf Brown, 7/94 */

#include <stdio.h>
#include <stdarg.h>
 
int quit(int rc, char *msg, ...)
{
   va_list args ;
 
   va_start(args,msg) ;
   vfprintf(stderr,msg,args) ;
   va_end(msg) ;
   exit(rc) ;
}
