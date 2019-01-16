
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

#define VERSION 2.03

typedef unsigned short flag;

#endif
