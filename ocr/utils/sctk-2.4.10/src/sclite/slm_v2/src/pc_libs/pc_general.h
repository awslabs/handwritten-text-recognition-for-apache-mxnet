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



/* Function prototypes for pc library */

#ifndef _PCGEN_H_
#define _PCGEN_H_

int pc_flagarg(int *argc, char **argv, char *flag);

char *pc_stringarg(int *argc, char **argv, char *flag, char *value);

int pc_intarg(int *argc, char **argv, char *flag, int value);

double pc_doublearg(int *argc, char **argv, char *flag, double value);

short *pc_shortarrayarg(int *argc, char **argv, char *flag, int elements, 
			int size);

int *pc_intarrayarg(int *argc, char **argv, char *flag, int elements, 
		    int size);

void pc_message(unsigned short verbosity, 
	       unsigned short priority, 
	       char *msg, ...);

void pc_report_unk_args(int *argc, char **argv, int verbosity);

void report_version(int *argc, char **argv);

#endif


