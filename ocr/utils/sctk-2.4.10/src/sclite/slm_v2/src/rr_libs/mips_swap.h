/* MIPS_SWAP.H  */
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

#ifndef _MIPS_SWAP_H_
#define _MIPS_SWAP_H_

#include "general.h"

#ifdef SLM_SWAP_BYTES    /* reverse byteorder */

/* the following works even for badly aligned pointers */

#define SWAPFIELD(x) {if     (sizeof(*(x))==sizeof(short)) {SWAPHALF((x))}  \
		      else if (sizeof(*(x))==sizeof(int))   {SWAPWORD((x))}  \
		      else if (sizeof(*(x))==sizeof(double)){SWAPDOUBLE((x))}\
		     }

#define SWAPHALF(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+1); \
			*((char*)(x)+1) = tmp_byte;        \
		    }
#define SWAPWORD(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+3); \
			*((char*)(x)+3) = tmp_byte;  	   \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+2); \
			*((char*)(x)+2) = tmp_byte;        \
		    }

#define SWAPDOUBLE(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+7); \
			*((char*)(x)+7) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+6); \
			*((char*)(x)+6) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+2); \
			*((char*)(x)+2) = *((char*)(x)+5); \
			*((char*)(x)+5) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+3); \
			*((char*)(x)+3) = *((char*)(x)+4); \
			*((char*)(x)+4) = tmp_byte;        \
		    }

#if 0 /* old */
#define SWAPHALF(x) *(short*)(x) = ((0xff   & (*(short*)(x)) >> 8) | \
				    (0xff00 & (*(short*)(x)) << 8))
#define SWAPWORD(x) *(int*)  (x) = ((0xff       & (*(int*)(x)) >> 24) | \
				    (0xff00     & (*(int*)(x)) >>  8) | \
                        	    (0xff0000   & (*(int*)(x)) <<  8) | \
				    (0xff000000 & (*(int*)(x)) << 24))
#define SWAPDOUBLE(x) { int *low  = (int *) (x), \
		            *high = (int *) (x) + 1, temp;\
                        SWAPWORD(low);  SWAPWORD(high);\
                        temp = *low; *low = *high; *high = temp;}
#endif /* old */
#else

#define SWAPFIELD(x)
#define SWAPHALF(x)
#define SWAPWORD(x)
#define SWAPDOUBLE(x)

#endif


#define ALWAYS_SWAPFIELD(x) {\
		      if      (sizeof(*(x))==sizeof(short)) {SWAPHALF((x))}  \
		      else if (sizeof(*(x))==sizeof(int))   {SWAPWORD((x))}  \
		      else if (sizeof(*(x))==sizeof(double)){SWAPDOUBLE((x))}\
		     }

#define ALWAYS_SWAPHALF(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+1); \
			*((char*)(x)+1) = tmp_byte;        \
		    }
#define ALWAYS_SWAPWORD(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+3); \
			*((char*)(x)+3) = tmp_byte;  	   \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+2); \
			*((char*)(x)+2) = tmp_byte;        \
		    }

#define ALWAYS_SWAPDOUBLE(x) {char tmp_byte; 			   \
			       tmp_byte = *((char*)(x)+0); \
			*((char*)(x)+0) = *((char*)(x)+7); \
			*((char*)(x)+7) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+1); \
			*((char*)(x)+1) = *((char*)(x)+6); \
			*((char*)(x)+6) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+2); \
			*((char*)(x)+2) = *((char*)(x)+5); \
			*((char*)(x)+5) = tmp_byte;        \
			       tmp_byte = *((char*)(x)+3); \
			*((char*)(x)+3) = *((char*)(x)+4); \
			*((char*)(x)+4) = tmp_byte;        \
		    }

#endif

