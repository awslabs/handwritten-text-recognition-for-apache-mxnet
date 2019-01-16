/* parse_line(): given a text line, return pointers to word begin/end positions. */
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

#include <ctype.h>

void parse_line(
	char *line, int mwords, int canonize,
	char **pword_begin, char **pword_end, int *p_nwords, int *p_overflow)
{
  char *pl, *psq, *ptmp, *pbegin, *pend;
  int  nwords=0;

  *p_overflow = 0;
  pl = line-1;
  psq = line;
  do {
     do pl++; while (isspace(*pl));           /* find beginning of next word */
     if (*pl==0) break;			      /* no more words */
     if (nwords>=mwords) {*p_overflow=1; break;} /* no room for next word */
     nwords++;
     pbegin = pl;
     do pl++; while (!isspace(*pl) && *pl!=0); /* find end of current word */
     pend = pl;   /* (word ends in whitespace or e.o.line) */

     if (canonize) {
        *pword_begin++ = psq;
        if (psq!=pbegin) for (ptmp=pbegin; ptmp<pend;) *psq++ = *ptmp++;
        else psq = pend;
        *pword_end++ = psq;
        *psq++ = ' ';
     }
     else {
        *pword_begin++ = pbegin;
        *pword_end++ = pend;
     }
  } while (*pl!=0);

  if (canonize) **(pword_end-1) = '\0';
  *p_nwords = nwords;
}
