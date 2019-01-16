/* file strcutr.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /***********************************************/
 /*  Char *strcutr(ps,n)                        */
 /*  Cuts n characters from tail of *ps.        */
 /***********************************************/
  Char *strcutr(Char *ps, int n)
{Char *p1;
   p1 = ps + strlen(ps) - n;
   if (p1 < ps)
     {fprintf(stderr,"*ERR: strcutr results in string of negative length.\n");
      p1 = ps;}
  *p1 = '\0';
   return ps;
  } /* end of strcutr() */
