/* file textlen.c */ 

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /*********************************************/
 /*  textlen(s)                               */
 /*  returns length of text in the string *s, */
 /*  ignoring head and tail whitespace.       */
 /*  (Uses the version of pltrim that doesn't */
 /*  shift the data.)                         */
 /*********************************************/

  int textlen(Char *s) {return (int)(prtrim(s) - pltrimf(s) + 1);}
