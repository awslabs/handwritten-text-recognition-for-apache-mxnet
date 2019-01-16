/* file pltrim.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /***********************************************/
 /*  pltrim(s)                                  */
 /*  Trims leading non-whitespace from *s,      */
 /* returning s, which is not changed.          */
 /* For a faster version which can be used if   */
 /* *s is not dynamic memory later freed, see   */
 /* pltrimf(s).                                 */
 /***********************************************/
  Char *pltrim(Char *s)
  {Char *from, *to;
   /* skip leading blanks */
   for (from = s; (*from != '\0') && isspace(*from); from++);
   /* copy rest of string */
   for (to = s; *from != '\0'; from++) *to++ = *from;
   *to = '\0';
   return s;
  }
