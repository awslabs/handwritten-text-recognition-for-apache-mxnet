/* file mupper.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /*********************************************/
 /*  make_upper(s)                            */
 /*  makes *s be all upper case               */
 /*  returns pointer to *s                    */
 /*********************************************/
  Char *make_upper(Char *s)
  {Char *pi;
   pi = s;
   while (*pi != '\0')
      {*pi = (Char)toupper((int)*pi);
       pi += 1;
      }
   return s;
  }
