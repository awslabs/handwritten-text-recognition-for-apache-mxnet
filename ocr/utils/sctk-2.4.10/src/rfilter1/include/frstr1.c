 /* file frstr1.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

void free_str(Char *s)
{if ((s != NULL)&&(s != EMPTY_STRING)&&(s != (Char*)EMPTY_INT_STRING))
     {if (memory_trace) printf("*DB: FREE %x\n",(int)s);
      free((Char*)s);
     }
   return;
 }
