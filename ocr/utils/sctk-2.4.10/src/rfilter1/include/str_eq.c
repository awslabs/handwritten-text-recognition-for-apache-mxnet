 /* file str_eq.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /*********************************************/
 /*  string_equal(cs,ct,ignore_case)          */
 /*  returns T is cs == ct, ignoring case iff */
 /*  "ignore_case" is T.                      */
 /*********************************************/
 boolean string_equal(Char *cs, Char *ct, int ignore_case)
 {if (ignore_case) return streqi(cs,ct); else return streq(cs,ct);
 }
