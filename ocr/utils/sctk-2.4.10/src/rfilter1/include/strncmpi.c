/* file strncmpi.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /*****************************************************************/
 /*  int strncmpi(ps1,ps2,n)                                      */
 /*  Strncmpi performs an unsigned comparison of at most n        */
 /* characters of string ps1 to ps2, without case sensitivity.    */
 /* It returns a value (<0, 0 or >0)                              */
 /*  based on the result of comparing ps1 (or part of it) to      */
 /*  ps2 (or part of it).  It is the same as strncmp except it    */
 /*  is not case-sensitive.  Turbo C has this function but ANSI   */
 /*  and BSD 4.2 don't.                                           */
 /*  Returns:  < 0 if *ps1 is less than *ps2 (first n chars)      */
 /*           == 0 if *ps1 is the same as *ps2 (first n chars)    */
 /*            > 0 if *ps1 is greater than *ps2 (first n chars)   */
 /*  Revised 11/5/96 by WMF to fix a bug when n == 0.             */
 /*****************************************************************/
 int strncmpi(Char *ps1, Char *ps2, int n)
 {Char *proc = "strncmpi";
  Char *px1, *px2;
  int indicator, i=0;
 db_enter_msg(proc,2); /* debug only */
  px1 = ps1; px2 = ps2;
  indicator = 9999;
  while (indicator == 9999)
    {if (++i > n) indicator = 0;
     else
       {if (*px1 == '\0')
          {if (*px2 == '\0') indicator = 0;
           else              indicator = -1;
	  }
        else
          {if (toupper((int)*px1) < toupper((int)*px2)) indicator = -1; 
           else
             {if (toupper((int)*px1) > toupper((int)*px2)) indicator = 1;
              else {px1 += 1; px2 += 1;}
    }  }  }  }
 db_leave_msg(proc,2); /* debug only */
  return indicator;
 }
