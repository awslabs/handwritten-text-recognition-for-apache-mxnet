/* file strcmpi.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /*****************************************************************/
 /*  int strcmpi(ps1,ps2)                                         */
 /*  Strcmpi performs an unsigned comparison of ps1 to ps2,       */
 /*  without case sensitivity.  It returns a value (<0, 0 or >0)  */
 /*  based on the result of comparing ps1 (or part of it) to      */
 /*  ps2 (or part of it).  It is the same as strcmp except it     */
 /*  is not case-sensitive.  Turbo C has this function but ANSI   */
 /*  and BSD 4.2 don't.                                           */
 /*  Returns:  < 0 if *ps1 is less than *ps2                      */
 /*           == 0 if *ps1 is the same as *ps2                    */
 /*            > 0 if *ps1 is greater than *ps2                   */
 /*****************************************************************/
 int strcmpi(Char *ps1, Char *ps2)
{
  Char *px1, *px2;
  int indicator;
  px1 = ps1; px2 = ps2;
  indicator = 9999;
  while (indicator == 9999)
    if (toupper((int)*px1) < toupper((int)*px2)) indicator = -1; 
    else if (toupper((int)*px1) > toupper((int)*px2)) indicator = 1;
         else if (*px1 == '\0') indicator = 0;
              else {px1 += 1; px2 += 1;}
  return indicator;
 }
