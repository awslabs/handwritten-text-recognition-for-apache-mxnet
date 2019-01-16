/* file ss_len.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /**********************************************************************/
 /*                                                                    */
 /*   int substr_length(substr)                                        */
 /*                                                                    */
 /* Returns length of SUBSTRING *substr, in bytes.                     */
 /*                                                                    */
 /**********************************************************************/
 int substr_length(SUBSTRING *substr)
 {int l;
  if ((substr->start == NULL) || (substr->end == NULL))
     l = 0;
  else 
     l = substr->end - substr->start + 1;
  return l;
  } /* end of function "substr_length" */
