/* file boolpr1.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /**********************************************************************/
 /*                                                                    */
 /*   Char *bool_print(x)                                              */
 /*                                                                    */
 /* Returns the printing equivalent of the boolean *x.                 */
 /*                                                                    */
 /**********************************************************************/

 Char *bool_print(boolean x) {if (x) return "T"; else return "F";}
