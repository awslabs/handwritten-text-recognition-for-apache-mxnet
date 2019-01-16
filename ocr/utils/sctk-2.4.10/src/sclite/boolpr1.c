/* file boolpr1.c */

#include "sctk.h"

 /**********************************************************************/
 /*                                                                    */
 /*   char *bool_print(x)                                              */
 /*                                                                    */
 /* Returns the printing equivalent of the boolean *x.                 */
 /*                                                                    */
 /**********************************************************************/

 char *bool_print(boolean x) {if (x) return "T"; else return "F";}
