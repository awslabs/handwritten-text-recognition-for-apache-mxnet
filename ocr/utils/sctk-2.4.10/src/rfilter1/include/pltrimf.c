/* file pltrimf.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /***********************************************/
 /*  pltrimf(s)                                 */
 /*  Returns a pointer to the first             */
 /*  non-whitespace character in the string s.  */
 /*  NOTE: if s is a string that was gotten     */
 /*  with dynamic memory allocation (e.g. with  */
 /*  malloc()) and will be freed later, this    */
 /*  function should NOT be used like this:     */
 /*            s = pltrimf(s)                   */
 /*  because the pointer returned may not be    */
 /*  the one sent down.  Pltrim() is a slower   */
 /*  function that can be used this way.        */
 /*                                             */
 /***********************************************/
  Char *pltrimf(Char *s)
{Char *p1; p1 = s; while (isspace(*p1)) p1++; return p1;}
