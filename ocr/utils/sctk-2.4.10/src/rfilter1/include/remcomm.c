/* file remcomm.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif


  /***************************************************************/
  /* remove_comments(s,comment_flag)                             */
  /* Removes comments from *s, replacing them with space,        */
  /* returning s.  ("Comments" are everything after the first    */
  /* occurrance of *comment_flag.)                               */
  /* Note: this should not just insert '\0' at the beginning of  */
  /* the comments, in case s was dynamically allocated.          */
  /* Changed 7/22/93 to do nothing if comment_flag is empty.     */
  /***************************************************************/
  Char *remove_comments(Char *s, Char *comment_flag)
  {Char *px;
   if (strlen(comment_flag) > 0)
     {px = strstr(s,comment_flag);
      if (px != NULL) while (*px != '\0') *px++ = ' ';
     }
   return s;
  } /* end remove_comments() */
