/* file gcomflag.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif


  /***************************************************************/
  /* get_comment_flag(s,comment_flag)                            */
  /* Gets *comment_flag as first token in *s.                    */
  /***************************************************************/
  void get_comment_flag(Char *s, Char *comment_flag)
  {Char *px;
   comment_flag = strcpy(comment_flag,s);
   for (px = comment_flag; isgraph(*px); px++);
   *px = '\0';
if (db_level > 1) printf("*DB: comment_flag='%s'\n",comment_flag);
   return;
  } /* end get_comment_flag */
