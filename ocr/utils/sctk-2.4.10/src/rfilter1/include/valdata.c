/* file valdata.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif


  /***************************************************************/
  /*  boolean valid_data_line(s,comment_flag)                    */
  /*  Indicates whether or not *s is a valid non-empty data line.*/
  /*  Changed 7/22/93 to return "T" if comment_flag is empty.    */
  /***************************************************************/
  boolean valid_data_line(Char *s, Char *comment_flag)
  {boolean ok;
   Char *px, *py;
   if (strlen(s) < 1)
     ok = F;
   else
     {if (strlen(comment_flag) < 1) ok = T;
      else
        {py = s;
         for (px=comment_flag; *px != '\0'; px++)
            if (*px != *(py++)) {ok = T; goto RETURN;}
         ok = F;
     }  }
 RETURN:
   return ok;
  } /* end valid_data_line */
