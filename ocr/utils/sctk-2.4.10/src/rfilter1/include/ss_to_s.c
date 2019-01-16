/* file ss_to_s.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 /**********************************************************************/
 /*                                                                    */
 /*   Char *substr_to_str(substr,str,lmax)                             */
 /*                                                                    */
 /* Converts the SUBSTRING *substr to the character string *str,       */
 /* not putting more than *lmax characters in (including the new end-  */
 /* of-string character).  If this forces the                          */
 /* truncation of some characters, the tail of *str, up to 2           */
 /* characters long, is returned as "**".                              */
 /* Returns pointer to str.                                            */
 /*                                                                    */
 /**********************************************************************/
 Char *substr_to_str(SUBSTRING *substr, Char *str, int lmax)
{Char *proc = "substr_to_str";
  Char *pn1, *pn2;
  int next_to_be_moved;
  boolean ovf;
  str = strcpy(str,"");
  if ((substr->start == NULL) || (substr->end == NULL)) goto RETURN;
  if (substr_length(substr) < 1) goto RETURN;
/* move characters - at most (lmax-1) of them, leaving room */
/* for the eos flag                                         */
  ovf = F;
  pn2 = str;
  next_to_be_moved = 1;
  for (pn1 = substr->start; (!ovf && pn1 <= substr->end); pn1++)
    {if (next_to_be_moved++ < lmax) *pn2++ = *pn1; else ovf = T;
    }
 if (ovf)
   {*(pn2 - 1) = '*';
    *(pn2 - 2) = '*';
   }
  *pn2 = '\0';
RETURN:
if (db_level > 2) printf("*DB: %s yields '%s'\n",proc,str);
  return str;
  } /* end of function "substr_to_str" */
