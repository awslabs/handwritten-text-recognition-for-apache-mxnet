/* file expenv.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

  /*****************************************************************/
  /*                                                               */
  /*  Char *expenv(s,slength)                                      */
  /*  Expands environment variables embedded in string *s.         */
  /*  Doesn't expand beyond s[slength].                            */
  /*  Uses local buffers of length LONG_LINE.                      */
  /*  Returns pointer to s so you can use it like s = expenv(s,x); */
  /*                                                               */
  /*****************************************************************/
  Char *expenv(Char *s, int slength)
   {Char *proc = "expenv";
    Char *pa, *pb, *pend;
    Char sxx[LONG_LINE], *sx = &sxx[0];
    Char vxx[LONG_LINE], *vname = &vxx[0];
    Char sex[LONG_LINE], *s_exp = &sex[0];
    boolean ok, no_problem;
    SUBSTRING ssx_data, *ssx = &ssx_data;
  /* instructions */
    db_enter_msg(proc,2);
if (db_level > 2) printf("%s s='%s'\n",pdb,s); 
    if (s != NULL)
      {sx = strcpy(sx,"");
       pa = s;
       pb = sx;
       pend = sx + slength;
       no_problem = T;
       while ((*pa != '\0') && no_problem)
         {if (*pa == '$')
	    {/* find possible environment variable */
             ssx->start = pa+1;
             ssx->end = ssx->start;
             for (ok=T; ok; ++(ssx->end))
               {if ( (!isalnum(*(ssx->end))) &&
                     (*(ssx->end) != '_'))
                 ok = F;
	       }
             ssx->end -= 2;
             substr_to_str(ssx,vname,LONG_LINE);
if (db_level > 2) printf("%s env var name = '%s'\n",pdb,vname);
             s_exp = getenv(vname);
if (db_level > 2) printf("%s s_exp = '%s'\n",pdb,s_exp);
             if (s_exp == NULL)
               {if (pb < pend) *(pb++) = *(pa++);
                else
                  {fprintf(stderr,"*ERR:%s:LONG_LINE too small.\n",proc);
                   no_problem = F;
	       }  }
             else
               {while ((*s_exp != '\0') && no_problem)
                  {if (pb < pend) *(pb++) = *(s_exp++);
                   else
                     {fprintf(stderr,"*ERR:%s:LONG_LINE too small.\n",proc);
                      no_problem = F;
		  }  }
                pa = ssx->end + 1;
	    }  }
          else if (pb < pend) *(pb++) = *(pa++);
         }
       *pb = '\0';
       s = strncpy(s,sx,slength);
      }
    db_leave_msg(proc,2);
    return s;
   } /* end expenv */
