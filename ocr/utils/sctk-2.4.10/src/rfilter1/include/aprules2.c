/* file aprules2.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

/*****************************************************************/
/*                                                               */
/*  Char *apply_rules2(pb,pa,rset,perr)                          */
/*  Applies RULESET2 *rset to string pa, producing string pb.    */
/*  Returns pb.                                                  */
/*  If rset == NULL, just copies over the input string.          */
/*  On return, *perr == 0 indicates no error; otherwise,         */
/*     11 means "invalid rule format"                            */
/*                                                               */
/*****************************************************************/
Char *apply_rules2(Char *pb, Char *pa, RULESET2 *rset, int *perr)
 {Char *proc = "apply_rules2";
  Char *pi, *pbx;
  int irule, key_char, jrule;
  boolean hit;
/* code: */
 db_enter_msg(proc,1); /* debug only */
  *perr = 0;
if (db_level > 1) printf("%s Input: '%s'\n",pdb,pa);
  if (rset == NULL)
    {pb = strcpy(pb,pa);
     goto RETURN;
    }
  if (!streq(rset->format,"NIST1"))
    {fprintf(stderr,"*ERR:%s: invalid format '%s'\n",proc,rset->format);
     *perr = 11;
     goto RETURN;
    }
  pi = pa;
  while (*pi != '\0')
    {hit = F;
if (db_level > 1) printf("%s *pi = '%s'\n",pdb,pi);
     key_char = (int)*pi;
     if (!rset->case_sensitive) key_char = toupper(key_char);
if (db_level > 1) printf("%s key_char = %d (%c)\n",pdb,key_char,key_char);
     for (jrule = rset->first_rule[key_char]; (!hit) && (jrule <= rset->last_rule[key_char]); jrule++)
       {irule = rset->rule_index[jrule];
if (db_level > 1) printf("%s jrule=%d, irule=%d\n",pdb,jrule,irule);
        if (streq(rset->format,"NIST1"))
          {if ( ((rset->case_sensitive)&&
                 (strncmp(pi,rset->rule[irule].sin,rset->rule[irule].sinl)==0)&&
                 (strncmp(pi-rset->rule[irule].lcontextl,
                          rset->rule[irule].lcontext,
                          rset->rule[irule].lcontextl)==0)&&
                 (strncmp(pi+rset->rule[irule].sinl,
                          rset->rule[irule].rcontext,
                          rset->rule[irule].rcontextl)==0))
             || ((!rset->case_sensitive)&&
                 (strncmpi(pi,rset->rule[irule].sin,rset->rule[irule].sinl)==0)&&
                 (strncmpi(pi-rset->rule[irule].lcontextl,
                          rset->rule[irule].lcontext,
                          rset->rule[irule].lcontextl)==0)&&
                 (strncmpi(pi+rset->rule[irule].sinl,
                          rset->rule[irule].rcontext,
                          rset->rule[irule].rcontextl)==0))
	      )
             {hit = T;
              pb = strcat(pb,rset->rule[irule].sout);
              pi += rset->rule[irule].sinl;
if (db_level > 1) printf("%s rule fired\n",pdb);
              rset->rule[irule].val1 += 1;
             }
if (db_level > 2) printf("%s irule=%d, lcontextl=%d, lcont match = %s\n",
pdb,irule,rset->rule[irule].lcontextl,
bool_print((strncmpi(pi-rset->rule[irule].lcontextl,
                    rset->rule[irule].lcontext,
                    rset->rule[irule].lcontextl)==0)));
          }
       }
     if (!hit)
       {if (rset->copy_no_hit)
          {pbx = pb + strlen(pb);
           *pbx = *pi; /* copy Char from pi to end of pb */
           *(++pbx) = '\0';
          }
        pi += 1;
    }  }
 RETURN:
if (db_level > 1) printf("%s Output:'%s'\n",pdb,pb);
 db_leave_msg(proc,1); /* debug only */
  return pb;
} /* end apply_rules2 */
