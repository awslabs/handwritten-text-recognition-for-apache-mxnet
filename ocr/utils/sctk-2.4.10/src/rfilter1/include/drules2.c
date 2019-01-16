/* file drules2.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

/*****************************************************************/
/*                                                               */
/*  dump_rules2(rset)                                             */
/*  dumps a ruleset to stdout.                                   */
/*                                                               */
/*****************************************************************/
void dump_rules2(RULESET2 *rset)
 {boolean printed[257];
  int i, key_char;
  fprintf(stdout,"\n");
  fprintf(stdout," RULESET2 DUMP\n\n");
  fprintf(stdout,"  name = '%s'\n",rset->name);
  fprintf(stdout,"  directory = '%s'\n",rset->directory);
  fprintf(stdout,"  desc = '%s'\n",rset->desc);
  fprintf(stdout,"  format = '%s'\n",rset->format);
  fprintf(stdout,"  copy_no_hit = '%s'\n",bool_print(rset->copy_no_hit));
  fprintf(stdout,"  case_sensitive = '%s'\n",bool_print(rset->case_sensitive));
  fprintf(stdout,"  indexed = '%s'\n",bool_print(rset->indexed));
  fprintf(stdout,"  max_nrules = %d\n",rset->max_nrules);
  fprintf(stdout,"  nrules = %d\n",rset->nrules);
  if (rset->nrules > 0 )
    {fprintf(stdout,"\n");
     fprintf(stdout,"  #      l.h.s.(length)    r.h.s.(length)  l.context(length) r.context(length)\n");
     for (i = 1; i <= rset->nrules; i++)
       {fprintf(stdout,"%4d [%s] (%d) => [%s] (%d) /  [%s] (%d) __ [%s] (%d)\n",
          i,
          rset->rule[i].sin, rset->rule[i].sinl,
          rset->rule[i].sout, rset->rule[i].soutl,
          rset->rule[i].lcontext, rset->rule[i].lcontextl,
          rset->rule[i].rcontext,rset->rule[i].rcontextl);
    }  }
  if (rset->indexed)
    {fprintf(stdout,"\n");
     fprintf(stdout," RULE TAB-SORT INDEX TABLE:\n");
     fprintf(stdout,"     I  RULE_INDEX[I]\n");
     for(i=1; i <= rset->nrules; i++) fprintf(stdout,"%6d  %6d\n",i,rset->rule_index[i]);
     fprintf(stdout,"\n");
     fprintf(stdout," RULE KEY_CHAR INDICES:\n");
     fprintf(stdout," KEY_CHAR  FIRST  LAST\n");
     for(i=0; i <= 256; i++) printed[i] = F;
     for(i=0; i <= 256; i++)
       {key_char = i;
        if (!rset->case_sensitive) key_char = toupper(key_char);
        if (rset->last_rule[key_char] > 0)
          {if (!printed[key_char])
             fprintf(stdout,"%5d(%c) %6d%6d\n", key_char, key_char,
               rset->first_rule[key_char], rset->last_rule[key_char]);
           printed[key_char] = T;
    }  }  }
  fprintf(stdout,"\n");
  fflush(stdout);
  return;
} /* end dump_rules */
