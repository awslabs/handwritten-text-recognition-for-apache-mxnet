/* file drules3.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

/*****************************************************************/
/*                                                               */
/*  dump_rules3(rset,fp)                                         */
/*  dumps a ruleset to the file *fp.                             */
/*                                                               */
/*****************************************************************/
void dump_rules3(RULESET2 *rset, FILE *fp)
 {boolean printed[257];
  int i, key_char;
  fprintf(fp,"\n");
  fprintf(fp," RULESET2 DUMP\n\n");
  fprintf(fp,"  name = '%s'\n",rset->name);
  fprintf(fp,"  directory = '%s'\n",rset->directory);
  fprintf(fp,"  desc = '%s'\n",rset->desc);
  fprintf(fp,"  format = '%s'\n",rset->format);
  fprintf(fp,"  copy_no_hit = '%s'\n",bool_print(rset->copy_no_hit));
  fprintf(fp,"  case_sensitive = '%s'\n",bool_print(rset->case_sensitive));
  fprintf(fp,"  indexed = '%s'\n",bool_print(rset->indexed));
  fprintf(fp,"  max_nrules = %d\n",rset->max_nrules);
  fprintf(fp,"  nrules = %d\n",rset->nrules);
  if (rset->nrules > 0 )
    {fprintf(fp,"\n");
     fprintf(fp,"   #  val1  l.h.s.(length)    r.h.s.(length)  l.context(length) r.context(length)\n");
     for (i = 1; i <= rset->nrules; i++)
       {fprintf(fp,"%4d %4d   [%s] (%d) => [%s] (%d) /  [%s] (%d) __ [%s] (%d)\n",
          i,
          rset->rule[i].val1,
          rset->rule[i].sin, rset->rule[i].sinl,
          rset->rule[i].sout, rset->rule[i].soutl,
          rset->rule[i].lcontext, rset->rule[i].lcontextl,
          rset->rule[i].rcontext,rset->rule[i].rcontextl);
    }  }
  if (rset->indexed)
    {fprintf(fp,"\n");
     fprintf(fp," RULE TAB-SORT INDEX TABLE:\n");
     fprintf(fp,"     I  RULE_INDEX[I]\n");
     for(i=1; i <= rset->nrules; i++) fprintf(fp,"%6d  %6d\n",i,rset->rule_index[i]);
     fprintf(fp,"\n");
     fprintf(fp," RULE KEY_CHAR INDICES:\n");
     fprintf(fp," KEY_CHAR  FIRST  LAST\n");
     for(i=0; i <= 256; i++) printed[i] = F;
     for(i=0; i <= 256; i++)
       {key_char = i;
        if (!rset->case_sensitive) key_char = toupper(key_char);
        if (rset->last_rule[key_char] > 0)
          {if (!printed[key_char])
             fprintf(fp,"%5d(%c) %6d%6d\n", key_char, key_char,
               rset->first_rule[key_char], rset->last_rule[key_char]);
           printed[key_char] = T;
    }  }  }
  fprintf(fp,"\n");
  fflush(fp);
  return;
} /* end dump_rules3 */
