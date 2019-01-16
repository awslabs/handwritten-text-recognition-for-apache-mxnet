/* file frules2.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

  /************************************************************/
  /*                                                          */
  /*  free_rules2(rset);                                      */
  /*                                                          */
  /* Frees memory allocated for rule set *rset.               */
  /************************************************************/
  void free_rules2(RULESET2 *rset)
   {Char *proc = "free_rules2";
/* data */
    int i;
/* coding  */
 db_enter_msg(proc,1); /* debug only */
    free(rset->name);
    free(rset->directory);
    free(rset->format);
    free(rset->desc);
    free(rset->rule_index);
    free(rset->first_rule);
    free(rset->last_rule);
    for(i=1; i <= rset->nrules; i++)
      {free(rset->rule[i].sin);
       free(rset->rule[i].sout);
       free(rset->rule[i].lcontext);
       free(rset->rule[i].rcontext);
      }
    free(rset->rule);
 db_leave_msg(proc,1); /* debug only */
    return;
  } /* end free_rules2 */
