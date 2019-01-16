/*        EXPORT VERSION OF:                                       */
/* rfilter1.c - filter that applies a rule set.                    */
/*  This version recognizes a context-sensitive rule formalism.    */
/*                                                                 */
/* Synopsis: rfilter1 fname_rules (db_level) < file1 > file2       */
/* Process: Applies rules in file <fname_rules> to STDIN, writing  */
/*   result to STDOUT.                                             */
/* Rules in general are of the form A => B / C __ D                */
/* command-line parameters:                                        */
/* par #   contents                                                */
/*   1     fname_rules (name of file holding rules)                */
/*   2*    db_level (level of de-bug printf's to do)               */
/*      (if db_level = 1, prints no. of hits for each rule,stderr) */
/* (*=optional)                                                    */
/*                                                                 */
/* Modified 11/8/96 by WMF to:                                     */
/*  - increase input buffer size                                   */
/*  - provide a warning error message if input seems truncated     */
/*  - free all allocated memory on program exit                    */
/*  - recognize a header record in the rules file that specifies   */
/*    case insensitivity for the matching parts of a rule          */
/*  - recognize optional left & right contexts in a rule           */
/*  - use a left-corner indexing scheme for speed                  */
/*******************************************************************/

/* standard compile environment files: */
#include "include/stdcenvp.h"

int main(int argc, char **argv)
 {char *proc = "RFILTER1 Ver. 2.0";
  time_t t_sec;
  Char rxf[LONG_LINE], *fname_rules = &rxf[0];
#define MAX_INPUT_REC 100000
  Char pax[MAX_INPUT_REC], *pa = &pax[0];
  Char pbx[MAX_INPUT_REC], *pb = &pbx[0];
  Char *path = "";
  int err; int *perr = &err;
  RULESET2 rset_data, *rset = &rset_data;
/*   code   */
  t_sec = time(&t_sec);      /* ANSI, Turbo C */
  pdb = strcpy(pdb,"*DB: ");
  if (argc < 2)
    {fprintf(stdout,"*ERR: no command-line parameters were given.\n");
     fprintf(stdout," %s synopsis:\n",proc);
     fprintf(stdout,"  rfilter1 rule_file_name (db_level) < in_file > out_file\n");
     return 2;
    }
  fname_rules = argv[1];
  if (argc > 2) db_level  = atoi(argv[2]); else db_level = 0;
  get_rules2(rset, path, fname_rules, perr);
  if (*perr > 0)
    {fprintf(stdout,"; *ERR: Get_rules2 returned err code =%d\n",*perr);
     return 3;
    }
if (db_level > 1) printf("%s No. of rules = %d\n",pdb,rset->nrules);
if (db_level > 2)
  {printf("%s *DB: the rules are:\n",pdb);
   dump_rules2(rset);
  }

/* exercise the rules */
  while (fgets(pa,MAX_INPUT_REC,stdin) != NULL)
    {if (*(pa + strlen(pa) - 1) != '\n')
       {fprintf(stderr,"*WARNING:%s: rec w/o newline char,",proc);
        fprintf(stderr," probably truncated.\n  Line is:'%s'\n",pa);
       }
if (db_level > 1) printf("%s pa=%s",pdb,pa);
     pb = strcpy(pb,"");
     pb = apply_rules2(pb,pa,rset,perr);
if (db_level > 1) printf("%s pb=%s",pdb,pb);
     if (*perr == 0) fprintf(stdout,"%s",pb);
     else
       {fprintf(stderr,"*ERR: apply_rules2 returns perr=%d\n",*perr);
        return 4;
    }  }
if (db_level > 0)
  {fprintf(stderr,
    "%s *DB: At end of run, the rules (with val1= no. of hits) are:\n",pdb);
   dump_rules3(rset,stderr);
  }
  free_rules2(rset);
  return 0;
 } /* end main */

/* copies of functions from libraries: */
#include "include/aprules2.c"
#include "include/drules2.c"
#include "include/drules3.c"
#include "include/frules2.c"
#include "include/grules2.c"
#include "include/dbpkg1.c"
#include "include/pltrimf.c"
#include "include/expenv.c"
#include "include/strdup2.c"
#include "include/boolpr1.c"
#include "include/ss_len.c"
#include "include/mfname1.c"
#include "include/prtrim.c"
#ifdef NEED_STRCMP
#include "include/strncmpi.c"
#endif
#include "include/calloc2.c"
#include "include/prtrim2.c"
#include "include/mupper.c"
#include "include/valdata.c"
#include "include/ss_to_s.c"
#include "include/remcomm.c"
#include "include/atobool.c"
#include "include/gcomflag.c"
#include "include/textlen.c"
#include "include/del_eol.c"
#include "include/frstr1.c"
#include "include/sstok2.c"
#include "include/fatalerr.c"
#include "include/strcutr.c"
#include "include/str_eq.c"
#include "include/pltrim.c"
#ifdef NEED_STRCMP
#include "include/strcmpi.c"
#endif
