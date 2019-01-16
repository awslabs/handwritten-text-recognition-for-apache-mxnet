/* file grules2.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

/***********************************************************/
/* get_rules2 and its sub-functions                        */
/* Created 11/4/96 by WMF to include context sensitivity,  */
/* based on get_rules1.                                    */
/***********************************************************/

  /************************************************************/
  /*   r_process_aux_line(rset,pline,perr)                    */
  /*                                                          */
  /*   error codes: 3: bungled aux line                       */
  /*   (Fatal error if memory allocation fails.)              */
  /************************************************************/
  static void r_process_aux_line(RULESET2 *rset, Char *pline, int *perr)
   {Char *proc = "r_process_aux_line";
    Char svx[LINE_LENGTH], *sval = &svx[0];
    Char upx[LINE_LENGTH], *upr_pline = &upx[0];
    SUBSTRING sspx, *ssp = &sspx;
/* coding */
 db_enter_msg(proc,1); /* debug only */
    upr_pline = make_upper(strcpy(upr_pline,pline));
  /* locate value in quotes */
    *ssp = sstok2(pline,"\"'");
    if (substr_length(ssp) < 0)
      {fprintf(stderr,"%s: aux line contains no quoted field.\n",proc);
       fprintf(stderr," line:'%s'\n",pline);
       *perr=3;
      }
    else
      {sval = substr_to_str(ssp,sval,LINE_LENGTH);
/* identify field and store its value */
       if (strstr(upr_pline,"NAME"))
         {free_str(rset->name);
          rset->name       = strdup_safe(sval,proc);
	 }
       if (strstr(upr_pline,"DESC"))
         {free_str(rset->desc);
          rset->desc       = strdup_safe(sval,proc);
	 }
       if (strstr(upr_pline,"FORMAT"))
         {free_str(rset->format);
          rset->format       = strdup_safe(sval,proc);
	 }
       if (strstr(upr_pline,"MAX_NRULES")) rset->max_nrules = atol(sval);
       if (strstr(upr_pline,"COPY_NO_HIT")) rset->copy_no_hit = atobool(sval);
       if (strstr(upr_pline,"CASE_SENSITIVE")) rset->case_sensitive = atobool(sval);
      }
 db_leave_msg(proc,1); /* debug only */
   } /* end r_process_aux_line */


  /************************************************************/
  /*   r_process_data_line(rset,pline,perr)                   */
  /* Process a data line; assumes comments have been removed. */
  /*    error codes:   1: no valid information in pline       */
  /*                  31: memory ovf (too many rules)         */
  /*                  32: no arrow                            */
  /*   (Fatal error if memory allocation fails.)              */
  /************************************************************/
  static void r_process_data_line(RULESET2 *rset, Char *pline, int *perr)
   {Char *proc = "r_process_data_line";
    Char *ieq, *icontext, *islot, *ilmark, *irmark;
    Char sx_data[LINE_LENGTH], *sx = &sx_data[0];
    RULE2 x, *rx = &x;
    SUBSTRING ssx_data, *ssx = &ssx_data;
 db_enter_msg(proc,1); /* debug only */
    *perr = 0;

if (db_level > 1) printf("%s data line '%s'\n",pdb,pline);

  /* check - blank line? */
    if (textlen(pline) < 1)
      {fprintf(stderr,"%s invalid input:blank line.\n",proc);
       *perr=1; goto RETURN;
      }
  /* check for which format  */
  /* NIST1: literal ascii, optionally context-sensitive  */
    if (streq(rset->format,"NIST1"))
      {/* get major boundary elements */

if (db_level > 1) printf("%s processing as format NIST1\n",pdb);

       ieq = strstr(pline,"=>");
       if (ieq == NULL)
         {fprintf(stderr,"*ERR:%s: no arrow, line='%s'\n",proc,pline);
          *perr = 32; goto RETURN;
	 }
/* context marker is last slash IFF not enclosed in [] */
       icontext = strrchr(pline,'/');
       if (icontext != NULL)
         {irmark = strchr(icontext,']');
          ilmark = strchr(icontext,'[');
          if (irmark < ilmark) /* enclosed in [] */
            {icontext = NULL;
	    }
          else
            {islot = strstr(icontext,"_");
	    }
	 }
     /* initialize rule */
       rx->sin = strdup_safe("",proc);
       rx->sinl = 0;
       rx->sout = strdup_safe("",proc);
       rx->soutl = 0;
       rx->lcontext = strdup_safe("",proc);
       rx->lcontextl = 0;
       rx->rcontext = strdup_safe("",proc);
       rx->rcontextl = 0;
     /* get left hand side */
if (db_level > 3) printf("%s getting left hand side\n",pdb);
       ssx->start = pltrimf(pline);
       ssx->end = ieq;
       if (ssx->start < ieq)
         {ssx->end -= 1;
          while ((ssx->end > ssx->start)&&(isspace(*(ssx->end)))) ssx->end -= 1;
      /* remove boundary characters, if any */
          if (((*ssx->start == '[')&&(*ssx->end == ']')) ||
              ((*ssx->start == '"')&&(*ssx->end == '"')) )
            {ssx->start += 1;
             ssx->end -= 1;
	    }
          rx->sinl = substr_length(ssx);
          free(rx->sin);
          rx->sin = strdup_safe(substr_to_str(ssx,sx,LINE_LENGTH),proc);
if (db_level > 3) printf("%s lhs='%s',len=%d\n",pdb,rx->sin,rx->sinl);
	 }
     /* get right hand side */
if (db_level > 3) printf("%s getting right hand side\n",pdb);
       ssx->start = ieq;
       if (icontext == NULL) ssx->end = prtrim(pline);
       else
          for (ssx->end = icontext-1;
               isspace(*(ssx->end))&&(ssx->end > ssx->start);
               ssx->end -= 1);
       while ((!isspace(*(ssx->start)))&&(ssx->start < ssx->end)) ssx->start += 1;
       if (ssx->start < ssx->end)
         {while (isspace(*(ssx->start))) ssx->start += 1;
      /* remove boundary characters, if any */
          if (((*ssx->start == '[')&&(*ssx->end == ']')) ||
              ((*ssx->start == '"')&&(*ssx->end == '"')) )
            {ssx->start += 1;
             ssx->end -= 1;
	    }
          rx->soutl = substr_length(ssx);
          free(rx->sout);
          rx->sout = strdup_safe(substr_to_str(ssx,sx,LINE_LENGTH),proc);
if (db_level > 3) printf("%s rhs='%s',len=%d\n",pdb,rx->sout,rx->soutl);
	 }
       /* get context, if any */
       if ((icontext != NULL)&&(islot != NULL)) /* context-sensitive formalism? */
         {/* get left context */
if (db_level > 3) printf("%s getting left context\n",pdb);
          ssx->start = icontext;
          for (ssx->end = islot-1;
               isspace(*(ssx->end))&&(ssx->end > ssx->start);
               ssx->end -= 1);
          while ((!isspace(*(ssx->start)))&&(ssx->start < ssx->end)) ssx->start += 1;
          if (ssx->start < ssx->end)
            {while (isspace(*(ssx->start))) ssx->start += 1;
            /* remove boundary characters, if any */
             if (((*ssx->start == '[')&&(*ssx->end == ']')) ||
                 ((*ssx->start == '"')&&(*ssx->end == '"')) )
               {ssx->start += 1;
                ssx->end -= 1;
	       }
            /* store left context value in rule */
             if (substr_length(ssx) > 0)
               {rx->lcontextl = substr_length(ssx);
                free(rx->lcontext);
                rx->lcontext = strdup_safe(substr_to_str(ssx,sx,LINE_LENGTH),proc);
if (db_level > 3) printf("%s lcontext='%s',len=%d\n",pdb,rx->lcontext,rx->lcontextl);
            }  }
          /* get right context */
if (db_level > 3) printf("%s getting right context\n",pdb);
           ssx->end = prtrim(pline);
          ssx->start = islot;
          while ((!isspace(*(ssx->start)))&&(ssx->start < ssx->end)) ssx->start += 1;
          if (ssx->start < ssx->end)
            {while (isspace(*(ssx->start))) ssx->start += 1;
         /* remove boundary characters, if any */
             if (((*ssx->start == '[')&&(*ssx->end == ']')) ||
                 ((*ssx->start == '"')&&(*ssx->end == '"')) )
               {ssx->start += 1;
                ssx->end -= 1;
	       }
            /* store right context value in rule */
             if (substr_length(ssx) > 0)
               {rx->rcontextl = substr_length(ssx);
                free(rx->rcontext);
                rx->rcontext = strdup_safe(substr_to_str(ssx,sx,LINE_LENGTH),proc);
if (db_level > 3) printf("%s rcontext='%s',len=%d\n",pdb,rx->rcontext,rx->rcontextl);
      }  }  }  }

    if (rset->nrules < rset->max_nrules)
      {rset->nrules += 1;
       rset->rule[rset->nrules] = *rx;
      }
    else
      {fprintf(stderr,"*ERR:%s: too many rules.\n",proc);
       fprintf(stderr,"  (max_nrules = %d)\n",rset->max_nrules);
       *perr = 31; goto RETURN;
      }

RETURN:
 db_leave_msg(proc,1); /* debug only */
    return;
   } /* end r_process_data_line */


  /************************************************************/
  /*                                                          */
  /*  get_rules2(rset, path, fname, perr);                    */
  /*                                                          */
  /*  Function #2 to read in a ruleset from a file.           */
  /*  This one uses context-sensitive rules.                  */
  /*  Input: *path+*fname contains the name of the file       */
  /*  Output: *rset is a ruleset structure                    */
  /*          *perr indicates success:                        */
  /*           0 iff a.o.k.                                   */
  /*           2 iff a pcode string had to be truncated.      */
  /*           3 an aux line was bungled.                     */
  /*           4 a data (code) line was bungled.              */
  /*          11 iff file *p1 can't be opened.                */
  /*          12 iff file *p1 is empty                        */
  /*          14 iff get_ppfcn2 returns an error indication.  */
  /*        (from function r_process_data_line:)              */
  /*           1: no valid information in pline               */
  /*          31: memory ovf (too many rules)                 */
  /*          32: no arrow                                    */
  /*   (Fatal error if memory allocation fails.)              */
  /* (Uses global parameter LINE_LENGTH.)                     */
  /*                                                          */
  /************************************************************/
  void get_rules2(RULESET2 *rset, Char *path, Char *fname, int *perr)
   {Char *proc = "get_rules2";
/* data */
    FILE *fp;
    Char line[LINE_LENGTH], *pline = &line[0];
    Char fnxx[LONG_LINE], *full_fname = &fnxx[0];
    Char cmx[LINE_LENGTH], *comment_flag = &cmx[0];
    boolean *indexed;
    int n_data_lines_processed;
    int nrules[257], key_char, key_char2, jind, i, i2;
/* coding  */
 db_enter_msg(proc,1); /* debug only */
 if (db_level > 1) printf("%sfname='%s'\n",pdb,fname);
    *perr = 0;
    n_data_lines_processed = 0;
/* make full name of pcodeset file */
    full_fname = make_full_fname(full_fname,path,fname);
/* expand environment variables in it */
    full_fname = expenv(full_fname,LONG_LINE);
/* initialize  */
    rset->name      = strdup_safe(fname,proc);
    rset->directory = strdup_safe(path,proc);
    rset->desc      = strdup_safe("",proc);
    rset->format    = strdup_safe("NIST1",proc);
    rset->copy_no_hit    = T;
    rset->case_sensitive = T;
    rset->indexed        = F;
    rset->nrules = 0;
    rset->max_nrules = 400;
  /* open rules file */
if (db_level > 1) printf("*DB: trying to open '%s'\n",full_fname);
    if ( (fp = fopen(full_fname,"r")) == NULL)
      {fprintf(stderr,"%s: can't open %s\n",proc,full_fname);
       *perr = 11;  goto RETURN;
      }
  /* read first line to get the comment_flag characters */
    if (fgets(pline,LINE_LENGTH,fp) != NULL)
      {get_comment_flag(pline,comment_flag);
      }
    else
      {fprintf(stderr,"%s: file %s is empty.\n",proc,fname);
       *perr = 12;  goto RETURN;
      }
    if (streq(comment_flag,"*"))
      {fprintf(stderr,"%s: WARNING: the comment_flag character in",proc);
       fprintf(stderr," file %s\n",fname);
       fprintf(stderr,"  is '*', which also marks auxiliary lines.\n");
      }
  /* process file contents */
    while (fgets(pline,LINE_LENGTH,fp) != NULL) /* get next line */
      {pline = prtrim2(del_eol(pline));
if (db_level > 2) printf("%s line:'%s'\n",pdb,pline);
       if (valid_data_line(pline,comment_flag))
         {  /* process a non-blank, non-comment line in s */
          if (line[0] == '*')
            {r_process_aux_line(rset,pline,perr);
             if (*perr > 0) goto RETURN;
	    }
          else
            {if (n_data_lines_processed == 0)
               rset->rule = (RULE2*) calloc_safe((size_t)rset->max_nrules,sizeof(RULE2),proc);
             remove_comments(pline,comment_flag);
             r_process_data_line(rset,pline,perr);
             if (*perr > 0) goto RETURN;
             n_data_lines_processed += 1;
	    }
          if (*perr > 0)
            {fprintf(stderr,"%s: error processing '%s'\n",proc,fname);
             fprintf(stderr," line='%s'\n",pline);
             goto RETURN;
      }  }  }
  /* close file & adjust some variables  */
    if (fclose(fp) != 0)
      fprintf(stderr,"%s: error return from fclose\n",proc);
  /* zero out value for each rule */
    for (i=0; i <= rset->nrules; i++) rset->rule[i].val1 = 0;
  /* make rule indexing tables */
    /* get rules aggregated by key Char (first Char of Source) */
    rset->rule_index = (int*)calloc_safe(rset->nrules+1,sizeof(int),proc);
    rset->first_rule = (int*)calloc_safe(257,sizeof(int),proc);
    rset->last_rule  = (int*)calloc_safe(257,sizeof(int),proc);
    indexed  = (int*)calloc_safe(rset->nrules+1,sizeof(int),proc);
    for (i = 1; i <= rset->nrules; i++) rset->rule_index[i] = i;
    for (key_char=0; key_char < 256; key_char++)
      {rset->first_rule[key_char] = 0;
       rset->last_rule[key_char] = -1;
       nrules[key_char] = 0;
      }
    /* make indices */
if (db_level > 1) printf("%s making indices\n",pdb);
    jind = 0;
    for (i = 1; i <= rset->nrules; i++) indexed[i] = F;
    for (i = 1; i <= rset->nrules; i++)
      {key_char = (int)(rset->rule[i].sin[0]);
       if (!rset->case_sensitive) key_char = toupper(key_char);
if (db_level > 1) printf("%s processing rule %d, key_char = %d (%c)\n",
 pdb,i,key_char,key_char);
       if (nrules[key_char] == 0)
         {/* we haven't handled this key Char yet */
          rset->rule_index[++jind] = i;
          indexed[i] = T;
          nrules[key_char] += 1;
          rset->first_rule[key_char] = jind;
          rset->last_rule[key_char] = jind;
         /* index all later rules with same key Char */
          for (i2 = i+1; i2 <= rset->nrules; i2++)
            {if (!indexed[i2])
               {key_char2 = (int)(rset->rule[i2].sin[0]);
                if (!rset->case_sensitive) key_char2 = toupper(key_char2);
                if (key_char2 == key_char)
                  {rset->rule_index[++jind] = i2;
                   indexed[i2] = T;
                   nrules[key_char] += 1;
                   rset->last_rule[key_char] = jind;
      }  }  }  }  }
    rset->indexed = T;
    free(indexed);

RETURN:
 db_leave_msg(proc,1); /* debug only */
    return;
  } /* end get_rules2 */

/* end include grules2.c */
