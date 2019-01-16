#include "sctk.h"


static void locate_insertions(SCORES *scor,int spkr,AUTO_LEX *talex);
static void locate_deletions(SCORES *scor,int spkr,AUTO_LEX *talex);
static void locate_confusion_pairs(SCORES *scor,int spkr,AUTO_LEX *talex);
static void locate_substitutions(SCORES *scor,int spkr,AUTO_LEX *talex);
static void locate_misrecognized(SCORES *scor,int spkr,AUTO_LEX *talex);


int score_dtl_sent(SCORES *scor, char *sys_root_name, int feedback){
    int spkr, p;
    FILE *fp;
    char *fname;

    for (spkr=0; spkr<scor->num_grp; spkr++){
	int sp_c, sp_s, sp_n, sp_d, sp_u, sp_r, sp_nw;

	/* initialize variables */
	sp_c = sp_s = sp_n = sp_d = sp_u = sp_r = sp_nw = 0;

	/* open an output file, if it fails, write to stdout */
	if (strcmp(sys_root_name,"-") == 0 ||
	    (fp = fopen(fname=rsprintf("%s.snt.%s",sys_root_name,
				       scor->grp[spkr].name),"w")) == NULL)
	    fp = stdout;
	else
	    if (feedback >= 1)
		printf("    Writing sentence scoring report for speaker '%s' to '%s'\n",
		       scor->grp[spkr].name, fname);

	fprintf(fp,"===============================================================================\n");
	fprintf(fp,"\n");
	fprintf(fp,"                     SENTENCE LEVEL REPORT FOR THE SYSTEM:\n");
	fprintf(fp,"    Name: %s\n",scor->title);
	fprintf(fp,"\n");
	fprintf(fp,"===============================================================================\n");
	fprintf(fp,"\n");
	fprintf(fp,"\n");
	fprintf(fp,"SPEAKER %s\n",scor->grp[spkr].name);

	for (p=0; p<scor->grp[spkr].num_path; p++){
	    int w, c, s, n, d, u, r, nw;
	    PATH *path=scor->grp[spkr].path[p];

	    PATH_print(path,fp,120);
	    fprintf(fp,"\n");
	   
	    for (w=c=s=d=n=u=0; w<path->num; w++)
		if ((path->pset[w].eval & P_CORR) != 0) c++;
		else if ((path->pset[w].eval & P_SUB) != 0) s++;
		else if ((path->pset[w].eval & P_INS) != 0) n++;
		else if ((path->pset[w].eval & P_DEL) != 0) d++;
		else u ++;
	    r = c + s + d; 
	    nw = path->num;

	    sp_c += c;
	    sp_s += s;
	    sp_d += d;
	    sp_n += n;
	    sp_u += u;
	    sp_r += r;
	    sp_nw += nw;


	    /* write out the information */
	    if (r > 0){
	      fprintf(fp,"Correct               =  %5.1f%%   %2d   (%3d)\n",
		      F_ROUND(pct(c,r),1),c,sp_c);
	      fprintf(fp,"Substitutions         =  %5.1f%%   %2d   (%3d)\n",
		      F_ROUND(pct(s,r),1),s,sp_s);
	      fprintf(fp,"Deletions             =  %5.1f%%   %2d   (%3d)\n",
		      F_ROUND(pct(d,r),1),d,sp_d);
	      fprintf(fp,"Insertions            =  %5.1f%%   %2d   (%3d)\n",
		      F_ROUND(pct(n,r),1),n,sp_n);
	      if (sp_u > 0)
		fprintf(fp,"UNKNOWN               =  %5.1f%%   %2d   (%3d)\n",
			F_ROUND(pct(u,r),1),u,sp_u);
	      fprintf(fp,"\n");
	      fprintf(fp,"Errors                =  %5.1f%%   %2d   (%3d)\n",
		      F_ROUND(pct((s+d+n),r),1),s+d+n,sp_s+sp_d+sp_n);
	      fprintf(fp,"\n");
	    } else {
	      fprintf(fp,"Correct               =  UNDEF%%   %2d   (%3d)\n",
		      c,sp_c);
	      fprintf(fp,"Substitutions         =  UNDEF%%   %2d   (%3d)\n",
		      s,sp_s);
	      fprintf(fp,"Deletions             =  UNDEF%%   %2d   (%3d)\n",
		      d,sp_d);
	      fprintf(fp,"Insertions            =  UNDEF%%   %2d   (%3d)\n",
		      n,sp_n);
	      if (sp_u > 0)
		fprintf(fp,"UNKNOWN               =  UNDEF%%   %2d   (%3d)\n",
			u,sp_u);
	      fprintf(fp,"\n");
	      fprintf(fp,"Errors                =  UNDEF%%   %2d   (%3d)\n",
		      s+d+n,sp_s+sp_d+sp_n);
	      fprintf(fp,"\n");
	    }

	    fprintf(fp,"Ref. words            =           %2d   (%3d)\n",
		    r, sp_r);
	    fprintf(fp,"Hyp. words            =           %2d   (%3d)\n",
		    c+s+n, sp_c+sp_s+sp_n);
	    fprintf(fp,"Aligned words         =           %2d   (%3d)\n",
		    nw, sp_nw);

	    fprintf(fp,"\n");

	    fprintf(fp,"------------------------------------------------"
		    "-------------------------------\n");
	    fprintf(fp,"\n");
	}
	if (fp != stdout) fclose(fp);
    }
    return(0);
}

int score_dtl_spkr(SCORES *scor, char *sys_root_name, int feedback){
    int spkr, p;
    FILE *fp;
    char *fname;
    int sp_c, sp_s, sp_n, sp_d, sp_u, sp_r, sp_nw;
    AUTO_LEX talex;

    for (spkr=0; spkr<scor->num_grp; spkr++){
	int sent_wsub, sent_wdel, sent_wins, r, e;

	/* initialize variables */
	sp_c = sp_s = sp_n = sp_d = sp_u = sp_r = sp_nw = 0;
	sent_wsub = sent_wdel = sent_wins = 0;

	for (p=0; p<scor->grp[spkr].num_path; p++){
	    int w, c, s, n, d, u;
	    PATH *path=scor->grp[spkr].path[p];

	    for (w=c=s=d=n=u=0; w<path->num; w++)
		if ((path->pset[w].eval & P_CORR) != 0) c++;
		else if ((path->pset[w].eval & P_SUB) != 0) s++;
		else if ((path->pset[w].eval & P_INS) != 0) n++;
		else if ((path->pset[w].eval & P_DEL) != 0) d++;
		else u ++;
	   if (s > 0) sent_wsub++;
	   if (n > 0) sent_wins++;
	   if (d > 0) sent_wdel++;
	}

	/* open an output file, if it fails, write to stdout */
	if (strcmp(sys_root_name,"-") == 0 ||
	    (fp = fopen(fname=rsprintf("%s.spk.%s",sys_root_name,
				       scor->grp[spkr].name),"w")) == NULL)
	    fp = stdout;
	else
	    if (feedback >= 1)
		printf("    Writing speaker scoring report for '%s' to '%s'\n",
		       scor->grp[spkr].name, fname);

	fprintf(fp,"SCORING FOR SPEAKER: %s\n",scor->grp[spkr].name);
	fprintf(fp,"     of %s\n",scor->title);
	fprintf(fp,"\n");
        fprintf(fp,"SENTENCE RECOGNITION PERFORMANCE\n");
	fprintf(fp,"\n");
	fprintf(fp," sentences                                        %4d\n",
		scor->grp[spkr].nsent);
	fprintf(fp," with errors                            %5.1f%%   (%4d)\n\n",
		F_ROUND(pct(scor->grp[spkr].serr,scor->grp[spkr].nsent),1),
		scor->grp[spkr].serr);
	fprintf(fp,"   with substitions                     %5.1f%%   (%4d)\n",
		F_ROUND(pct(sent_wsub,scor->grp[spkr].nsent),1),sent_wsub);
	fprintf(fp,"   with deletions                       %5.1f%%   (%4d)\n",
		F_ROUND(pct(sent_wdel,scor->grp[spkr].nsent),1),sent_wdel);
	fprintf(fp,"   with insertions                      %5.1f%%   (%4d)\n",
 		F_ROUND(pct(sent_wins,scor->grp[spkr].nsent),1),sent_wins);

	fprintf(fp,"\n");
	fprintf(fp,"\n");

	fprintf(fp,"WORD RECOGNITION PERFORMANCE\n");
	r = scor->grp[spkr].corr + scor->grp[spkr].sub + scor->grp[spkr].del;
	e = scor->grp[spkr].sub + scor->grp[spkr].del + scor->grp[spkr].ins;

	fprintf(fp,"\n");
	if (r > 0){
	  fprintf(fp,"Percent Total Error       =  %5.1f%%   (%4d)\n",
		  F_ROUND(pct(e,r),1), e);
	  fprintf(fp,"\n");
	  
	  fprintf(fp,"Percent Correct           =  %5.1f%%   (%4d)\n",
		  F_ROUND(pct(scor->grp[spkr].corr,r),1),scor->grp[spkr].corr);
	  fprintf(fp,"\n");
	  fprintf(fp,"Percent Substitution      =  %5.1f%%   (%4d)\n",
		  F_ROUND(pct(scor->grp[spkr].sub,r),1),scor->grp[spkr].sub);
	  fprintf(fp,"Percent Deletions         =  %5.1f%%   (%4d)\n",
		  F_ROUND(pct(scor->grp[spkr].del,r),1),scor->grp[spkr].del);
	  fprintf(fp,"Percent Insertions        =  %5.1f%%   (%4d)\n",
		  F_ROUND(pct(scor->grp[spkr].ins,r),1),scor->grp[spkr].ins);
	  
	  fprintf(fp,"Percent Word Accuracy     =  %5.1f%%\n",
		  F_ROUND(100.0 -  pct(e , r),1));
	} else {
	  fprintf(fp,"Percent Total Error       =  UNDEF%%   (%4d)\n",e);
	  fprintf(fp,"\n");
	  
	  fprintf(fp,"Percent Correct           =  UNDEF%%   (%4d)\n",
		  scor->grp[spkr].corr);
	  fprintf(fp,"\n");
	  fprintf(fp,"Percent Substitution      =  UNDEF%%   (%4d)\n",
		  scor->grp[spkr].sub);
	  fprintf(fp,"Percent Deletions         =  UNDEF%%   (%4d)\n",
		  scor->grp[spkr].del);
	  fprintf(fp,"Percent Insertions        =  UNDEF%%   (%4d)\n",
		  scor->grp[spkr].ins);
	  
	  fprintf(fp,"Percent Word Accuracy     =  UNDEF%%\n");
	}
	fprintf(fp,"\n");
	fprintf(fp,"\n");

	fprintf(fp,"Ref. words                =           (%4d)\n",r);
	fprintf(fp,"Hyp. words                =           (%4d)\n",
		scor->grp[spkr].corr + scor->grp[spkr].sub +
		scor->grp[spkr].ins);
	fprintf(fp,"Aligned words             =           (%4d)\n",
		scor->grp[spkr].corr + scor->grp[spkr].sub +
		scor->grp[spkr].ins + scor->grp[spkr].del);
 
	AUTO_LEX_init(&talex, 200);
	locate_confusion_pairs(scor,spkr,&talex);
	fprintf(fp,"\n");
	AUTO_LEX_printout(&talex, fp, "CONFUSION PAIRS", 1);
	AUTO_LEX_free(&talex);
	fprintf(fp,"\n");
	
	AUTO_LEX_init(&talex, 200);
	locate_insertions(scor,spkr,&talex);
	fprintf(fp,"\n");
	AUTO_LEX_printout(&talex, fp, "INSERTIONS", 1);
	AUTO_LEX_free(&talex);
	fprintf(fp,"\n");
	
	AUTO_LEX_init(&talex, 200);
	locate_deletions(scor,spkr,&talex);
	fprintf(fp,"\n");
	AUTO_LEX_printout(&talex, fp, "DELETIONS", 1);
	AUTO_LEX_free(&talex);

	fprintf(fp,"\n");
	
	AUTO_LEX_init(&talex, 200);
	locate_substitutions(scor,spkr,&talex);
	fprintf(fp,"\n");
	AUTO_LEX_printout(&talex, fp, "SUBSTITUTIONS", 1);
	AUTO_LEX_free(&talex);
	fprintf(fp,"\n");
	fprintf(fp,"* NOTE: The 'Substitution' words are those reference words\n");
	fprintf(fp,"        for which the recognizer supplied an incorrect word.\n");
	fprintf(fp,"\n");
		
	AUTO_LEX_init(&talex, 200);
	locate_misrecognized(scor,spkr,&talex);
	fprintf(fp,"\n");
	AUTO_LEX_printout(&talex, fp, "FALSELY RECOGNIZED", 1);
	AUTO_LEX_free(&talex);
	fprintf(fp,"\n");
	fprintf(fp,"* NOTE: The 'Falsely Recognized' words are those hypothesis words\n");
	fprintf(fp,"        which the recognizer incorrectly substituted for a reference word.\n");
	fprintf(fp,"\n");

	if (fp != stdout) fclose(fp);
    }
    return(0);
}

int score_dtl_overall(SCORES *scor, char *sys_root_name, int feedback){
    int spkr, p;
    FILE *fp;
    char *fname;
    int sp_c, sp_s, sp_n, sp_d, sp_u, sp_r, sp_nw, sp_nsent, sp_nserr;
    AUTO_LEX talex;
    int sent_wsub, sent_wdel, sent_wins, r, e;

    /* initialize variables */
    sp_c = sp_s = sp_n = sp_d = sp_u = sp_r = sp_nw = sp_nsent = sp_nserr = 0;
    sent_wsub = sent_wdel = sent_wins = 0;

    for (spkr=0; spkr<scor->num_grp; spkr++){
	for (p=0; p<scor->grp[spkr].num_path; p++){
	    int w, c, s, n, d, u;
	    PATH *path=scor->grp[spkr].path[p];
	    
	    for (w=c=s=d=n=u=0; w<path->num; w++)
		if ((path->pset[w].eval & P_CORR) != 0) c++;
		else if ((path->pset[w].eval & P_SUB) != 0) s++;
		else if ((path->pset[w].eval & P_INS) != 0) n++;
		else if ((path->pset[w].eval & P_DEL) != 0) d++;
		else u ++;
	    if (s > 0) sent_wsub++;
	    if (n > 0) sent_wins++;
	    if (d > 0) sent_wdel++;
	}
	sp_c += scor->grp[spkr].corr;
	sp_d += scor->grp[spkr].del;
	sp_n += scor->grp[spkr].ins;
	sp_s += scor->grp[spkr].sub;
	sp_nsent += scor->grp[spkr].nsent;
	sp_nserr += scor->grp[spkr].serr;
    }

    /* open an output file, if it fails, write to stdout */
    if (strcmp(sys_root_name,"-") == 0 ||
	(fp = fopen(fname=rsprintf("%s.dtl",sys_root_name),"w")) == NULL)
	fp = stdout;
    else
	if (feedback >= 1)
	    printf("    Writing overall detailed scoring report '%s'\n",
		   fname);
    
    fprintf(fp,"DETAILED OVERALL REPORT FOR THE SYSTEM: %s\n",scor->title);
    fprintf(fp,"\n");
    fprintf(fp,"SENTENCE RECOGNITION PERFORMANCE\n");
    fprintf(fp,"\n");
    fprintf(fp," sentences                                        %4d\n",
	    sp_nsent);
    fprintf(fp," with errors                            %5.1f%%   (%4d)\n\n",
	    F_ROUND(pct(sp_nserr,sp_nsent),1), sp_nserr);
    fprintf(fp,"   with substitions                     %5.1f%%   (%4d)\n",
	    F_ROUND(pct(sent_wsub,sp_nsent),1),sent_wsub);
    fprintf(fp,"   with deletions                       %5.1f%%   (%4d)\n",
	    F_ROUND(pct(sent_wdel,sp_nsent),1),sent_wdel);
    fprintf(fp,"   with insertions                      %5.1f%%   (%4d)\n",
	    F_ROUND(pct(sent_wins,sp_nsent),1),sent_wins);
    
    fprintf(fp,"\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"WORD RECOGNITION PERFORMANCE\n");
    r = sp_c + sp_s + sp_d;
    e = sp_s + sp_d + sp_n;

    fprintf(fp,"\n");
    if (r > 0){
      fprintf(fp,"Percent Total Error       =  %5.1f%%   (%4d)\n",
	      F_ROUND(pct(e,r),1), e);
      fprintf(fp,"\n");
      
      fprintf(fp,"Percent Correct           =  %5.1f%%   (%4d)\n",
	    F_ROUND(pct(sp_c,r),1),sp_c);
      fprintf(fp,"\n");
      fprintf(fp,"Percent Substitution      =  %5.1f%%   (%4d)\n",
	      F_ROUND(pct(sp_s,r),1),sp_s);
      fprintf(fp,"Percent Deletions         =  %5.1f%%   (%4d)\n",
	      F_ROUND(pct(sp_d,r),1),sp_d);
      fprintf(fp,"Percent Insertions        =  %5.1f%%   (%4d)\n",
	      F_ROUND(pct(sp_n,r),1),sp_n);
      
      fprintf(fp,"Percent Word Accuracy     =  %5.1f%%\n",
	      F_ROUND(100.0 -  pct(e , r),1));
    } else {
      fprintf(fp,"Percent Total Error       =  UNDEF%%   (%4d)\n",e);
      fprintf(fp,"\n");
      
      fprintf(fp,"Percent Correct           =  UNDEF%%   (%4d)\n",sp_c);
      fprintf(fp,"\n");
      fprintf(fp,"Percent Substitution      =  UNDEF%%   (%4d)\n",sp_s);
      fprintf(fp,"Percent Deletions         =  UNDEF%%   (%4d)\n",sp_d);
      fprintf(fp,"Percent Insertions        =  UNDEF%%   (%4d)\n",sp_n);
      
      fprintf(fp,"Percent Word Accuracy     =  UNDEF%%\n");
    }
    
    fprintf(fp,"\n");
    fprintf(fp,"\n");
    
    fprintf(fp,"Ref. words                =           (%4d)\n",r);
    fprintf(fp,"Hyp. words                =           (%4d)\n",
	    sp_c + sp_s + sp_n);
    fprintf(fp,"Aligned words             =           (%4d)\n",
	    sp_c + sp_s + sp_n + sp_d);
    
    AUTO_LEX_init(&talex, 200);
    for (spkr=0; spkr<scor->num_grp; spkr++)
	locate_confusion_pairs(scor,spkr,&talex);
    fprintf(fp,"\n");
    AUTO_LEX_printout(&talex, fp, "CONFUSION PAIRS", 1);
    AUTO_LEX_free(&talex);
    fprintf(fp,"\n");
    
    AUTO_LEX_init(&talex, 200);
    for (spkr=0; spkr<scor->num_grp; spkr++)
	locate_insertions(scor,spkr,&talex);
    fprintf(fp,"\n");
    AUTO_LEX_printout(&talex, fp, "INSERTIONS", 1);
    AUTO_LEX_free(&talex);
    fprintf(fp,"\n");
    
    AUTO_LEX_init(&talex, 200);
    for (spkr=0; spkr<scor->num_grp; spkr++)
	locate_deletions(scor,spkr,&talex);
    fprintf(fp,"\n");
    AUTO_LEX_printout(&talex, fp, "DELETIONS", 1);
    AUTO_LEX_free(&talex);
    
    fprintf(fp,"\n");
    
    AUTO_LEX_init(&talex, 200);
    for (spkr=0; spkr<scor->num_grp; spkr++)
	locate_substitutions(scor,spkr,&talex);
    fprintf(fp,"\n");
    AUTO_LEX_printout(&talex, fp, "SUBSTITUTIONS", 1);
    AUTO_LEX_free(&talex);
    fprintf(fp,"\n");
    fprintf(fp,"* NOTE: The 'Substitution' words are those reference words\n");
    fprintf(fp,"        for which the recognizer supplied an incorrect word.\n");
    fprintf(fp,"\n");
    
    AUTO_LEX_init(&talex, 200);
    for (spkr=0; spkr<scor->num_grp; spkr++)
	locate_misrecognized(scor,spkr,&talex);
    fprintf(fp,"\n");
    AUTO_LEX_printout(&talex, fp, "FALSELY RECOGNIZED", 1);
    AUTO_LEX_free(&talex);
    fprintf(fp,"\n");
    fprintf(fp,"* NOTE: The 'Falsely Recognized' words are those hypothesis words\n");
    fprintf(fp,"        which the recognizer incorrectly substituted for a reference word.\n");
    fprintf(fp,"\n");

    if (fp != stdout) fclose(fp);

    return(0);
}


static void locate_insertions(SCORES *scor,int spkr,AUTO_LEX *talex){
    int p, w, lid;    

    for (p=0; p<scor->grp[spkr].num_path; p++){
	PATH *path=scor->grp[spkr].path[p];
	for (w=0; w<path->num; w++){
	    if ((path->pset[w].eval & P_INS) != 0){
		lid = AUTO_LEX_insert(talex,((WORD *)(path->pset[w].b_ptr))->value);
		talex->field_a[lid] ++;
	    }
	}
    }
}

static void locate_deletions(SCORES *scor,int spkr,AUTO_LEX *talex){
    int p, w, lid;    

    for (p=0; p<scor->grp[spkr].num_path; p++){
	PATH *path=scor->grp[spkr].path[p];
	for (w=0; w<path->num; w++){
	    if ((path->pset[w].eval & P_DEL) != 0){
		lid = AUTO_LEX_insert(talex,((WORD *)(path->pset[w].a_ptr))->value);
		talex->field_a[lid] ++;
	    }
	}
    }
}

static void locate_confusion_pairs(SCORES *scor,int spkr,AUTO_LEX *talex){
    int p, w, lid;    

    for (p=0; p<scor->grp[spkr].num_path; p++){
	PATH *path=scor->grp[spkr].path[p];
	for (w=0; w<path->num; w++){
	    if ((path->pset[w].eval & P_SUB) != 0){
		lid = AUTO_LEX_insert(talex,
		      (TEXT *)rsprintf("%s ==> %s",
				       ((WORD *)(path->pset[w].a_ptr))->value,
				       ((WORD *)(path->pset[w].b_ptr))->value));
		talex->field_a[lid] ++;
	    }
	}
    }
}

static void locate_substitutions(SCORES *scor,int spkr,AUTO_LEX *talex){
    int p, w, lid;    

    for (p=0; p<scor->grp[spkr].num_path; p++){
	PATH *path=scor->grp[spkr].path[p];
	for (w=0; w<path->num; w++){
	    if ((path->pset[w].eval & P_SUB) != 0){
		lid = AUTO_LEX_insert(talex, ((WORD *)(path->pset[w].a_ptr))->value);
		talex->field_a[lid] ++;
	    }
	}
    }

}

static void locate_misrecognized(SCORES *scor,int spkr,AUTO_LEX *talex){
    int p, w, lid;    

    for (p=0; p<scor->grp[spkr].num_path; p++){
	PATH *path=scor->grp[spkr].path[p];
	for (w=0; w<path->num; w++){
	    if ((path->pset[w].eval & P_SUB) != 0){
		lid = AUTO_LEX_insert(talex, ((WORD *)(path->pset[w].b_ptr))->value);
		talex->field_a[lid] ++;
	    }
	}
    }

}

