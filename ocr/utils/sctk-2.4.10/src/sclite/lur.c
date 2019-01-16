#include "sctk.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct label_stores_struct{
    int nref;              /* number of reference words */
    int nerr;              /* number of errors */
} LABEL_SCORES;

typedef struct label_struct{
    int *location;         /* array of categories for a label */
    LABEL_SCORES ***lsc;   /* indexed by sys, label, spkrs */
} LABEL;

#ifdef __cplusplus
}
#endif

static LABEL *alloc_LABEL_SCORES(int nscor, SCORES *sc);
static void compute_label_scores(SCORES *sc,int scnum, LABEL *lab);
static void create_report(LABEL *label, SCORES *sc, FILE *fp);
static void create_N_lur_report(LABEL *label, SCORES *sc[], int nsc, char *tname, FILE *fp);
static void free_LABEL_SCORES(LABEL *tlab, SCORES *sc, int nscor);

static LABEL *alloc_LABEL_SCORES(int nscor, SCORES *sc)
{
    LABEL *tlab;

    int i,j,k;
    alloc_singarr(tlab,1,LABEL);
    alloc_3dimarr(tlab->lsc,nscor,sc->aset.num_plab,
		  sc->num_grp,LABEL_SCORES);
    alloc_singZ(tlab->location,sc->aset.num_plab,int,(-1));
    for (i=0; i<nscor; i++)
	for (j=0; j<sc->aset.num_plab; j++)
	    for (k=0; k<sc->num_grp; k++)
		tlab->lsc[i][j][k].nref = tlab->lsc[i][j][k].nerr = 0;
    return(tlab);
}

static void free_LABEL_SCORES(LABEL *tlab, SCORES *sc, int nscor)
{
    free_singarr(tlab->location,int);
    free_3dimarr(tlab->lsc,nscor,sc->aset.num_plab,LABEL_SCORES);
    free_singarr(tlab,LABEL);
}

static void compute_label_scores(SCORES *sc,int scnum, LABEL *lab){
    int i,p,ind,ms=0,sets,id;
    char *cp,hs[30];
    PATH *path=(PATH *)0;

    for (i=0; i<sc->num_grp; i++){
	for (p=0; p<sc->grp[i].num_path; p++){
	    path = sc->grp[i].path[p];
	    if (path->labels != (char *)0){
		/* fprintf(stdout,"Labels: %s\n",path->labels); */
		/* parse the labels */
		sets = 0;
		cp = path->labels;
		/* Skip the first separator */
		if (*(cp++) != '<') { ms = 1; goto FAILED; }
		while (*cp != '\0' && *cp != '>'){
		    /* look for the end, if *cp == '<' or ',' */
		    if (*cp != '<' && *cp != ','){
			if ((ind = strcspn(cp,"<,>")) < 1) {ms=1;goto FAILED;};
		    } else
			ind=0;
		    if (ind != 0){
			strncpy(hs,cp,ind);   hs[ind] = '\0';
			/* printf("    set %s\n",hs);*/
			/* Lookup the set in the label structure */
                        if ((id = find_PATHLABEL_id(sc,hs)) == -1){
			    ms=2; goto FAILED;}
			/* printf("    id = %d\n",id); */
			if (lab->location[id] == -1)
			    lab->location[id] = sets;
			else if (lab->location[id] != sets){
			    ms=3; goto FAILED;}

			/* add the scores into the LABEL structure  */
			{ int w,c=0,s=0,n=0,d=0,u=0;
			  for (w=0; w<path->num; w++)
			      if ((path->pset[w].eval & P_CORR) != 0) c++;
			      else if ((path->pset[w].eval & P_SUB) != 0) s++;
			      else if ((path->pset[w].eval & P_INS) != 0) n++;
			      else if ((path->pset[w].eval & P_DEL) != 0) d++;
			      else u ++;

			  lab->lsc[scnum][id][i].nref += c + d + s;
			  lab->lsc[scnum][id][i].nerr += n + d + s;
		        }
		        cp += ind+1;
		    } else
		        cp ++;
		    sets ++;
		}
	    }
	}
    }

    return;

  FAILED:
    switch (ms){
      case 1: fprintf(scfp,"Error: Bad Label '%s' for utt %s\n"
		      "       Make sure utterance labels have been properly"
		      " formatted in the reference file.\n",
		      path->labels,path->id);
	break;
      case 2: fprintf(scfp,"Error: Label id '%s' lookup failed for utt %s\n"
		      "       Make sure labels have been defined in the"
		      " comment lines of the reference file.\n",
		      hs,path->id);
	break;
      case 3: fprintf(scfp,"Error: Label id '%s' of utt '%s' (label def %s)"
		      " occurs in multiple label category fields.\n",
		      hs,path->id,path->labels);

	break;
      default:
	;
    }
    exit(1);
}

static void create_report(LABEL *labels, SCORES *sc, FILE *fp)
{ 
    int j, k, npres, sumr, sume, i;
    int null_spkr_ref=0;
    char titlefmt[50],fmt[50],categfmt[50];
    double *ref_words, *err_words;
    double **ref_sts, **err_sts;
    
    alloc_singarr(ref_words,sc->num_grp,double);
    alloc_singarr(err_words,sc->num_grp,double);
    alloc_2dimarr(ref_sts,sc->aset.num_plab,6,double);
    alloc_2dimarr(err_sts,sc->aset.num_plab,6,double);
    
    /* create the format statements */
    strcpy(titlefmt,"c");
    strcpy(fmt,"c");
    strcpy(categfmt,"c");
    for (j=0; j<sc->aset.num_plab; j++){
	if (j > 0 && labels->location[j] != labels->location[j-1]){
	    strcat(titlefmt,"=");
	    strcat(fmt,"=");
	    strcat(categfmt,"=");
	} else {
	    strcat(titlefmt,"|");
	    strcat(fmt,"|");
	    strcat(categfmt,"|");
	}
	strcat(titlefmt,"ca");
	strcat(fmt,"rr");
	if ((j == 0) ||
	    (j > 0 && labels->location[j] != labels->location[j-1]))
	    strcat(categfmt,"ca");
	else
	    strcat(categfmt,"aa");
    }
    Desc_erase();
    /* add the label set here for now */
    Desc_add_row_values("c",rsprintf("System: %s",sc->title));
    Desc_add_row_separation(' ', BEFORE_ROW);
    for (j=0; j<sc->aset.num_plab; j++)
	Desc_add_row_values("ccc",Desc_rm_lf(sc->aset.plab[j].title),"->",
			    sc->aset.plab[j].desc);
    Desc_add_row_separation(' ', AFTER_ROW);
    Desc_add_row_values("ccc","","","");
    Desc_add_row_separation('-', BEFORE_ROW);

    /* sub category titles */
    if (sc->aset.num_cat > 0){
	Desc_set_iterated_format(categfmt);
	Desc_set_iterated_value("");
	for (j=0; j<sc->aset.num_cat; j++)
	    Desc_set_iterated_value(sc->aset.cat[j].title);
	Desc_flush_iterated_row();
	Desc_add_row_separation('-', BEFORE_ROW);
    }

    /* sub category distinctions */
    Desc_set_iterated_format(titlefmt);
    Desc_set_iterated_value("SPKR");
    for (j=0; j<sc->aset.num_plab; j++)
	Desc_set_iterated_value(sc->aset.plab[j].title);
    Desc_flush_iterated_row();
    
    Desc_set_iterated_format(titlefmt);
    Desc_set_iterated_value("");
    for (j=0; j<sc->aset.num_plab; j++)
	Desc_set_iterated_value(" #Wrd %WE");
    Desc_flush_iterated_row();
    
    for (k=0; k<sc->num_grp; k++){
	Desc_add_row_separation('-', BEFORE_ROW);
	Desc_set_iterated_format(fmt);
	Desc_set_iterated_value(sc->grp[k].name);
	for (j=0; j<sc->aset.num_plab; j++){
	    if (labels->lsc[0][j][k].nref != 0){
		Desc_set_iterated_value(rsprintf("[%d]",
						 labels->lsc[0][j][k].nref));
		Desc_set_iterated_value(rsprintf("%.1f",
				 F_ROUND(pct(labels->lsc[0][j][k].nerr,
					     labels->lsc[0][j][k].nref),1)));
	    } else if (labels->lsc[0][j][k].nerr != 0){
		Desc_set_iterated_value(rsprintf("%d",
						 labels->lsc[0][j][k].nref));
		Desc_set_iterated_value(rsprintf("* %d *",
						 labels->lsc[0][j][k].nerr));
		null_spkr_ref = 1;
	    } else {
		Desc_set_iterated_value("");
		Desc_set_iterated_value("");	
	    }
	}
	Desc_flush_iterated_row();
    }

    /* the totals */
    for (j=0; j<sc->aset.num_plab; j++){
	npres = sumr = sume = 0;
	for (k=0; k<sc->num_grp; k++){
	    if (labels->lsc[0][j][k].nref > 0){
		ref_words[npres] = labels->lsc[0][j][k].nref;
		err_words[npres] = pct(labels->lsc[0][j][k].nerr,
				       labels->lsc[0][j][k].nref);
		npres++;
	    }
	    sumr += labels->lsc[0][j][k].nref;
	    sume += labels->lsc[0][j][k].nerr;
	}
	/* calc and store statistics */
	calc_mean_var_std_dev_Zstat_double(ref_words,npres,ref_sts[j]+1,
					   &(ref_sts[j][4]),&(ref_sts[j][2]),
					   &(ref_sts[j][3]),&(ref_sts[j][5]));
	calc_mean_var_std_dev_Zstat_double(err_words,npres,err_sts[j]+1,
					   &(err_sts[j][4]),&(err_sts[j][2]),
					   &(err_sts[j][3]),&(err_sts[j][5]));
	ref_sts[j][0] = sumr;
	err_sts[j][0] = F_ROUND(pct(sume,sumr),1);
    }

    Desc_add_row_separation('=', BEFORE_ROW);
    for (i=0; i<4; i++) {
	char *name="";
	switch (i){
	  case 0: name="Set Sum/Avg"; break;
	  case 1: name=(null_spkr_ref) ? "Mean +" : "Mean";
	    Desc_add_row_separation('-',BEFORE_ROW); break;
	  case 2: name=(null_spkr_ref) ? "StdDev +" : "StdDev"; break;
	  case 3: name="Median";Desc_add_row_separation(' ',BEFORE_ROW); break;
	  default:
	    ;
	}
	Desc_set_iterated_format(fmt);
	Desc_set_iterated_value(name);
	for (j=0; j<sc->aset.num_plab; j++){
	    Desc_set_iterated_value(rsprintf("[%d]",(int)ref_sts[j][i]));
	    Desc_set_iterated_value(rsprintf("%.1f",F_ROUND(err_sts[j][i],1)));
	}
	Desc_flush_iterated_row();
    }

    if (null_spkr_ref){
	Desc_add_row_separation('-',BEFORE_ROW);
	Desc_add_row_values("l","Note:  * Speaker Has no reference word tokens.  Number of incorrect words presented");
	Desc_add_row_values("l","       + Speakers with no reference words ignored");
    }
	
    Desc_dump_report(1,fp);
    free_singarr(ref_words,double);
    free_singarr(err_words,double);
    free_2dimarr(ref_sts,sc->aset.num_plab,double);
    free_2dimarr(err_sts,sc->aset.num_plab,double);
}

static void create_N_lur_report(LABEL *labels, SCORES *sc[], int nsc, char *test_name, FILE *fp)
{ 
    int j, k, npres, sumr, sume, i, s;
    char titlefmt[50],fmt[50],categfmt[50],statfmt[50];
    double *ref_words, *err_words;
    double ***ref_sts, ***err_sts;
    
    alloc_singarr(ref_words,sc[0]->num_grp,double);
    alloc_singarr(err_words,sc[0]->num_grp,double);
    alloc_3dimarr(ref_sts,nsc,sc[0]->aset.num_plab,6,double);
    alloc_3dimarr(err_sts,nsc,sc[0]->aset.num_plab,6,double);
    
    /* create the format statements */
    strcpy(titlefmt,"c");
    strcpy(fmt,"c");
    strcpy(categfmt,"c");
    strcpy(statfmt,"c|");
    for (j=0; j<sc[0]->aset.num_plab; j++){
	if (j > 0 && labels->location[j] != labels->location[j-1]){
	    strcat(titlefmt,"=");
	    strcat(fmt,"=");
	    strcat(categfmt,"=");
	} else {
	    strcat(titlefmt,"|");
	    strcat(fmt,"|");
	    strcat(categfmt,"|");
	}
	strcat(titlefmt,"ca");
	strcat(fmt,"rr");
	if ((j == 0) ||
	    (j > 0 && labels->location[j] != labels->location[j-1]))
	    strcat(categfmt,"ca");
	else
	    strcat(categfmt,"aa");
	strcat(statfmt,(j == 0) ? "ca" : "aa");
    }
    Desc_erase();

    Desc_add_row_values("c","By System Test Subset Scoring Summary");
    Desc_add_row_values("c",rsprintf("For the %s Test",test_name));
    Desc_add_row_separation(' ', BEFORE_ROW);

    /* add the label set here for now */
    for (j=0; j<sc[0]->aset.num_plab; j++)	
	Desc_add_row_values("ccc",Desc_rm_lf(sc[0]->aset.plab[j].title),"->",
			    sc[0]->aset.plab[j].desc);
    Desc_add_row_separation(' ', AFTER_ROW);
    Desc_add_row_values("ccc","","","");
    Desc_add_row_separation('-', BEFORE_ROW);

    /* sub category titles */
    if (sc[0]->aset.num_cat > 0){
	Desc_set_iterated_format(categfmt);
	Desc_set_iterated_value("");
	for (j=0; j<sc[0]->aset.num_cat; j++)
	    Desc_set_iterated_value(sc[0]->aset.cat[j].title);
	Desc_flush_iterated_row();
	Desc_add_row_separation('-', BEFORE_ROW);
    }

    /* sub category distinctions */
    Desc_set_iterated_format(titlefmt);
    Desc_set_iterated_value("SYSTEM");
    for (j=0; j<sc[0]->aset.num_plab; j++)
	Desc_set_iterated_value(sc[0]->aset.plab[j].title);
    Desc_flush_iterated_row();
    
    Desc_set_iterated_format(titlefmt);
    Desc_set_iterated_value("");
    for (j=0; j<sc[0]->aset.num_plab; j++)
	Desc_set_iterated_value(" #Wrd %WE");
    Desc_flush_iterated_row();

    /* ref_sts[s][0][0]  =  system s, plab 0  - Global average */
    /* ref_sts[s][0][1]  =  system s, plab 0  - Global mean */
    /* ref_sts[s][0][2]  =  system s, plab 0  - Global StdDev */
    /* ref_sts[s][0][3]  =  system s, plab 0  - Global Median */
    /* ref_sts[s][0][4]  =  system s, plab 0  - Global VAriance */
    /* ref_sts[s][0][5]  =  system s, plab 0  - Global Z_stat */
       
    /* the totals */
    for (j=0; j<sc[0]->aset.num_plab; j++){
	for (s=0; s<nsc; s++){
	    npres = sumr = sume = 0;
	    for (k=0; k<sc[s]->num_grp; k++)
		if (labels->lsc[s][j][k].nref + labels->lsc[s][j][k].nerr != 0){
		    ref_words[npres] = labels->lsc[s][j][k].nref;
		    err_words[npres] = pct(labels->lsc[s][j][k].nerr,
					   labels->lsc[s][j][k].nref);
		    sumr += labels->lsc[s][j][k].nref;
		    sume += labels->lsc[s][j][k].nerr;
		    npres++;
		}
	    /* calc and store statistics */
	    calc_mean_var_std_dev_Zstat_double(ref_words,npres,&(ref_sts[s][j][1]),
					       &(ref_sts[s][j][4]),&(ref_sts[s][j][2]),
					       &(ref_sts[s][j][3]),&(ref_sts[s][j][5]));
	    calc_mean_var_std_dev_Zstat_double(err_words,npres,&(err_sts[s][j][1]),
					       &(err_sts[s][j][4]),&(err_sts[s][j][2]),
					       &(err_sts[s][j][3]),&(err_sts[s][j][5]));
	    ref_sts[s][j][0] = sumr;
	    err_sts[s][j][0] = pct(sume,sumr);
	}
    }

    for (i=0; i<4; i++) {
	char *name="";
	switch (i){
	  case 0: name="Set/Subset #Words and System Set/Subset Average Word Error Rate"; break;
	  case 1: name="Set/Subset Mean #Words/Speaker and Set/Subset Mean Word Error Rate/Speaker"; break;
	  case 2: name="Associated Standard Deviations"; break;
	  case 3: name="Set/Subset Median #Words/Speaker and Set/Subset Median Word Error Rate/Speaker"; break;
	  default:
	    ;
	}
	Desc_add_row_separation((i != 2) ? '=' : '-', BEFORE_ROW);

	Desc_add_row_values(statfmt,"",name);
	Desc_add_row_separation('-', BEFORE_ROW);
	for (s=0; s<nsc; s++){
	    Desc_set_iterated_format(fmt);
	    Desc_set_iterated_value(sc[s]->title);
	    for (j=0; j<sc[s]->aset.num_plab; j++){
		Desc_set_iterated_value(rsprintf("[%d]",(int)ref_sts[s][j][i]));
		Desc_set_iterated_value(rsprintf("%.1f",F_ROUND(err_sts[s][j][i],1)));
	    }
	    Desc_flush_iterated_row();
	}
    }
#ifdef as
    Desc_add_row_separation('=', BEFORE_ROW);
    for (i=0; i<4; i++) {
	char *name="";
	switch (i){
	  case 0: name="Set Sum/Avg"; break;
	  case 1: name="Mean"; Desc_add_row_separation('-',BEFORE_ROW); break;
	  case 2: name="StdDev"; break;
	  case 3: name="Median";Desc_add_row_separation(' ',BEFORE_ROW); break;
	  default:
	    ;
	}
	Desc_set_iterated_format(fmt);
	Desc_set_iterated_value(name);
	for (j=0; j<sc->aset.num_plab; j++){
	    Desc_set_iterated_value(rsprintf("[%d]",(int)ref_sts[j][i]));
	    Desc_set_iterated_value(rsprintf("%.1f",F_ROUND(err_sts[j][i],1)));
	}
	Desc_flush_iterated_row();
    }
#endif
    
    Desc_dump_report(1,fp);
    free_singarr(ref_words,double);
    free_singarr(err_words,double);
    free_3dimarr(ref_sts,nsc,sc[0]->aset.num_plab,double);
    free_3dimarr(err_sts,nsc,sc[0]->aset.num_plab,double);
}



/************************************************************************/
/*   print the Labelled Utterance Report                                */
/************************************************************************/
void print_lur(SCORES *sc, char *sys_root_name, int feedback)
{
    char *fname;
    FILE *fp;
    LABEL *labels = (LABEL *)0;

    if (strcmp(sys_root_name,"-") == 0 ||
	((fp = fopen(fname=rsprintf("%s.lur",sys_root_name),"w")) == NULL))
	fp = stdout;
    else
	if (feedback >= 1)
	    printf("    Writing LUR scoring report to '%s'\n",fname);

    labels = alloc_LABEL_SCORES(1, sc);

    compute_label_scores(sc, 0, labels);

    create_report(labels, sc, fp);

    free_LABEL_SCORES(labels, sc, 1);

    if (fp != stdout) fclose(fp);
}


/************************************************************************/
/*   print the several system Labelled Utterance Report                 */
/************************************************************************/
void print_N_lur(SCORES *sc[], int nsc, char *out_root_name, char *test_name, int feedback)
{
    FILE *fp;
    char *fname;
    int i;
    LABEL *labels = (LABEL *)0;

    if (strcmp(out_root_name,"-") == 0 ||
	((fp = fopen(fname=rsprintf("%s.lur",out_root_name),"w")) == NULL))
	fp = stdout;
    else
	if (feedback >= 1)
	    printf("    Writing LUR scoring report to '%s'\n",fname);

    labels = alloc_LABEL_SCORES(nsc, sc[0]);

    for (i=0; i<nsc; i++)
	compute_label_scores(sc[i], i, labels);


    create_N_lur_report(labels, sc, nsc, test_name, fp);

    if (fp != stdout) fclose(fp);
    free_LABEL_SCORES(labels,sc[0],nsc);
}





