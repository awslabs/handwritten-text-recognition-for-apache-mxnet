/**********************************************************************/
/*                                                                    */
/*             FILENAME  wwscr_f.c                                    */
/*             BY:  Jonathan G. Fiscus                                */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*           DESC:  This file contains utilities to perform word      */
/*                  weighted word Scoring                             */
/*                                                                    */
/**********************************************************************/

#include "sctk.h"
#define MAX_BUFF_LEN 1000


static void make_total_summary (WWL_SCORE *wwlscore, FILE *fp);
static void make_speaker_summary (WWL_SCORE *wwlscore, FILE *fp);
static void expand_WWL_SCORE (WWL_SCORE *wwlscr);
static void do_weighted_score(SCORES *sc, WWL *wwl, int dbg, double (*Weight) (TEXT *, WWL *), WWL_FUNC *wwscf);
static WWL_SCORE *alloc_WWL_SCORE (int num_func, int num_spk);
static double get_weight_from_WW (WWL *wwl, TEXT *str);
static int compare_WW (const void *ww1, const void *ww2);
static void free_WWL_SCORE (WWL_SCORE *wwlscr);

int perform_word_weighted_scoring(SCORES *sc, TEXT *sys_root_name, int do_weight_one, int n_wwlf, TEXT **wwl_files, int make_sum, int make_ovrall, int dbg, int feedback) {    
    WWL *wwl=(WWL *)0;
    WWL_SCORE *wwlscore;
    int wf, wfc, maxwwlf;

    maxwwlf = n_wwlf + ((do_weight_one) ? 1 : 0);
    wwlscore = alloc_WWL_SCORE(maxwwlf,sc->num_grp);

    if (do_weight_one){
        wwlscore->func[wwlscore->numfunc].title = TEXT_strdup((TEXT *)"Weight 1");
	do_weighted_score(sc,(WWL *)0,dbg,Weight_one,
			  &(wwlscore->func[wwlscore->numfunc]));
	wwlscore->numfunc++;
    }

    for (wf=0; wf<n_wwlf; wf++){
	if (load_WWL(&wwl, wwl_files[wf]) != 0){
	    fprintf(stderr,"Error: Unable to read WWL file '%s'\n",
		    wwl_files[wf]);
	    return(1);
	}
	/* dump_WWL (wwl, stdout);*/
	for (wfc=0; wfc < wwl->num_w; wfc++){
	    if (wwlscore->numfunc >= wwlscore->maxfunc)
		expand_WWL_SCORE(wwlscore);
	    wwl->curw = wfc;
	    wwlscore->func[wwlscore->numfunc].title = 
		TEXT_strdup(wwl->weight_desc[wfc]);
	    do_weighted_score(sc,wwl,dbg,Weight_wwl,
			      &(wwlscore->func[wwlscore->numfunc]));
	    wwlscore->numfunc++;
	}
	free_WWL(&wwl);
    }

    if (make_sum) {
	FILE *fp;
	char *fname;
	/* open an output file, if it fails, write to stdout */
	fp = stdout;
	if (sys_root_name != (TEXT *)0 && 
	    TEXT_strcmp(sys_root_name,(TEXT*)"-") != 0){
	  if ((fp = fopen(fname=rsprintf("%s.%s",sys_root_name,
					 "wws_brief"),"w")) != NULL)
	    if (feedback > 0)
	      printf("   Writing Weighted Word summary to '%s'\n", fname);
	}
	make_total_summary(wwlscore, fp);
	if (fp != stdout)
	    fclose(fp);
    }

    if (make_ovrall) {
	FILE *fp;
	char *fname;
	/* open an output file, if it fails, write to stdout */
	fp = stdout;
	if (sys_root_name != (TEXT *)0 &&
	    TEXT_strcmp(sys_root_name,(TEXT*)"-") != 0){
	  if ((fp = fopen(fname=rsprintf("%s.%s",sys_root_name,
					 "wws"),"w")) != NULL)
	    printf("Writing Weighted Word summary to '%s'\n",fname);
	}
	make_speaker_summary(wwlscore, fp);
	if (fp != stdout)
	    fclose(fp);
    }

    free_WWL_SCORE(wwlscore);
    return(0);
}

static void expand_WWL_SCORE(WWL_SCORE *wwlscr)
{
    int min = MAX(wwlscr->maxfunc,1), i;
    WWL_FUNC *twwlf;
    /* printf("Expanding WWLSCORE structure\n"); */

    alloc_singarr(twwlf, (min*2), WWL_FUNC);
    memcpy(twwlf,wwlscr->func,wwlscr->maxfunc * sizeof(WWL_FUNC));
    free_singarr(wwlscr->func,WWL_FUNC);
    wwlscr->func = twwlf;
    for (i=wwlscr->maxfunc; i<min*2; i++){
	wwlscr->func[i].n_spkrs = wwlscr->func[0].n_spkrs;
	wwlscr->func[i].title = (TEXT *)0;
	alloc_singarr(wwlscr->func[i].spkr,wwlscr->func[i].n_spkrs,WWL_SPKR);
    }
    wwlscr->maxfunc = min*2;
}

static WWL_SCORE *alloc_WWL_SCORE(int num_func, int num_spk)
{
    WWL_SCORE *twwlscr;
    int i;

    alloc_singarr(twwlscr,1,WWL_SCORE);
    twwlscr->maxfunc = num_func;
    twwlscr->numfunc = 0;
    alloc_singarr(twwlscr->func,twwlscr->maxfunc,WWL_FUNC);
    for (i=0; i<twwlscr->maxfunc; i++){
	twwlscr->func[i].n_spkrs = num_spk;
	twwlscr->func[i].title = (TEXT *)0;
	alloc_singarr(twwlscr->func[i].spkr,twwlscr->func[i].n_spkrs,WWL_SPKR);
    }
    return (twwlscr);
}

static void free_WWL_SCORE(WWL_SCORE *wwlscr)
{
    int i, s;

    for (i=0; i<wwlscr->maxfunc; i++){
	if (i < wwlscr->numfunc){
	    TEXT_free(wwlscr->func[i].title);
	    for (s=0; s<wwlscr->func[i].n_spkrs; s++)
		TEXT_free(wwlscr->func[i].spkr[s].id);
	}
	free_singarr(wwlscr->func[i].spkr,WWL_SPKR);
    }
    free_singarr(wwlscr->func,WWL_FUNC);
    free_singarr(wwlscr,WWL_SCORE);
}

static void make_total_summary(WWL_SCORE *wwlscore, FILE *fp)
{
    int cfunc;
    char *ffmt="%6.2f ";
    Desc_erase();
    Desc_set_page_center(SCREEN_WIDTH);
    Desc_add_row_values("l|c|cccccc"," Weighting ","Err","Corr",
			" Sub"," Del"," Ins"," Spl"," Mrg");
    Desc_add_row_separation('-',BEFORE_ROW);
    
    for (cfunc=0; cfunc < wwlscore->numfunc; cfunc++){ 
	WWL_FUNC *tfunc;
	tfunc = &(wwlscore->func[cfunc]);
	Desc_set_iterated_format("l|c|cccccc");
	Desc_set_iterated_value(rsprintf(" %s ",tfunc->title));
	Desc_set_iterated_value(rsprintf(" %6.2f ",
					 pct(tfunc->sub+tfunc->mrg+
					     tfunc->del+tfunc->ins+
					     tfunc->spl,tfunc->ref)));
	Desc_set_iterated_value(rsprintf(" %6.2f ",
					 pct(tfunc->corr,tfunc->ref)));
	Desc_set_iterated_value(rsprintf(ffmt,pct(tfunc->sub,tfunc->ref)));
	Desc_set_iterated_value(rsprintf(ffmt,pct(tfunc->del,tfunc->ref)));
	Desc_set_iterated_value(rsprintf(ffmt,pct(tfunc->ins,tfunc->ref)));
	Desc_set_iterated_value(rsprintf(ffmt,pct(tfunc->spl,tfunc->ref)));
	Desc_set_iterated_value(rsprintf(ffmt,pct(tfunc->mrg,tfunc->ref)));
	Desc_flush_iterated_row();
    }
    Desc_dump_report(0,fp);
}

static void make_speaker_summary(WWL_SCORE *wwlscore, FILE *fp)
{
    int s, cfunc;
    int splt_mrg = 0;

    Desc_erase();
    Desc_set_page_center(SCREEN_WIDTH);
    if (splt_mrg)
      Desc_add_row_values("c|l|c|cccccc","Spkr"," Weighting ","Err","Corr",
			  " Sub"," Del"," Ins"," Spl"," Mrg");
    else
      Desc_add_row_values("c|l|c|cccc","Spkr"," Weighting ","Err","Corr",
			  " Sub"," Del"," Ins");
    Desc_add_row_separation('-',BEFORE_ROW);
    
    for (s=0; s<wwlscore->func[0].n_spkrs; s++){
	for (cfunc=0; cfunc < wwlscore->numfunc; cfunc++){
	    WWL_FUNC *tfunc;
	    WWL_SPKR *tsp;
	    tfunc = &(wwlscore->func[cfunc]);
	    tsp = &(tfunc->spkr[s]);
	    if (splt_mrg)
	      Desc_set_iterated_format("c|l|c|cccccc");	
	    else
	      Desc_set_iterated_format("c|l|c|cccc");	
	    Desc_set_iterated_value(rsprintf(" %s ",tfunc->spkr[s].id));
	    Desc_set_iterated_value(rsprintf(" %s ",tfunc->title));
	    Desc_set_iterated_value(rsprintf(" %6.2f ",
					     pct(tsp->sub+tsp->mrg+
						 tsp->del+tsp->ins+
						 tsp->spl,tsp->ref)));
	    printf("Vale %f %f\n",tsp->corr,tsp->ref);
	    Desc_set_iterated_value(rsprintf(" %6.2f ",
					     pct(tsp->corr,tsp->ref)));
	    Desc_set_iterated_value(rsprintf("%6.2f ",pct(tsp->sub,tsp->ref)));
	    Desc_set_iterated_value(rsprintf("%6.2f ",pct(tsp->del,tsp->ref)));
	    Desc_set_iterated_value(rsprintf("%6.2f ",pct(tsp->ins,tsp->ref)));
	    if (splt_mrg){
	      Desc_set_iterated_value(rsprintf("%6.2f ",pct(tsp->spl,tsp->ref)));
	      Desc_set_iterated_value(rsprintf("%6.2f ",pct(tsp->mrg,tsp->ref)));
	    }
	    Desc_flush_iterated_row();
	}
	if (s < wwlscore->func[0].n_spkrs-1)
	    Desc_add_row_separation('-',BEFORE_ROW);
    }
    Desc_add_row_separation('=',BEFORE_ROW);
    for (cfunc=0; cfunc < wwlscore->numfunc; cfunc++){
	WWL_FUNC *tfunc;
	tfunc = &(wwlscore->func[cfunc]);

	if (splt_mrg)
	  Desc_set_iterated_format("c|l|c|cccccc");	
	else
	  Desc_set_iterated_format("c|l|c|cccc");	
	Desc_set_iterated_value(rsprintf(" %s ","Ave"));
	Desc_set_iterated_value(rsprintf(" %s ",tfunc->title));
	Desc_set_iterated_value(rsprintf(" %6.2f ",
					 pct(tfunc->sub+tfunc->mrg+
					 tfunc->del+tfunc->ins+
					 tfunc->spl,tfunc->ref)));
	Desc_set_iterated_value(rsprintf(" %6.2f ",
					 pct(tfunc->corr,tfunc->ref)));
	Desc_set_iterated_value(rsprintf("%6.2f ",pct(tfunc->sub,tfunc->ref)));
	Desc_set_iterated_value(rsprintf("%6.2f ",pct(tfunc->del,tfunc->ref)));
	Desc_set_iterated_value(rsprintf("%6.2f ",pct(tfunc->ins,tfunc->ref)));
	if (splt_mrg){
	  Desc_set_iterated_value(rsprintf("%6.2f ",pct(tfunc->spl,tfunc->ref)));
	  Desc_set_iterated_value(rsprintf("%6.2f ",pct(tfunc->mrg,tfunc->ref)));
	}
	Desc_flush_iterated_row();
    }
    Desc_dump_report(0,fp);
}

double Weight_wwl(TEXT *str, WWL *wwl)
{
    double sum=0.0;
    TEXT buffer[100], *p;

    if (TEXT_strchr(str, '=') == (TEXT *)0)
      return (get_weight_from_WW(wwl, str));

    TEXT_strcpy(buffer,str);
    /* Loop through each sub-word and compute the weight */
    /* printf("LOOK AT %s\n",buffer); */
    p = TEXT_strtok(buffer,(TEXT *)"=");
    while (p != NULL){
	/* printf("Computing weight for word '%s'\n",p); */

	sum += get_weight_from_WW(wwl, p);
	/* printf("      %6.2f\n",sum); */

	p = TEXT_strtok(NULL,(TEXT *)"=");
    }
    return(sum);
}


double Weight_one(TEXT *str, WWL *wwl)
{
    return (1.0);
}

static int compare_WW(const void *ww1, const void *ww2)
{
    return(TEXT_strcmp((*(WW **)ww1)->str,(*(WW **)ww2)->str));
}

void dump_WWL(WWL *wwl, FILE *fp)
{
    int i, w;
    fprintf(fp,"Dump of WWL\nFile name: %s\n",wwl->filename);
    for (w=0; w<wwl->num_w; w++)
	fprintf(fp,"   Desc: '%s'\n",wwl->weight_desc[w]);
    fprintf(fp,"Default Missing Weight: %e\n",wwl->default_weight);
    fprintf(fp,"      #: ");
    for (w=0; w<wwl->num_w; w++)
	fprintf(fp,"     Wgt     ");
    fprintf(fp,"   IND   String\n"); 
    for (i=0; i<wwl->num; i++){
	fprintf(fp,"   %4d: ",i);
	for (w=0; w<wwl->num_w; w++)
	    fprintf(fp,"%e ",wwl->words[i]->weight[w]);
	fprintf(fp,"  %s\n",wwl->words[i]->str);
    }
    fprintf(fp,"\n\n");
}

void free_WWL(WWL **wwl)
{
    int w, s;

    TEXT_free((*wwl)->filename);
    for (w=0; w<(*wwl)->num_w; w++)
	TEXT_free((*wwl)->weight_desc[w]);
    
    for (s=0; s<(*wwl)->max; s++)
      if (s < (*wwl)->num) TEXT_free((*wwl)->words[s]->str);

    free_2dimarr((*wwl)->words,(*wwl)->num,WW);

    free_singarr((*wwl),WWL);
}
int load_WWL(WWL **wwl, TEXT *filename)
{
    FILE *fp;
    WWL *twwl;
    WW **twords;
    TEXT buff[MAX_BUFF_LEN], word[MAX_BUFF_LEN];
    TEXT weight[MAX_W][MAX_BUFF_LEN], *p, *e;
    int line=0, count_guess, w, ret, min_weight=10;

    /* Initialize the structur */
    alloc_singarr(twwl,1,WWL);
    twwl->max = twwl->num = twwl->num_w = 0;
    twwl->words = (WW **)0;
    twwl->default_weight = 0.0; 
    for (w=0; w<MAX_W; w++)
	twwl->weight_desc[w] = (TEXT *)0;

    if (filename == (TEXT *)0) {
        return 1;
    }
    if (*(filename) == NULL_TEXT) {
        return 1;
    }

    twwl->filename = TEXT_strdup(filename);

    /************************************************************/
    /*** If filename is "unity", this is a special flag,   ******/
    /*** indicating a weight if 1 should be applied to     ******/
    /*** all words, except "optionally deletable" words.   ******/
    if (TEXT_strcmp(twwl->filename,(TEXT *)"unity") == 0){
      /** the struct is all ready alloc'd, add a dummy entry for **/
      /** sanity, set the default to 1, and return **/
      twwl->default_weight = 1.0;

      alloc_singZ(twwl->words,1,WW *,(WW *)0);
      alloc_singarr(twwl->words[twwl->num],1,WW);
      twwl->words[twwl->num]->weight[0] = twwl->default_weight;
      twwl->words[twwl->num]->str = TEXT_strdup((TEXT *)"the");
      twwl->num++;
      twwl->weight_desc[0] = TEXT_strdup((TEXT*)"Unity Weighting");
    } else {
      count_guess = 100;

      if ((fp=fopen((char *)filename,"r")) == NULL){
        fprintf(stderr,"Warning: Could not open Word-weight file: %s",
		filename);
        fprintf(stderr," No words loaded ! ! ! !\n");
        return 1;
      }
      
      while (TEXT_fgets(buff,MAX_BUFF_LEN,fp) != NULL){
	line++;
        /* ignore comments */
        if (!TEXT_is_comment(buff) && !TEXT_is_empty(buff)){
	  if ((ret=sscanf((char *)buff,"%s %s %s %s %s %s",word,weight[0], weight[1],
			  weight[2], weight[3], weight[4])) < 2){
	    fprintf(stderr,
		    "Error: Unable to parse line %d of WWF file '%s'\n",
		    line,buff);
	    return(1);
	  }
	  if (ret < min_weight) min_weight = ret;
	  if (ret > MAX_W+1) ret = MAX_W + 1;
	  if (twwl->num + 1 > twwl->max) {
	    /* I need much more space, alloc a new one, then free */
	    alloc_singZ(twords,count_guess,WW *,(WW *)0);
	    memcpy(twords,twwl->words,sizeof(WW *) * twwl->num);
	    if (twwl->words != (WW **)0) free_singarr(twwl->words,WW *);
	    twwl->words = twords;
	    
	    twwl->max = count_guess;
	    
	    count_guess *= 1.5;    
	  }
	  alloc_singarr(twwl->words[twwl->num],1,WW);
	  for (w=0; w<ret-1; w++)
	    twwl->words[twwl->num]->weight[w] = (double)atof((char*)weight[w]);
	    
	    twwl->words[twwl->num]->str = TEXT_strdup(word);
	    
	    twwl->num++;
	} else if ((p = TEXT_strstr(buff,(TEXT *)"'Headings'")) != (TEXT *)NULL){
	  /* increment p by the appropriate length */
	  p += strlen("'Headings'");
	  /* skip over the 'Word Spelling' string */
	  if ((p = TEXT_strstr(p,(TEXT *)"'Word Spelling'")) == (TEXT *)NULL){
	    fprintf(stderr,"Warning: Unable to parse label line %s",buff);
	    continue;
	  }
	  p += strlen("'Word Spelling'");
	  /* parse out the description field */
	  while ((p = TEXT_strchr(p,'\'')) != NULL){
	    if ((e = TEXT_strchr(p+1,'\'')) != NULL){
	      alloc_singZ(twwl->weight_desc[twwl->num_w],
			  e-p +1,TEXT,'\0');
	      TEXT_strBcpy(twwl->weight_desc[twwl->num_w],p+1,e-p-1);
	      twwl->num_w ++;
	    }
	    p = e + 1;
	  }
	} else if ((p = TEXT_strstr(buff,(TEXT *)"Default missing weight"))!=(TEXT *)NULL){
	  /* parse out the missing weight field */
	  if ((p = TEXT_strchr(buff,(TEXT)'\'')) != NULL){
	    if ((e = TEXT_strchr(p+1,(TEXT)'\'')) != NULL){
	      *word = (char)'\0';
	      TEXT_strBcpy(word,p+1,e-p-1);
	      twwl->default_weight = (double)atof((char *)word);
	      continue;
	    }
	  }
	  fprintf(stderr,"Warning: Failure to parse 'Default missing weight' %s",buff);
	}
      }
      fclose(fp);
      if (twwl->num_w == 0){
	if (min_weight < 2){
	  fprintf(stderr,"Error: No wieghts defined in WWL '%s'\n",
		  filename);
	  return(1);
	}
	twwl->num_w = min_weight-1;
      }
      
      for (w=0; w<twwl->num_w; w++)
	if (twwl->weight_desc[w] == (TEXT *)0)
	  twwl->weight_desc[w] = TEXT_strdup((TEXT*)rsprintf("%s Col %d",
							     filename,w+1));
    } 
    
    /* Sort the words by pcind */
    qsort(twwl->words,twwl->num,sizeof(WW *),compare_WW);
    *wwl = twwl;

    return (0);
}

static void do_weighted_score(SCORES *sc, WWL *wwl, int dbg, double (*Weight) (TEXT *, WWL *), WWL_FUNC *wwscf){
    int spkr, st, wd;
    double refW, corrW, subW, delW, insW, mrgW, splW;
    double SrefW, ScorrW, SsubW, SdelW, SinsW, SmrgW, SsplW; 
    double rW, hW;
    PATH *path;

    wwscf->ref  =  wwscf->corr = wwscf->sub = wwscf->ins = wwscf->del = 0.0;
    wwscf->mrg  =  wwscf->spl  = 0.0;

    for (spkr=0; spkr < sc->num_grp; spkr++){
	wwscf->spkr[spkr].id = TEXT_strdup((TEXT *)sc->grp[spkr].name);
	SrefW = ScorrW = SsubW = SdelW = SinsW = SmrgW = SsplW = 0.0;
        for (st=0; st<sc->grp[spkr].num_path; st++){
	    refW = corrW = subW = delW = insW = mrgW = splW = 0.0;
	    path = sc->grp[spkr].path[st];
	    if (dbg > 5)
	        PATH_print(path,stdout,80);
	    /* #define WWOP(_a,_b) MAX(_a,_b) */
#define WWOP(_a,_b) (_a + _b)
	    for (wd=0; wd<path->num; wd++){
		if ((path->pset[wd].eval & P_CORR) != 0){
		    refW  += rW = Weight(((WORD *)(path->pset[wd].a_ptr))->value,wwl);
		    corrW += hW = Weight(((WORD *)(path->pset[wd].a_ptr))->value,wwl);
		    if (dbg > 5) printf("Word: R:%.2f C:%.2f\n",rW,hW);
		} else if ((path->pset[wd].eval & P_SUB) != 0){
		    rW     = Weight(((WORD *)(path->pset[wd].a_ptr))->value,wwl);
		    hW     = Weight(((WORD *)(path->pset[wd].b_ptr))->value,wwl);
		    refW  += rW;
		    subW  += WWOP(rW,hW);
		    if (dbg > 5) printf("Word: R:%.2f S:%.2f\n",rW,hW);
		} else if ((path->pset[wd].eval & P_DEL) != 0){
		    refW  += rW = Weight(((WORD *)(path->pset[wd].a_ptr))->value,wwl);
		    delW  += hW = Weight(((WORD *)(path->pset[wd].a_ptr))->value,wwl);
		    if (dbg > 5) printf("Word: R:%.2f D:%.2f\n",rW,hW);
		} else if ((path->pset[wd].eval & P_INS) != 0){
		    insW  += hW = Weight(((WORD *)(path->pset[wd].b_ptr))->value,wwl);
		    if (dbg > 5) printf("Word: R:%.2f I:%.2f\n",0.0,hW);
		} else if ((path->pset[wd].eval & P_MRG) != 0){
		    rW     = Weight(((WORD *)(path->pset[wd].a_ptr))->value,wwl);
		    hW     = Weight(((WORD *)(path->pset[wd].b_ptr))->value,wwl);
		    refW  += rW;
		    mrgW  += WWOP(rW,hW);
		    if (dbg > 5) printf("Word: R:%.2f M:%.2f\n",rW,hW);
		} else if ((path->pset[wd].eval & P_SPL) != 0){
		    rW     = Weight(((WORD *)(path->pset[wd].a_ptr))->value,wwl);
		    hW     = Weight(((WORD *)(path->pset[wd].b_ptr))->value,wwl);
		    refW  += rW;
		    splW  += WWOP(rW,hW);
		    if (dbg > 5) printf("Word: R:%.2f P:%.2f\n",rW,hW);
		} else {
		    fprintf(stderr,"Error: undefined evaluation %d word %d\n",
			    path->pset[wd].eval,wd);
		}
	    }
	    if (dbg > 4){
		printf("Sentence: R:%.2f  C:%.2f  S:%.2f  D:%.2f",
		       refW,corrW,subW,delW);
		printf("  I:%.2f  Mr:%.2f  Sp:%.2f\n",insW,mrgW,splW);
	    }
	    SrefW  += refW;  ScorrW += corrW;   SsubW  += subW;  SdelW += delW;
	    SinsW  += insW;  SmrgW  += mrgW;	SsplW  += splW;
	}
	if (dbg > 3){
	    printf("Speaker: R:%.2f  C:%.2f  S:%.2f  D:%.2f",
		   SrefW,ScorrW,SsubW,SdelW);
	    printf("  I:%.2f  Mr:%.2f  Sp:%.2f\n",SinsW,SmrgW,SsplW);
	}
	wwscf->ref  += SrefW;	wwscf->spkr[spkr].ref  = SrefW;
	wwscf->corr += ScorrW;	wwscf->spkr[spkr].corr = ScorrW;
	wwscf->sub  += SsubW;	wwscf->spkr[spkr].sub  = SsubW;
	wwscf->ins  += SinsW;	wwscf->spkr[spkr].ins  = SinsW;
	wwscf->del  += SdelW;	wwscf->spkr[spkr].del  = SdelW;
	wwscf->mrg  += SmrgW;	wwscf->spkr[spkr].mrg  = SmrgW;
	wwscf->spl  += SsplW;	wwscf->spkr[spkr].spl  = SsplW;
    }
    if (dbg > 2){
	printf("Total: R:%e  C:%e  S:%e  D:%e",
	       wwscf->ref, wwscf->corr, wwscf->sub,wwscf->del);
	printf("  I:%e  Mr:%e  Sp:%e\n",
	       wwscf->ins, wwscf->mrg, wwscf->spl);
    }
}


static double get_weight_from_WW(WWL *wwl, TEXT *str)
{
    /* This implements a binary search */
    int low, high, mid, eval;
    static int env_checked = 0, dbg = 0;
    
    if (!env_checked){
      if (getenv("WWL_PRINT_DEFAULT") != NULL) { dbg = 1; }
      env_checked = 1;
    }

    low = 0, high = wwl->num-1;
 
    do { 
        mid = (low + high)/2;
	if (mid < wwl->num){
	  eval = TEXT_strcmp(str,wwl->words[mid]->str);
	  if (eval == 0)
            return(wwl->words[mid]->weight[wwl->curw]);
	  if (eval < 0)
            high = mid-1;
	  else
            low = mid+1;
	}
    } while (low <= high);
    if (dbg)
      printf("Default weight: %f supplied for word %s\n",wwl->default_weight,str);
    return(wwl->default_weight);
}
