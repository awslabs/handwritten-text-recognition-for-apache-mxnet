#include "sctk.h"

static int scores_are_diff(PATH **path_set,int npath);

SC_CORRESPONDENCE *alloc_SC_CORRESPONDENCE(SCORES *scor[], int nsc){
  SC_CORRESPONDENCE *tcor;
  int g;

  alloc_singarr(tcor,1,SC_CORRESPONDENCE);
  tcor->scor = scor;
  tcor->nscor = nsc;
  tcor->max_grp = scor[0]->num_grp;
  tcor->num_grp = 0;
  alloc_2dimarr(tcor->grp,tcor->max_grp,1,SC_COR_GRP);
  for (g=0; g < tcor->max_grp; g++)
    alloc_singZ(tcor->grp[g]->grp_ptr,tcor->nscor,int,0);

  /* calculate the maximum number of paths */
  tcor->max_path = 0;
  for (g=0; g < tcor->max_grp; g++)
    if (tcor->max_path < tcor->scor[0]->grp[g].num_path)
      tcor->max_path = tcor->scor[0]->grp[g].num_path;

  /* allocate data for the paths */
  for (g=0; g < tcor->max_grp; g++){
    tcor->grp[g]->num_path = 0;
    alloc_2dimZ(tcor->grp[g]->path_ptr,tcor->max_path,tcor->nscor,int,0);
  }

  return(tcor);
}

void locate_matched_data(SCORES *scor[], int nscor, SC_CORRESPONDENCE **corresp){
  *corresp = alloc_SC_CORRESPONDENCE(scor, nscor);

  find_matched_grp(*corresp);
  find_matched_paths(*corresp);

}

void dump_paths_of_SC_CORRESPONDENCE(SC_CORRESPONDENCE *corresp, int maxlen, FILE *fp, int score_diff){
  int gr, sc, pa;
  int totRefWord = 0, refWord = 0;
  int totErrRefWord = 0, errRefWord = 0;
  int numPaths = 0, numIdentPaths = 0;
  AUTO_LEX alex; 

  AUTO_LEX_init(&alex, 1000);

  fprintf(fp,"Common scoring speakers and segments\n");
  fprintf(fp,"Version 0.1\n");
  fprintf(fp,"\n");

  fprintf(fp,"Represented Systems\n");
  for (sc=0; sc<corresp->nscor; sc++){
    fprintf(fp,"    name: %s   Ref: %s   Hyp: %s\n",
	    corresp->scor[sc]->title,
	    corresp->scor[sc]->ref_fname,
	    corresp->scor[sc]->hyp_fname);
  }
  fprintf(fp,"    End of Represented Systems\n");
  fprintf(fp,"\n");

  if (corresp->scor[0]->aset.num_plab > 0 || corresp->scor[0]->aset.num_cat > 0){
    int i;
    fprintf(fp,"Utterance Label Definitions:\n");
    for (i=0; i<corresp->scor[0]->aset.num_cat; i++)
      fprintf(fp,"    Category: id: \"%s\" title: \"%s\" "
	      "description: \"%s\"\n",corresp->scor[0]->aset.cat[i].id,
	      corresp->scor[0]->aset.cat[i].title,corresp->scor[0]->aset.cat[i].desc);
    for (i=0; i<corresp->scor[0]->aset.num_plab; i++)
      fprintf(fp,"    Label: id: \"%s\" title: \"%s\" "
	      "description: \"%s\"\n",corresp->scor[0]->aset.plab[i].id,
	      corresp->scor[0]->aset.plab[i].title,corresp->scor[0]->aset.plab[i].desc);
  }
  fprintf(fp,"    End of Utterance Label Definitions\n");
  fprintf(fp,"\n\n");

  for (gr=0; gr<corresp->num_grp; gr++){ 
    int to_print = 1;

    for (pa=0; pa<corresp->grp[gr]->num_path; pa++){
      PATH *path_set[100]; 
      fprintf(fp,"Speaker: %s\n",
	      corresp->scor[0]->grp[corresp->grp[gr]->grp_ptr[0]].name);
      /* load an array with paths !!! */
      for (sc=0; sc<corresp->nscor; sc++){
	int gnum, pnum;
	gnum = corresp->grp[gr]->grp_ptr[sc]; 
	pnum = corresp->grp[gr]->path_ptr[pa][sc];
	path_set[sc] = corresp->scor[sc]->grp[gnum].path[pnum];
      }

      if (score_diff && ! scores_are_diff(path_set,corresp->nscor))
	to_print = 0;
      if (to_print)
	PATH_multi_print(corresp->scor, path_set,corresp->nscor, maxlen, fp, &refWord, &errRefWord, &alex);
      totRefWord += refWord;
      totErrRefWord += errRefWord;
    }
  }

  fprintf(fp,"Lattice Summary: #TotRefWords %d #TotRefErrWords %d Accuracy %5.2f\n",totRefWord, totErrRefWord,
	  ((float)(totRefWord - totErrRefWord)/(float)totRefWord)*100.0);

  AUTO_LEX_printout(&alex, fp, "\nList of not correct reference words", 1);
}

void dump_SC_CORRESPONDENCE(SC_CORRESPONDENCE *corresp, FILE *fp){
  int gr, sc, pa;

  fprintf(fp,"SCORES CORRESPONDENCE Common groups\n");
  for (gr=0; gr<corresp->num_grp; gr++){
    fprintf(fp,"  ");
    for (sc=0; sc<corresp->nscor; sc++)
      fprintf(fp,"%s ",
	      corresp->scor[sc]->grp[corresp->grp[gr]->grp_ptr[sc]].name);
    fprintf(fp,"\n");
  }
  for (gr=0; gr<corresp->num_grp; gr++){ 
    fprintf(fp,"%s\n",
	    corresp->scor[0]->grp[corresp->grp[gr]->grp_ptr[0]].name);
    for (pa=0; pa<corresp->grp[gr]->num_path; pa++){
      fprintf(fp,"   ");
      for (sc=0; sc<corresp->nscor; sc++)
	fprintf(fp," %s",corresp->scor[sc]->grp[corresp->grp[gr]->grp_ptr[sc]].
		path[corresp->grp[gr]->path_ptr[pa][sc]]->id);
      fprintf(fp,"\n");
    }
  }
}


void find_matched_grp(SC_CORRESPONDENCE *corresp){
  int mas_grp, sc, gr;

  /* locate all the corresponding speakers, using scor[0] as the reference 
   * point */

  for (corresp->num_grp=0, mas_grp=0; mas_grp < corresp->scor[0]->num_grp;
       mas_grp++) {
    /* locate this group in all the other corresp->scor structures */
    corresp->grp[corresp->num_grp]->grp_ptr[0] = mas_grp;
    for (sc=1; sc<corresp->nscor; sc++){
      for (gr=0; gr < corresp->scor[sc]->num_grp; gr++){ /* for all speakers */
	corresp->grp[corresp->num_grp]->grp_ptr[sc] = gr;
	if (strcmp(corresp->scor[0]->
		   grp[corresp->grp[corresp->num_grp]->grp_ptr[0]].name,
		   corresp->scor[sc]->
		   grp[corresp->grp[corresp->num_grp]->grp_ptr[sc]].name) == 0)
	  break;
      }      
      if (gr == corresp->scor[sc]->num_grp)
	break;
    }
    if (sc < corresp->nscor)
      break;

    corresp->num_grp++;
    
  }

  /* look to see if any speakers are not in common */
  for (sc=0; sc<corresp->nscor; sc++){
    int scmis, cgr;
    scmis = 0;
    for (gr=0; gr < corresp->scor[sc]->num_grp; gr++){ /* for all speakers */
      /* search the mapping array for the group */
      for (cgr=0; cgr<corresp->num_grp; cgr++)
	if (corresp->grp[cgr]->grp_ptr[sc] == gr)
	  break;
      if (cgr == corresp->num_grp){
	if (scmis ++ == 0)
	  fprintf(scfp,"Warning: system %s has non-unique speakers:",
		  corresp->scor[sc]->title);
	fprintf(scfp," %s",corresp->scor[sc]->grp[gr].name);
      }
    }
    if (scmis > 0) fprintf(scfp,"\n");
  }
}

void find_matched_paths(SC_CORRESPONDENCE *corresp){
  int sc, gr, *np, mpa, pa;

  /* locate all the corresponding paths, using grp[0] as the reference 
   * point */

  for (gr=0; gr < corresp->num_grp; gr++) {
    np = &(corresp->grp[gr]->num_path);
    for (mpa=0;
	 mpa < corresp->scor[0]->grp[corresp->grp[gr]->grp_ptr[0]].num_path;
	 mpa++){
      /* set the next path to mpa */
      corresp->grp[corresp->grp[gr]->grp_ptr[0]]->path_ptr[*np][0] = mpa;
      
      /* search all the other scor strucs for a matching id */
      for (sc=1; sc<corresp->nscor; sc++){
	for (pa=0; pa < 
	       corresp->scor[sc]->grp[corresp->grp[gr]->grp_ptr[sc]].num_path;
	     pa++){
	  corresp->grp[gr]->path_ptr[*np][sc] = pa;

	  if (strcmp(corresp->scor[0]->
		     grp[corresp->grp[gr]->grp_ptr[0]] .path[mpa]->id,
		     corresp->scor[sc]->
		     grp[corresp->grp[gr]->grp_ptr[sc]].path[pa]->id) == 0)
	    break;
	}      
	if (pa==corresp->scor[sc]->grp[corresp->grp[gr]->grp_ptr[sc]].num_path)
	  break;
      }
      if (sc == corresp->nscor)
	(*np)++;
    }
  }

  /* look to see if any of the common speakers paths are not in common */
  for (gr=0; gr < corresp->num_grp; gr++) {
    for (sc=0; sc<corresp->nscor; sc++){
      int scmis, p;
      /* search the paths in the score, making sure they are there */
      for (scmis = 0, pa=0; pa < 
	     corresp->scor[sc]->grp[corresp->grp[gr]->grp_ptr[sc]].num_path;
	   pa++){
	for (p=0; p < corresp->grp[gr]->num_path; p++){
	  if( corresp->grp[gr]->path_ptr[p][sc]==pa)
	    break;
	}
	if (p == corresp->grp[gr]->num_path){
	  if (scmis++ == 0)
	    fprintf(scfp,"Warning: system %s, speaker %s has non-unique path:",
		    corresp->scor[sc]->title,
		   corresp->scor[sc]->grp[corresp->grp[gr]->grp_ptr[sc]].name);
	  fprintf(scfp," %s",
		  corresp->scor[sc]->grp[corresp->grp[gr]->grp_ptr[sc]].
		  path[pa]->id);
	}	
      }
      if (scmis > 0) fprintf(scfp,"\n");
    }
  }
}

static int scores_are_diff(PATH **path_set,int npath){
  int p, r, e, nref=0, nerr=0, w;

  for (p=0; p<npath; p++){
    r = e = 0;
    for (w=0; w < path_set[0]->num; w++){
      if ((path_set[p]->pset[w].eval & P_INS) == 0)	r++;
      if ( ((path_set[p]->pset[w].eval & P_DEL) != 0) ||
	   ((path_set[p]->pset[w].eval & P_INS) != 0) || 
	   ((path_set[p]->pset[w].eval & P_SUB) != 0))
	e++;
    }
    if (p == 0) { nref = r;  nerr = e; }
    else {
      if (nref != r || nerr != e)
	return(1);
    }
  }
  return(0);
}


/*****************************************************************/
/***********  Multi-path_print-Functions            **************/
static int more_data(PATH **path_set, int npath, int *psets);
static int inserts_exist(SCORES **scor, PATH **path_set, int npath, int *psets);
static void add_header_data(SCORES **scor, PATH **path_set, int npath, TEXT **outbuf, int *outlen, int maxlen, int continuation);
static void flush_text(SCORES **scor, PATH **path_set, int npath, TEXT **outbuf, int *outlen, int maxlen, FILE *fp);
static void process_inserts(SCORES **scor, PATH **path_set, int npath, int *psets, TEXT **outbuf, int *outlen, int maxlen, FILE *fp);
static void process_rest(SCORES **scor, PATH **path_set, int npath, int *psets, TEXT **outbuf, int *outlen, int maxlen, FILE *fp);
static int identical_refs(int npaths,PATH **paths);
static void lattice_error(int npaths,PATH **paths, int *refWord, int *refErrRefWord, AUTO_LEX *alex);


/* return 1 if the paths have the identical reference strings */
static int identical_refs(int npaths,PATH **paths){
  int p, w, wp;  

  if (npaths == 0) {return 0;}
  if (npaths == 1) {return 1;}

  for (p=1; p<npaths; p++){
    w=0; wp=0;
    while (w < paths[0]->num && wp < paths[p]->num){
      /* skip insertions */
      while (w < paths[0]->num && (paths[0]->pset[w].eval == P_INS))
	w++;
      while (wp < paths[p]->num && (paths[p]->pset[wp].eval == P_INS))
	wp++;
      if (w >= paths[0]->num && wp >= paths[p]->num)
	;
      else {
	if (w > paths[0]->num || wp > paths[p]->num)
	  return 0;
	if (TEXT_strcmp(((WORD *)(paths[0]->pset[w].a_ptr))->value,
			((WORD *)(paths[p]->pset[wp].a_ptr))->value) != 0)
	  return(0);
	w++; wp++;
      }
    }    
  }
  return 1;
}

/* Counts the number of ref words and the number with at least on systemwith the correct word */
static void lattice_error(int npaths,PATH **paths, int *refWord, int *refErrRefWord, AUTO_LEX *alex){
  int p, w, wp, ind;  
  int *evals;

  if (npaths == 0 || npaths == 1) {
    fprintf(stderr,"Error: these are paths are not compatable");
    exit(1);
  }

  /* Build an array to keep track */
  alloc_singZ(evals, paths[0]->num, int, P_INS);
  w = 0;
  while (w < paths[0]->num){
    evals[w] = paths[0]->pset[w].eval;
    w++;
  }

  for (p=1; p<npaths; p++){
    w=0; wp=0;
    while (w < paths[0]->num && wp < paths[p]->num){
      /* skip insertions */
      while (w < paths[0]->num && (paths[0]->pset[w].eval == P_INS))
	w++;
      while (wp < paths[p]->num && (paths[p]->pset[wp].eval == P_INS))
	wp++;
      if (w >= paths[0]->num && wp >= paths[p]->num)
	;
      else {
	if (w > paths[0]->num || wp > paths[p]->num){
	  fprintf(stderr,"Error: these are paths are not compatable");
	  exit(1);
	}
	if (TEXT_strcmp(((WORD *)(paths[0]->pset[w].a_ptr))->value,
			((WORD *)(paths[p]->pset[wp].a_ptr))->value) != 0){
	  fprintf(stderr,"Error: these are paths are not compatable");
	  exit(1);	
	}
	if (paths[p]->pset[wp].eval == P_CORR)
	  evals[w] = P_CORR;
	w++; wp++;
      }
    }    
  }
  *refWord = 0;
  *refErrRefWord = 0;
  for (w=0; w < paths[0]->num; w++){
    if (evals[w] != P_INS){
      if (evals[w] != P_CORR){
	ind = AUTO_LEX_find(alex, ((WORD *)(paths[0]->pset[w].a_ptr))->value);
	if (ind < 0){
	  ind = AUTO_LEX_insert(alex, ((WORD *)(paths[0]->pset[w].a_ptr))->value);
	  alex->field_a[ind] = 0;
	}
	alex->field_a[ind] ++;
	(*refErrRefWord) ++;
      }
      (*refWord) ++;
    }
  }
  free_singarr(evals, int);
}

static int more_data(PATH **path_set, int npath, int *psets){
  int p;
  for (p=0; p<npath; p++)
    if (path_set[p]->num > psets[p])
      return(1);
  return(0);
}

static int inserts_exist(SCORES **scor, PATH **path_set, int npath, int *psets){
  int p;
  for (p=0; p<npath; p++)
    if ((path_set[p]->num > psets[p]) && 
	(path_set[p]->pset[psets[p]].eval & P_INS) != 0)
      return(1);
  return(0);
}

static void add_header_data(SCORES **scor, PATH **path_set, int npath, TEXT **outbuf, int *outlen, int maxlen, int continuation){
  int max=0, p; 
  char fmt[40];

  /* for ref: */
  max = 3;
  for (p=0; p<npath; p++)
    if (max < TEXT_strlen((TEXT*)scor[p]->title))
      max = TEXT_strlen((TEXT*)scor[p]->title);

  if (max + 2 + ((continuation)?4:0) > maxlen){
    fprintf(stderr,"Error: Failed to add header data lines for PRN report.  System names too long.  increase via -l option\n");
    exit(1);
  }

  sprintf(fmt,"%s%%-%ds",(continuation)?">>> ":"",max + 2);
  for (p=0; p<npath; p++)
    sprintf((char *)outbuf[p],fmt, rsprintf("%s:",scor[p]->title));
  sprintf((char *)outbuf[npath],fmt, "REF:");
  
  *outlen = max + 2 + ((continuation)?4:0);
  
}

static void flush_text(SCORES **scor, PATH **path_set, int npath, TEXT **outbuf, int *outlen, int maxlen, FILE *fp){
  int p; 
  fprintf(fp,"%s\n",outbuf[npath]);
  for (p=0; p<npath; p++)
    fprintf(fp,"%s\n",outbuf[p]);
  fprintf(fp,"\n");

  add_header_data(scor, path_set, npath, outbuf, outlen, maxlen, 1);
}

static void process_inserts(SCORES **scor, PATH **path_set, int npath, int *psets, TEXT **outbuf, int *outlen, int maxlen, FILE *fp){
  int p, max;
  char fmt[40];

  /* make measurements Ref first, then hyps */
  max =  0;
  for (p=0; p<npath; p++){    
    if ((path_set[p]->num > psets[p]) && 
	((path_set[p]->pset[psets[p]].eval & P_INS) != 0)){
      if (max < TEXT_strlen(((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value))
	max = TEXT_strlen(((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value);
    }
  }
  sprintf(fmt,"%%-%ds",max);
  if (max + *outlen > maxlen-1)
    flush_text(scor, path_set, npath, outbuf, outlen, maxlen, fp);

  for (p=0; p<npath; p++){    
    if ((path_set[p]->num > psets[p]) && 
	(path_set[p]->pset[psets[p]].eval & P_INS) != 0){
      TEXT *buf;
      buf = ((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value;
      if (BF_notSET(path_set[p]->attrib,PA_CASE_SENSE)){
        buf = TEXT_str_to_master(buf, 0);
      }
      //      TEXT_strBcpy(buf,((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value,200);
      //      if (BF_notSET(path_set[p]->attrib,PA_CASE_SENSE)) 
      //      	TEXT_str_to_upp(buf);
      TEXT_strcat(outbuf[p],(TEXT *)rsprintf(fmt,buf));
      psets[p] ++;
    } else 
      TEXT_strcat(outbuf[p],(TEXT *)rsprintf(fmt,""));
  }
  TEXT_strcat(outbuf[npath],(TEXT *)rsprintf(fmt,""));

  *outlen += max;

  if (1 + *outlen < maxlen-1){
    for (p=0; p<=npath; p++)
      TEXT_strcat(outbuf[p],(TEXT *)" ");
    *outlen += 1;
  }
}

static void process_rest(SCORES **scor, PATH **path_set, int npath, int *psets, TEXT **outbuf, int *outlen, int maxlen, FILE *fp){
  int p, max, ref_done;
  char fmt[40];

  /* make measurements Ref first, then hyps */
  max = TEXT_strlen(((WORD *)path_set[0]->pset[psets[0]].a_ptr)->value);
  for (p=0; p<npath; p++){    
    if ((path_set[p]->num > psets[p]) && 
	((path_set[p]->pset[psets[p]].eval & P_DEL) == 0)){
      if (max < TEXT_strlen(((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value))
	max = TEXT_strlen(((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value);
    }
  }
  sprintf(fmt,"%%-%ds",max);
  if (max + *outlen > maxlen-1)
    flush_text(scor, path_set, npath, outbuf, outlen, maxlen, fp);

  ref_done = 0;
  for (p=0; p<npath; p++){
    if ((! ref_done) && (path_set[p]->num > psets[p])){
      TEXT_strcat(outbuf[npath],(TEXT *)rsprintf(fmt,
		 ((WORD *)path_set[p]->pset[psets[p]].a_ptr)->value));
      ref_done = 1;
    }
    if (path_set[p]->num > psets[p]){
      TEXT *buf, *aster = (TEXT *)"****************************************";
      if ((path_set[p]->pset[psets[p]].eval & P_DEL) == 0){
        buf = ((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value;
	if (((path_set[p]->pset[psets[p]].eval & P_CORR) == 0) && BF_notSET(path_set[p]->attrib,PA_CASE_SENSE))
	  buf = TEXT_str_to_master(buf, 0);
      } else {
        buf = aster + 40 - MIN(40,max);
      }
//      if ((path_set[p]->pset[psets[p]].eval & P_DEL) == 0){
//	TEXT_strBcpy(buf,((WORD *)path_set[p]->pset[psets[p]].b_ptr)->value,200);
//	if (((path_set[p]->pset[psets[p]].eval & P_CORR) == 0) &&
//	    BF_notSET(path_set[p]->attrib,PA_CASE_SENSE))
//	  TEXT_str_to_upp(buf);
//      } else {
//	TEXT_strBcpy(buf,aster + 40 - MIN(40,max),40); 
//      }
      TEXT_strcat(outbuf[p],(TEXT *)rsprintf(fmt,buf));
      psets[p] ++;
    }
  }
  *outlen += max;

  if (1 + *outlen < maxlen-1){
    for (p=0; p<=npath; p++)
      TEXT_strcat(outbuf[p],(TEXT *)" ");
    *outlen += 1;
  }
}

/* Pretty print n paths, using the ref transcript as the guide */
void PATH_multi_print(SCORES **scor, PATH **path_set, int npath, int maxlen, FILE *fp, int *refWord,int *errRefWord, AUTO_LEX *alex){
  int p;

  if (maxlen < 80) maxlen = 80;

  if (npath == 1)
    PATH_print(path_set[0],fp,maxlen);
  else if (identical_refs(npath,path_set) == 0){
    fprintf(fp,
	    ";; Reference strings not identical, dumping individual paths\n");
    for (p=0; p<npath; p++){
      PATH_print(path_set[p],fp,maxlen);
    }
  } else {
    int outlen, *psets;
    TEXT **outbuf;

    lattice_error(npath, path_set, refWord, errRefWord, alex);

    fprintf(fp,"Id:     %s\n",path_set[0]->id);
    if (path_set[0]->labels != (char *)0)
      fprintf(fp,"Labels: %s\n",path_set[0]->labels);
    if (path_set[0]->file != (char *)0)
      fprintf(fp,"File: %s\n",path_set[0]->file);
    if (path_set[0]->channel != (char *)0)
      fprintf(fp,"Channel: %s\n",path_set[0]->channel);
    if (BF_isSET(path_set[0]->attrib,PA_REF_TIMES))
	fprintf(fp,"Ref times: t1= %.2f t2= %.2f\n",path_set[0]->ref_t1,path_set[0]->ref_t2);
    if (BF_isSET(path_set[0]->attrib,PA_HYP_TIMES))
	fprintf(fp,"Hyp times: t1= %.2f t2= %.2f\n",path_set[0]->hyp_t1,path_set[0]->hyp_t2);

    fprintf(fp,"Lattice Analysis: #refWords %d #refErrWords %d\n",*refWord, *errRefWord);
#ifdef later
    for (p=0; p<npath; p++){
      int c,s,d,n,u;
      c = s = d = n = u = 0;
      for (i=used; i<to; i++)
	if ((path->pset[i].eval & P_CORR) != 0) c++;
	else if ((path->pset[i].eval & P_SUB) != 0) s++;
	else if ((path->pset[i].eval & P_INS) != 0) n++;
	else if ((path->pset[i].eval & P_DEL) != 0) d++;
	else u ++;
      if (u > 0)
	fprintf(fp,"Scores: (#C #S #D #I #UNK) %d %d %d %d %d\n",c,s,d,n,u);
      else
	fprintf(fp,"Scores: (#C #S #D #I) %d %d %d %d\n",c,s,d,n);	   
    }

#endif
    fprintf(fp,"\n\n");
    
    /* output[npath] is the reference string */
    alloc_2dimZ(outbuf,npath + 1,maxlen+1+40,TEXT,(TEXT)'\0');
    alloc_1dimZ(psets,npath,int,0);   /* initialized to zero */
    add_header_data(scor, path_set, npath, outbuf, &outlen, maxlen, 0);
    while (more_data(path_set, npath, psets) > 0){
      if (inserts_exist(scor, path_set, npath, psets))
	process_inserts(scor, path_set, npath, psets, outbuf, &outlen, maxlen, fp);
      else
	process_rest(scor, path_set, npath, psets, outbuf, &outlen, maxlen, fp);
    }
    flush_text(scor, path_set, npath, outbuf, &outlen, maxlen, fp);
    
    free_2dimarr(outbuf,npath+1,TEXT);
    free_1dimarr(psets,int); 
  }
}

/***********  END OF Multi-path_print-Functio       **************/
/*****************************************************************/
