#include "sctk.h"

/* This variable is used to set the sequence number in the path structure */
static int PATH_sequence_number = 0;

void PATH_add_utt_id(PATH *path, char *utt_id){
    path->id = (char *)TEXT_strdup((TEXT *)utt_id);
}

void PATH_add_label(PATH *path, char *label){
    if (label != (char *)0)
	path->labels = (char *)TEXT_strdup((TEXT *)label);
}

void PATH_add_file(PATH *path, char *file){
    if (file != (char *)0)
	path->file = (char *)TEXT_strdup((TEXT *)file);
}

void PATH_add_channel(PATH *path, char *channel){
    if (channel != (char *)0)
	path->channel = (char *)TEXT_strdup((TEXT *)channel);
}

void PATH_set_sequence(PATH *path){ 
   path->sequence = PATH_sequence_number++;
}

PATH *PATH_alloc(int s){
    PATH *tp;
    /* Set a minimum size for the maximum number of items */
    if (s < 4) s=5;
    alloc_singarr(tp,1,PATH);
    tp->max = s;
    tp->num = 0;
    tp->id = (char *)0;
    tp->labels = (char *)0;
    tp->file = (char *)0;
    tp->channel = (char *)0;
    tp->attrib = PA_NONE;
    tp->sequence = -1;
    tp->ref_t1 = tp->ref_t2 = 0.0; 
    tp->hyp_t1 = tp->hyp_t2 = 0.0;
    alloc_singarr(tp->pset,tp->max,PATH_SET);
    return tp;
}

void check_space_in_PATH(PATH *path){
    if (path->num >= path->max)
	expand_singarr(path->pset,path->num,path->max,2,PATH_SET);
}

void PATH_print(PATH *path, FILE *fp, int max){
   PATH_n_print(path, fp, 0, path->num, max);
}

void PATH_n_print(PATH *path, FILE *fp, int from, int to, int max){
    int i, lena, lenb, alen=40, plen=0, endpass, used;
    int c,s,d,n,u,aux_max=0,line=0,wmax;
    char fmt[30];
    TEXT *astr, *bstr, 
         *aster=(TEXT *) "****************************************",
         buf[100];

    if (to > path->num) to = path->num;
    used = (from >= 0 && from < to) ? from : 0;
    c = s = d = n = u = 0;

    /* collect preliminary info */
    for (i=used; i<to; i++)
	if ((path->pset[i].eval & P_CORR) != 0) c++;
	else if ((path->pset[i].eval & P_SUB) != 0) s++;
	else if ((path->pset[i].eval & P_INS) != 0) n++;
	else if ((path->pset[i].eval & P_DEL) != 0) d++;
	else u ++;
    /* compute the max of the auxillary info */
    if (BF_isSET(path->attrib,PA_HYP_CONF))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_REF_CONF))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_HYP_WTIMES))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_REF_WTIMES))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_HYP_WEIGHT))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_REF_WEIGHT))
	aux_max = (aux_max < 7) ? 7 : aux_max;


    fprintf(fp,"id: %s\n",(path->id == (char *)0) ? "" : path->id);
    if (from != 0 || to != path->num)
	fprintf(fp,"Range: %d-%d of %d words\n",from,to,path->num);
    if (path->labels != (char *)0)
	fprintf(fp,"Labels: %s\n",path->labels);
    if (path->file != (char *)0)
	fprintf(fp,"File: %s\n",path->file);
    if (path->channel != (char *)0)
	fprintf(fp,"Channel: %s\n",path->channel);
    if (u > 0)
	fprintf(fp,"Scores: (#C #S #D #I #UNK) %d %d %d %d %d\n",c,s,d,n,u);
    else
	fprintf(fp,"Scores: (#C #S #D #I) %d %d %d %d\n",c,s,d,n);	   
    if (BF_isSET(path->attrib,PA_CHAR_ALIGN) ||
	BF_isSET(path->attrib,PA_CASE_SENSE) ||
	BF_isSET(path->attrib,PA_REF_WEIGHT) ||
	BF_isSET(path->attrib,PA_HYP_WEIGHT)){
	fprintf(fp,"Attributes: ");
	if (BF_isSET(path->attrib,PA_CHAR_ALIGN))
	    fprintf(fp,"Character_align ");
	if (BF_isSET(path->attrib,PA_CASE_SENSE))
	    fprintf(fp,"Case_sensitve ");
	if (BF_isSET(path->attrib,PA_REF_WEIGHT))
	    fprintf(fp,"Ref_weight ");
	if (BF_isSET(path->attrib,PA_HYP_WEIGHT))
	    fprintf(fp,"Hyp_weight ");
	fprintf(fp,"\n");	
    }
    if (BF_isSET(path->attrib,PA_REF_TIMES))
	fprintf(fp,"Ref times: t1= %.2f t2= %.2f\n",path->ref_t1,path->ref_t2);
    if (BF_isSET(path->attrib,PA_HYP_TIMES))
	fprintf(fp,"Hyp times: t1= %.2f t2= %.2f\n",path->hyp_t1,path->hyp_t2);

    while (used < to){
	/* calculate the max words for this pass */
	plen = (used > 0) ? 9 : 6;
	for (i=used; plen < max && i<to; i++){
	    lena = (path->pset[i].eval == P_INS) ? 1 :
		TEXT_strlen(((WORD *)(path->pset[i].a_ptr))->value);
	    lenb = (path->pset[i].eval == P_DEL) ? 1 :
		TEXT_strlen(((WORD *)(path->pset[i].b_ptr))->value);
	    plen += 1 + MAX3(aux_max,lena,lenb);

	}
	if (used > 0) fprintf(fp,"\n");
	endpass = i - ((i < to) ? 1 : 0);

	for (line=0; line<11; line++){
	    switch(line) {
	      case 0: {
		  if (used > 0) fprintf(fp,">> ");
		  fprintf(fp,"REF:  "); break; }
	      case 1: 
	      case 2: {	
		  if (BF_notSET(path->attrib,PA_REF_WTIMES)) continue;
		  if (used > 0) fprintf(fp,">> ");
		  if (line == 1) fprintf(fp,"R_T1: ");
		  if (line == 2) fprintf(fp,"R_T2: ");
		  break; }
  	      case 3: {
		  if (BF_notSET(path->attrib,PA_REF_CONF)) continue;
		  if (used > 0) fprintf(fp,">> ");
		  fprintf(fp,"RCNF: "); break; }
	      case 4: {
		  if (used > 0) fprintf(fp,">> ");
		  fprintf(fp,"HYP:  "); break; }	
	      case 5: 
	      case 6: {	
		  if (BF_notSET(path->attrib,PA_HYP_WTIMES)) continue;
		  if (used > 0) fprintf(fp,">> ");
		  if (line == 5) fprintf(fp,"H_T1: ");
		  if (line == 6) fprintf(fp,"H_T2: ");
		  break; }
	      case 7: {	
		  if (BF_notSET(path->attrib,PA_HYP_CONF)) continue;
		  if (used > 0) fprintf(fp,">> ");
		  fprintf(fp,"CONF: "); break; }
	      case 8: {	
		if (BF_notSET(path->attrib,PA_REF_WEIGHT)) continue;
		  if (used > 0) fprintf(fp,">> ");
		  fprintf(fp,"R_WE: ");
		  break; }
	      case 9: {	
		  if (BF_notSET(path->attrib,PA_HYP_WEIGHT)) continue;
		  if (used > 0) fprintf(fp,">> ");
		  fprintf(fp,"H_WE: ");		  
		  break; }
	      case 10: {
		  if (used > 0) fprintf(fp,">> ");
		  fprintf(fp,"Eval: "); break; }
	    }

	    /* loop through the words */
	    for (i=used; i<endpass && i<to; i++){
		if (path->pset[i].eval == P_INS)
		    lena = 1, astr = aster;
		else {
		    astr = ((WORD *)(path->pset[i].a_ptr))->value;
		    lena = TEXT_strlen(astr);
		}
		if (path->pset[i].eval == P_DEL)
		    lenb = 1, bstr = aster;
		else {
		    bstr = ((WORD *)(path->pset[i].b_ptr))->value;
		    lenb = TEXT_strlen(bstr);
		}
		wmax = MAX3(aux_max,lena,lenb);
		
		/* set the format statement and print the word */
		switch(line) {
		  case 0: {
		      sprintf(fmt,"%%-%ds ",wmax);
		      if (astr == aster) 
			  astr = aster + (alen - lenb);
		      else {
			  if (BF_notSET(path->attrib,PA_CASE_SENSE) &&
			      path->pset[i].eval != P_CORR) {
			      astr = TEXT_str_to_master(astr, 0);
			  }
		      }
		      fprintf(fp,fmt,astr);
		      break;
		  }
		  case 1:
		  case 2: {	
		      if ((line == 1) && (path->pset[i].eval != P_INS)){
			  sprintf(fmt,"%%-%d.2f ",wmax);
			  fprintf(fp,fmt,((WORD *)(path->pset[i].a_ptr))->T1);
		      } else if ((line == 2) && (path->pset[i].eval != P_INS)){
			  sprintf(fmt,"%%-%d.2f ",wmax);
			  fprintf(fp,fmt,((WORD *)(path->pset[i].a_ptr))->T2);
		      } else {
			  sprintf(fmt,"%%%ds ",wmax);
			  fprintf(fp,fmt,"");
		      }		    
		      break; 
		  }
		  case 3: {	
		      if (path->pset[i].eval != P_INS){
			  sprintf(fmt,"%%-%d.4f ",wmax);
			 fprintf(fp,fmt,((WORD *)(path->pset[i].a_ptr))->conf);
		      } else {
			  sprintf(fmt,"%%%ds ",wmax);
			  fprintf(fp,fmt,"");
		      }		    
		      break; 
		  }	
		  case 4: {	
		      sprintf(fmt,"%%-%ds ",wmax);
		      if (bstr == aster) 
			  bstr = aster + (alen - lena);
		      else {
			  if (BF_notSET(path->attrib,PA_CASE_SENSE) &&
			      path->pset[i].eval != P_CORR){
			      bstr = TEXT_str_to_master(bstr, 0);
			  }
		      }
		      fprintf(fp,fmt,bstr);
		      break;
		  }
		  case 5: 
		  case 6: {
		      if ((line == 5) && (path->pset[i].eval != P_DEL)){
			  sprintf(fmt,"%%-%d.2f ",wmax);
			  fprintf(fp,fmt,((WORD *)(path->pset[i].b_ptr))->T1);
		      } else if ((line == 6) && (path->pset[i].eval != P_DEL)){
			  sprintf(fmt,"%%-%d.2f ",wmax);
			  fprintf(fp,fmt,((WORD *)(path->pset[i].b_ptr))->T2);
		      } else {
			  sprintf(fmt,"%%%ds ",wmax);
			  fprintf(fp,fmt,"");
		      }		    
		      break; 
		  }
		  case 7: {	
		      if (path->pset[i].eval != P_DEL){
			  sprintf(fmt,"%%-%d.4f ",wmax);
			 fprintf(fp,fmt,((WORD *)(path->pset[i].b_ptr))->conf);
		      } else {
			  sprintf(fmt,"%%%ds ",wmax);
			  fprintf(fp,fmt,"");
		      }		    
		      break; 
		  }	
		  case 8: {	
		      if (path->pset[i].eval != P_INS){
			sprintf(fmt,"%%-%d.4f ",wmax);
			fprintf(fp,fmt,((WORD *)(path->pset[i].a_ptr))->weight);
		      } else {
			sprintf(fmt,"%%%ds ",wmax);
			fprintf(fp,fmt,"");
		      }		    
		      break; 
		  }
		  case 9: {	
		      if (path->pset[i].eval != P_DEL){
			  sprintf(fmt,"%%-%d.4f ",wmax);
			 fprintf(fp,fmt,((WORD *)(path->pset[i].b_ptr))->weight);
		      } else {
			  sprintf(fmt,"%%%ds ",wmax);
			  fprintf(fp,fmt,"");
		      }		    
		      break; 
		  }
		  case 10: {
		      sprintf(fmt,"%%-%ds ",wmax);
		      fprintf(fp,fmt,(astr == aster) ? "I" : 
			      ((bstr == aster) ? "D" : 
			       ((path->pset[i].eval != P_CORR) ? "S" : "")));
		      break;
		  }	
		}
	    }
	    fprintf(fp,"\n");
	}
	used = endpass;
    }
}



void PATH_print_html(PATH *path, FILE *fp, int max, int header){
   PATH_n_print_html(path, fp, 0, path->num, max, header);
}

void PATH_n_print_html(PATH *path, FILE *fp, int from, int to, int max, int header){
    int i, lena, lenb, alen=40, plen=0, endpass, used;
    int c,s,d,n,u,aux_max=0,line=0,wmax;
    char fmt[30];
    TEXT *astr, *bstr, 
         *aster=(TEXT *) "****************************************";

    if (to > path->num) to = path->num;
    used = (from >= 0 && from < to) ? from : 0;
    c = s = d = n = u = 0;

    /* collect preliminary info */
    for (i=used; i<to; i++)
	if ((path->pset[i].eval & P_CORR) != 0) c++;
	else if ((path->pset[i].eval & P_SUB) != 0) s++;
	else if ((path->pset[i].eval & P_INS) != 0) n++;
	else if ((path->pset[i].eval & P_DEL) != 0) d++;
	else u ++;
    /* compute the max of the auxillary info */
    if (BF_isSET(path->attrib,PA_HYP_CONF))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_REF_CONF))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_HYP_WTIMES))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    if (BF_isSET(path->attrib,PA_REF_WTIMES))
	aux_max = (aux_max < 7) ? 7 : aux_max;
    
    if (header){
      fprintf(fp,"<B>\n");
      fprintf(fp,"id: %s <BR>\n",(path->id == (char *)0) ? "" : path->id);
      if (from != 0 || to != path->num)
	fprintf(fp,"Range: %d-%d of %d words <BR>\n",from,to,path->num);
      if (path->labels != (char *)0)
	fprintf(fp,"Labels: %s<BR>\n",path->labels);
      if (u > 0)
	fprintf(fp,"Scores: (#C #S #D #I #UNK) %d %d %d %d %d<BR>\n",
		c,s,d,n,u);
      else
	fprintf(fp,"Scores: (#C #S #D #I) %d %d %d %d<BR>\n",c,s,d,n);	   
      
      if (BF_isSET(path->attrib,PA_CHAR_ALIGN) ||
	  BF_isSET(path->attrib,PA_CASE_SENSE)){
	fprintf(fp,"Attributes: ");
	fprintf(fp,"<UL>\n");
	if (BF_isSET(path->attrib,PA_CHAR_ALIGN))
	  fprintf(fp,"Character_align ");
	if (BF_isSET(path->attrib,PA_CASE_SENSE))
	  fprintf(fp,"Case_sensitve ");
	fprintf(fp,"\n");	
	fprintf(fp,"</UL>\n");
      }
      if (BF_isSET(path->attrib,PA_REF_TIMES))
	fprintf(fp,"Ref times: t1= %.2f t2= %.2f<BR>\n",
		path->ref_t1,path->ref_t2);
      if (BF_isSET(path->attrib,PA_HYP_TIMES))
	fprintf(fp,"Hyp times: t1= %.2f t2= %.2f<BR>\n",
		path->hyp_t1,path->hyp_t2);
      fprintf(fp,"</B>\n");
    }

    while (used < to){
	/* calculate the max words for this pass */
	plen = (used > 0) ? 9 : 6;
	for (i=used; plen < max && i<to; i++){
	    lena = (path->pset[i].eval == P_INS) ? 1 :
		TEXT_strlen(((WORD *)(path->pset[i].a_ptr))->value);
	    lenb = (path->pset[i].eval == P_DEL) ? 1 :
		TEXT_strlen(((WORD *)(path->pset[i].b_ptr))->value);
	    plen += 1 + MAX3(aux_max,lena,lenb);

	}
	if (used > 0) fprintf(fp,"\n");
	endpass = i - ((i < to) ? 1 : 0);

	fprintf(fp,"<TABLE CELLSPACING=0 CELLPADDING=1>\n");

	for (line=0; line<9; line++){
	    fprintf(fp,"<TR>\n");
	    /* write the header */
	    switch(line) {
	      case 0: {
		fprintf(fp,"<TD>");
		if (used > 0) fprintf(fp,">> ");
		fprintf(fp,"REF:");
		fprintf(fp,"</TD>\n");
		break; }
   	      case 1: 
	      case 2: {	
		  if (BF_notSET(path->attrib,PA_REF_WTIMES)) continue;
		  fprintf(fp,"<TD>");
		  if (used > 0) fprintf(fp,">> ");
		  if (line == 1) fprintf(fp,"R_T1:");
		  if (line == 2) fprintf(fp,"R_T2:");
		  fprintf(fp,"</TD>\n");
		  break; }
  	      case 3: {
		if (BF_notSET(path->attrib,PA_REF_CONF)) continue;
		fprintf(fp,"<TD>");
		if (used > 0) fprintf(fp,">> ");
		fprintf(fp,"RCNF:");
		fprintf(fp,"</TD>\n");
		break; }
	      case 4: {
		fprintf(fp,"<TD>");
		if (used > 0) fprintf(fp,">> ");
		fprintf(fp,"HYP:"); 
		fprintf(fp,"</TD>\n");
		break; }	
	      case 5: 
	      case 6: {	
		  if (BF_notSET(path->attrib,PA_HYP_WTIMES)) continue;
		  fprintf(fp,"<TD>");
		  if (used > 0) fprintf(fp,">> ");
		  if (line == 5) fprintf(fp,"H_T1:");
		  if (line == 6) fprintf(fp,"H_T2:");
		  fprintf(fp,"</TD>\n");
		  break; }
	      case 7: {	
		if (BF_notSET(path->attrib,PA_HYP_CONF)) continue;
		fprintf(fp,"<TD>");
		if (used > 0) fprintf(fp,">> ");
		fprintf(fp,"CONF:");
		fprintf(fp,"</TD>\n");
		break; 
	      }
	      case 8: {
		fprintf(fp,"<TD>");
		if (used > 0) fprintf(fp,">> ");
		fprintf(fp,"Eval: "); 
		fprintf(fp,"</TD>\n");
		break; }
	    }

	    /* loop through the words */
	   for (i=used; i<endpass && i<to; i++){
		if (path->pset[i].eval == P_INS)
		    lena = 1, astr = aster;
		else {
		    astr = ((WORD *)(path->pset[i].a_ptr))->value;
		    lena = TEXT_strlen(astr);
		}
		if (path->pset[i].eval == P_DEL)
		    lenb = 1, bstr = aster;
		else {
		    bstr = ((WORD *)(path->pset[i].b_ptr))->value;
		    lenb = TEXT_strlen(bstr);
		}
		wmax = MAX3(aux_max,lena,lenb);
		
		/* set the format statement and print the word */
		switch(line) {
		  case 0: {
		      sprintf(fmt,"<TD>%%-%ds</TD>",wmax);
		      if (astr == aster) 
			  astr = aster + (alen - lenb);
		      else {
			  if (BF_notSET(path->attrib,PA_CASE_SENSE) &&
			      path->pset[i].eval != P_CORR) {
			      astr = TEXT_str_to_master(astr, 0);
			  }
		      }
		      fprintf(fp,fmt,astr);
		      break;
		  }
		  case 1:
		  case 2: {	
		    sprintf(fmt,"<TD>%%.2f</TD>");
		      if ((line == 1) && (path->pset[i].eval != P_INS)){
			fprintf(fp,fmt,((WORD *)(path->pset[i].a_ptr))->T1);
		      } else if ((line == 2) && (path->pset[i].eval != P_INS)){
			fprintf(fp,fmt,((WORD *)(path->pset[i].a_ptr))->T2);
		      } else {
			sprintf(fmt,"<TD></TD>");
		      }		    
		      break; 
		  }
		  case 3: {	
		    if (path->pset[i].eval != P_INS){
		      fprintf(fp,"<TD>%.4f</TD>",
			      ((WORD *)(path->pset[i].a_ptr))->conf);
		      } else 
			fprintf(fp,"<TD></TD>");
		      break; 
		  }	
		  case 4: {	
		      sprintf(fmt,"<TD>%%s</TD>");
		      if (bstr == aster) 
			  bstr = aster + (alen - lena);
		      else {
			  if (BF_notSET(path->attrib,PA_CASE_SENSE) &&
			      path->pset[i].eval != P_CORR){
			      bstr = TEXT_str_to_master(bstr, 0);
			  }
		      }
		      fprintf(fp,fmt,bstr);
		      break;
		  }
		  case 5: 
		  case 6: {
		    sprintf(fmt,"<TD>%%.2f</TD>");
		    if ((line == 5) && (path->pset[i].eval != P_DEL)){
		      fprintf(fp,fmt,((WORD *)(path->pset[i].b_ptr))->T1);
		    } else if ((line == 6) && (path->pset[i].eval != P_DEL)){
		      fprintf(fp,fmt,((WORD *)(path->pset[i].b_ptr))->T2);
		    } else {
		      fprintf(fp,"<TD></TD>");
		    }		    
		    break; 
		  }
		  case 7: {	
		      if (path->pset[i].eval != P_DEL){
			fprintf(fp,"<TD>%.4f</TD>",
				((WORD *)(path->pset[i].b_ptr))->conf);
		      } else {
			fprintf(fp,"<TD></TD>");
		      }		    
		      break; 
		  }	
		  case 8: {
		      fprintf(fp,"<TD>%s</TD>",(astr == aster) ? "I" : 
			      ((bstr == aster) ? "D" : 
			       ((path->pset[i].eval != P_CORR) ? "S" : "")));
		      break;
		  }	
		}
	    }
	    fprintf(fp,"<TR>\n");
	}
	fprintf(fp,"</TABLE>\n");
	used = endpass;
    }
}

void PATH_print_wt(PATH *path, FILE *fp){
    int w;
    fprintf(fp,"Path dump id: %s  Labels: %s\n",
	    path->id, (path->labels!=(char*)0)?path->labels:"");
    for (w=0; w<path->num; w++){
	fprintf(fp,"%4d:  ",w);
	print_2_WORD_wt(path->pset[w].a_ptr,path->pset[w].b_ptr,fp);
    }
}

void PATH_increment_WORD_use(PATH *path){
    int i;

    for (i=0; i<path->num; i++){
	if (path->pset[i].a_ptr != (void *)0) 
	    ((WORD *)(path->pset[i].a_ptr))->use ++;
	if (path->pset[i].b_ptr != (void *)0) 
	    ((WORD *)(path->pset[i].b_ptr))->use ++;
    }
}

void PATH_free(PATH *path){
    int i;

    for (i=0; i<path->num; i++){
	if (path->pset[i].a_ptr != (void *)0) 
	    release_WORD((WORD *)(path->pset[i].a_ptr));
	if (path->pset[i].b_ptr != (void *)0) 
	    release_WORD((WORD *)(path->pset[i].b_ptr));
    }
    if (path->id != (char *)0)  free_singarr(path->id,char);
    if (path->labels != (char *)0)  free_singarr(path->labels,char);
    if (path->file != (char *)0)  free_singarr(path->file,char);
    if (path->channel != (char *)0)  free_singarr(path->channel,char);
    free_singarr(path->pset,PATH_SET);
    free_singarr(path, PATH);
}

void PATH_append(PATH *path, void *ap, void *bp, int eval)
{
    check_space_in_PATH(path);
    path->pset[path->num].a_ptr = ap;
    path->pset[path->num].b_ptr = bp;
    path->pset[path->num].eval  = eval;
    path->num++;
}

void PATH_remove(PATH *path)
{
    if (path->num > 0)
	path->num--;
}

/******************************************************/
/*  sort the Insertions and Deletions in the sentence */
/*  by the time in which they occure.                 */
/******************************************************/
#define SW(_a1,_a2,_tt) {_tt _t=_a1; _a1=_a2; _a2=_t;}
void sort_PATH_time_marks(PATH *path)
{
    int x=0, tx, change;
    double tb, ta;
    
    /* Sort the insertions and deletions, so that later down the */
    /* road split and merges are properly done.                  */
    while (x < path->num){
        while (x < path->num &&
	       (! ((path->pset[x].eval == P_DEL) ||
		   (path->pset[x].eval == P_INS))))
            x++;
        /* printf("Located In/Del  %d\n",x); */
        /* word at x is either EOF or INS or del */
        change = TRUE;
        while ( x < path->num && change){
            change = FALSE;
            tx = x+1;
            while (tx < path->num && 
                   (path->pset[tx].eval == P_DEL ||
                    path->pset[tx].eval == P_INS)){
		/* printf("inner loop\n"); */
                ta = (path->pset[tx-1].eval == P_DEL) ?
                    ((WORD *)(path->pset[tx-1].a_ptr))->T1 :
			((WORD *)(path->pset[tx-1].b_ptr))->T1;
		tb = (path->pset[tx].eval == P_DEL) ?
                    ((WORD *)(path->pset[tx].a_ptr))->T1 :
			((WORD *)(path->pset[tx].b_ptr))->T1;
                /* printf("ta=%f tb=%f\n",ta,tb); */
                if (ta > tb) {
                    /* printf("Swapping \n"); */
                    /* I need to swap */
		    SW(path->pset[tx],path->pset[tx-1],PATH_SET);
                    change = TRUE;
                }
                tx++;
            }
        }
        while (x < path->num &&
	       (path->pset[x].eval == P_DEL  ||
		path->pset[x].eval == P_INS))
            x++;
    }
    /* printf("Sort done\n"); */
}

