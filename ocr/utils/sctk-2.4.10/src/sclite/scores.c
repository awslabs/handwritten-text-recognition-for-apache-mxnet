#include "sctk.h"

/**********************************************************************/
/*                                                                    */
/**********************************************************************/
void dump_SCORES(SCORES *sc, FILE *fp){
    int i;

    fprintf(fp,"Dump of SCORES:\n");
    fprintf(fp,"   Title: %s\n\n",sc->title);
    for (i=0; i<sc->num_grp; i++){
	fprintf(fp,"%8s   NS:%d  SE:%d  ",sc->grp[i].name,
		sc->grp[i].nsent,sc->grp[i].serr);
	fprintf(fp,"C:%d  S:%d  D:%d  I:%d  M:%d  P:%d\n",
		sc->grp[i].corr  ,
		sc->grp[i].sub   ,
		sc->grp[i].del   ,
		sc->grp[i].ins   ,
		sc->grp[i].merges,
		sc->grp[i].splits);
    }
}

void dump_SCORES_alignments(SCORES *sc, FILE *fp, int lw, int full){
    int i, p;
    
    if (full){
	fprintf(fp,"NIST_TEXT_ALIGNMENT\n");
	fprintf(fp,"VERSION 0.1\n");
    }
    fprintf(fp,"\n\n\t\tDUMP OF SYSTEM ALIGNMENT STRUCTURE\n\n");
    fprintf(fp,"System name:   %s\n",sc->title);
    if (full) {
	if (sc->ref_fname != NULL) 
	  fprintf(fp,"Ref file:      %s\n",sc->ref_fname);
	if (sc->hyp_fname != NULL) 
	  fprintf(fp,"Hyp file:      %s\n",sc->hyp_fname);
	if (sc->creation_date != NULL)
	  fprintf(fp,"Creation date: \"%s\"\n",sc->creation_date);
	if (sc->weight_ali)
	  fprintf(fp,"Word Weight Aligned by file: \"%s\"\n",sc->weight_file);
	if (sc->frag_corr)
	  fprintf(fp,"Fragment Correct Flag Set\n");
	if (sc->opt_del)
	  fprintf(fp,"Optionally Deletable Flag Set\n");
    }

    fprintf(fp,"\n");
    if (full) fprintf(fp,"Speaker Count: %d\n",sc->num_grp);

    fprintf(fp,"Speakers: \n");
    for (i=0; i<sc->num_grp; i++)
        fprintf(fp,"   %2d:  %s\n",i,sc->grp[i].name);
    fprintf(fp,"\n");

    if (full){
        PATH *pathx;
	if (sc->aset.num_plab > 0 || sc->aset.num_cat > 0){
	    fprintf(fp,"Utterance Label definitions:\n");
	    for (i=0; i<sc->aset.num_cat; i++)
		fprintf(fp,"    Category: id: \"%s\" title: \"%s\" "
			"description: \"%s\"\n",sc->aset.cat[i].id,
			sc->aset.cat[i].title,sc->aset.cat[i].desc);
	    for (i=0; i<sc->aset.num_plab; i++)
		fprintf(fp,"    Label: id: \"%s\" title: \"%s\" "
			"description: \"%s\"\n",sc->aset.plab[i].id,
			sc->aset.plab[i].title,sc->aset.plab[i].desc);
	}
	fprintf(fp,"\n");

	/* set pathx to the first avaliable path */
	for (i=0, pathx = (PATH *)0; i<sc->num_grp; i++)
	  if (sc->grp[i].num_path > 0){
	    pathx = sc->grp[i].path[0];
	    break;
	  }
	if (pathx == (PATH*)0) { 
	  fprintf(fp,"No utterances found\n");
	  return;
	}
	if (pathx->sequence >= 0){
	    /* sort the output to be order by aligment sequence */
	    int *curpath;
	    int nextgrp;
	    int minseq;
	    alloc_singZ(curpath,sc->num_grp,int,0);
	    
	    do {
	      minseq = 999999;
	      /* search through the groups, lookin for the lowest begintime */
	      nextgrp = -1;
	      for (i=0; i<sc->num_grp; i++)
		if (curpath[i] < sc->grp[i].num_path)
		  if (sc->grp[i].path[curpath[i]]->sequence < minseq){
		    minseq = sc->grp[i].path[curpath[i]]->sequence;
		    nextgrp = i;
		  }
	      
	      if (nextgrp >= 0) { /* print it */
		  i = nextgrp;
		  fprintf(fp,"Speaker sentences%4d:  %s   utt# %d of %d\n",i,
			  sc->grp[i].name,curpath[i],sc->grp[i].nsent);
		  PATH_print(sc->grp[i].path[curpath[i]],fp,lw);
		  fprintf(fp,"\n");
		  curpath[nextgrp] ++;
	      }
	    } while (nextgrp >= 0);
	    free_singarr(curpath,int);
	} else {
	    for (i=0; i<sc->num_grp; i++){
	        fprintf(fp,"Speaker sentences%4d:  %s   #utts: %d\n",i,
			sc->grp[i].name,sc->grp[i].nsent);
		for (p=0; p<sc->grp[i].num_path; p++){
		    PATH_print(sc->grp[i].path[p],fp,lw);
		    fprintf(fp,"\n");
		}
	    }
	}
    } else {
        for (i=0; i<sc->num_grp; i++){
	    int attrib = PA_NONE;
	    fprintf(fp,"Speaker sentences%4d:  %s   #utts: %d\n",i,
		    sc->grp[i].name,sc->grp[i].nsent);
	    for (p=0; p<sc->grp[i].num_path; p++){
		attrib = sc->grp[i].path[p]->attrib;
		sc->grp[i].path[p]->attrib = PA_NONE;
		/* retain the case sensitive attribute */
		if (BF_isSET(attrib,PA_CASE_SENSE))
  		    BF_SET(sc->grp[i].path[p]->attrib,PA_CASE_SENSE);
		PATH_print(sc->grp[i].path[p],fp,lw);
		sc->grp[i].path[p]->attrib = attrib;
		fprintf(fp,"\n");
	    }
	}
    }
    fprintf(fp,"\n");
}

TEXT *formatWordForSGML(WORD *word, TEXT *buffer){
  TEXT_strcpy_escaped(buffer, word->value, WORD_SGML_SUB_WORD_SEP_CHR);

  if (word->tag1 == (TEXT *)0 && word->tag2 == (TEXT *)0)
    return buffer;

  TEXT_strcpy(buffer + TEXT_strlen(buffer), (TEXT *)WORD_SGML_SUB_WORD_SEP_STR);
  TEXT_strcpy_escaped(buffer + TEXT_strlen(buffer), 
		      (word->tag1 != (TEXT *)0) ? word->tag1 : (TEXT *)"", WORD_SGML_SUB_WORD_SEP_CHR);
  TEXT_strcpy(buffer + TEXT_strlen(buffer), (TEXT *)WORD_SGML_SUB_WORD_SEP_STR);
  TEXT_strcpy_escaped(buffer + TEXT_strlen(buffer), 
		      (word->tag2 != (TEXT *)0) ? word->tag2 : (TEXT *)"", WORD_SGML_SUB_WORD_SEP_CHR);
  return buffer;
}

void dump_SCORES_sgml(SCORES *sc, FILE *fp, TEXT *token_separator, TEXT *token_attribute_separator){
    int i, p, w;
    TEXT bufA[1000];
    TEXT bufB[1000];

    fprintf(fp,"<SYSTEM title=\"%s\"",sc->title);
    fprintf(fp," ref_fname=\"%s\"",sc->ref_fname);
    fprintf(fp," hyp_fname=\"%s\"",sc->hyp_fname);
    fprintf(fp," creation_date=\"%s\"",sc->creation_date);
    fprintf(fp," format=\"2.4\"");
    fprintf(fp," frag_corr=\"%s\"", sc->frag_corr ? "TRUE" : "FALSE");
    fprintf(fp," opt_del=\"%s\"", sc->opt_del ? "TRUE" : "FALSE");
    fprintf(fp," weight_ali=\"%s\"", sc->weight_ali ? "TRUE" : "FALSE");
    fprintf(fp," weight_filename=\"%s\"", sc->weight_ali ? sc->weight_file : "");
    fprintf(fp,">\n");

    for (i=0; i<sc->aset.num_plab; i++)
	fprintf(fp,"<LABEL id=\"%s\" title=\"%s\" desc=\"%s\">\n</LABEL>\n",
		sc->aset.plab[i].id,
		sc->aset.plab[i].title,
		sc->aset.plab[i].desc);

    for (i=0; i<sc->aset.num_cat; i++)
	fprintf(fp,"<CATEGORY id=\"%s\" title=\"%s\" desc=\"%s\">\n</CATEGORY>\n",
		sc->aset.cat[i].id,
		sc->aset.cat[i].title,
		sc->aset.cat[i].desc);

    for (i=0; i<sc->num_grp; i++){
	fprintf(fp,"<SPEAKER id=\"%s\">\n",sc->grp[i].name);
	for (p=0; p<sc->grp[i].num_path; p++){
	    PATH *path = sc->grp[i].path[p];

	    fprintf(fp,"<PATH id=\"%s\" word_cnt=\"%d\"",
		    path->id, path->num);
	    if (path->labels != (char *)0)
		fprintf(fp," labels=\"%s\"",path->labels);
	    if (path->file != (char *)0)
		fprintf(fp," file=\"%s\"",path->file);
	    if (path->channel != (char *)0)
		fprintf(fp," channel=\"%s\"",path->channel);
	    fprintf(fp," sequence=\"%d\"",path->sequence);
	    if (path->attrib != PA_NONE){
		if (BF_isSET(path->attrib,PA_REF_TIMES)){
		    fprintf(fp," R_T1=\"%.3f\"",path->ref_t1);
		    fprintf(fp," R_T2=\"%.3f\"",path->ref_t2);
		}
		if (BF_isSET(path->attrib,PA_HYP_TIMES)){
		    fprintf(fp," H_T1=\"%.3f\"",path->hyp_t1);
		    fprintf(fp," H_T2=\"%.3f\"",path->hyp_t2);
		}
		if (BF_isSET(path->attrib,PA_CHAR_ALIGN))
		    fprintf(fp," char_align=\"1\"");
		if (BF_isSET(path->attrib,PA_CASE_SENSE))
		    fprintf(fp," case_sense=\"1\"");	
		if (BF_isSET(path->attrib,PA_HYP_WEIGHT))
		    fprintf(fp," hyp_weight=\"1\"");	
		if (BF_isSET(path->attrib,PA_REF_WEIGHT))
		    fprintf(fp," ref_weight=\"1\"");	
		if (BF_isSET(path->attrib,PA_REF_WTIMES) ||
		    BF_isSET(path->attrib,PA_HYP_WTIMES) ||
		    BF_isSET(path->attrib,PA_REF_CONF) ||
		    BF_isSET(path->attrib,PA_HYP_CONF) ||
		    BF_isSET(path->attrib,PA_REF_WEIGHT) ||
		    BF_isSET(path->attrib,PA_HYP_WEIGHT)){
		    int f=0;
		    fprintf(fp," word_aux=\"");
		    if (BF_isSET(path->attrib,PA_REF_WTIMES)){
			fprintf(fp,"r_t1+t2");
			f++;
		    }
		    if (BF_isSET(path->attrib,PA_HYP_WTIMES)){
			if (f++ != 0) fprintf(fp,",");
			fprintf(fp,"h_t1+t2");
		    }
		    if (BF_isSET(path->attrib,PA_REF_CONF)){
			if (f++ != 0) fprintf(fp,",");
			fprintf(fp,"r_conf");
		    }
		    if (BF_isSET(path->attrib,PA_HYP_CONF)){
			if (f++ != 0) fprintf(fp,",");
			fprintf(fp,"h_conf");
		    }
		    if (BF_isSET(path->attrib,PA_REF_WEIGHT)){
			if (f++ != 0) fprintf(fp,",");
			fprintf(fp,"r_weight");
		    }
		    if (BF_isSET(path->attrib,PA_HYP_WEIGHT)){
			if (f++ != 0) fprintf(fp,",");
			fprintf(fp,"h_weight");
		    }
		    fprintf(fp,"\"");
		}
	    }
	    fprintf(fp,">\n");

	    /* current format of paths:
	       PATH :== <ALIGN>[:<ALIGN>]*
	       ALIGN :== <EVAL>,<WORD>[,<WORD>]<AUX_FIELDS>
	       EVAL :== S | C | D | I | M | T | U 
	       WORD :== "<TEXT>"
	       */
	    for (w=0; w<path->num; w++) {
		if (path->pset[w].eval == P_INS)
		  fprintf(fp,"I%s%s\"%s\"", token_attribute_separator, token_attribute_separator, 
			  formatWordForSGML((WORD *)path->pset[w].b_ptr, bufB));
		else if (path->pset[w].eval == P_DEL)
		    fprintf(fp,"D%s\"%s\"%s",token_attribute_separator, 
			    formatWordForSGML((WORD *)path->pset[w].a_ptr, bufA),
			    token_attribute_separator);
		else if (path->pset[w].eval == P_CORR)
		    fprintf(fp,"C%s\"%s\"%s\"%s\"",
			    token_attribute_separator, 
			    formatWordForSGML((WORD *)path->pset[w].a_ptr, bufA),
			    token_attribute_separator, 
			    formatWordForSGML((WORD *)path->pset[w].b_ptr, bufB));
		else {
		    fprintf(fp,"%c%s\"%s\"%s\"%s\"",			    
			    ((path->pset[w].eval == P_SUB) ? 'S' :
			     ((path->pset[w].eval == P_MRG) ? 'M' : 
			      ((path->pset[w].eval == P_SPL) ? 'T' : 'U'))),
			    token_attribute_separator, 
			    formatWordForSGML((WORD *)path->pset[w].a_ptr, bufA),
			    token_attribute_separator, 
		            formatWordForSGML((WORD *)path->pset[w].b_ptr, bufB));
		}
		/* Extra attributes */
		if (BF_isSET(path->attrib,PA_REF_WTIMES))
		    if (path->pset[w].eval != P_INS)
			fprintf(fp,"%s%.3f+%.3f",
				token_attribute_separator, 
				((WORD *)path->pset[w].a_ptr)->T1,
				((WORD *)path->pset[w].a_ptr)->T2);
		    else
			fprintf(fp,"%s",token_attribute_separator);
		if (BF_isSET(path->attrib,PA_HYP_WTIMES))
		    if (path->pset[w].eval != P_DEL)
			fprintf(fp,"%s%.3f+%.3f",
				token_attribute_separator, 
				((WORD *)path->pset[w].b_ptr)->T1,
				((WORD *)path->pset[w].b_ptr)->T2);
		    else
			fprintf(fp,"%s",token_attribute_separator);
		if (BF_isSET(path->attrib,PA_REF_CONF))
		    if (path->pset[w].eval != P_INS)
			fprintf(fp,"%s%f",
				token_attribute_separator,
				((WORD *)path->pset[w].a_ptr)->conf);
		    else
			fprintf(fp,"%s",token_attribute_separator);
		if (BF_isSET(path->attrib,PA_HYP_CONF))
		    if (path->pset[w].eval != P_DEL)
			fprintf(fp,"%s%f",
				token_attribute_separator, 
				((WORD *)path->pset[w].b_ptr)->conf);
		    else
			fprintf(fp,"%s",token_attribute_separator);
		if (BF_isSET(path->attrib,PA_REF_WEIGHT))
		    if (path->pset[w].eval != P_INS)
			fprintf(fp,"%s%f",
				token_attribute_separator, 
				((WORD *)path->pset[w].a_ptr)->weight);
		    else
			fprintf(fp,"%s",token_attribute_separator);
		if (BF_isSET(path->attrib,PA_HYP_WEIGHT))
		    if (path->pset[w].eval != P_DEL)
			fprintf(fp,"%s%f",
				token_attribute_separator, 
				((WORD *)path->pset[w].b_ptr)->weight);
		    else
			fprintf(fp,"%s",token_attribute_separator);

		if (w < path->num-1) fprintf(fp,"%s",token_separator);
	    }
	    fprintf(fp,"\n</PATH>\n");
	}
	fprintf(fp,"</SPEAKER>\n");
    }

    fprintf(fp,"</SYSTEM>\n");
}
 
int load_SCORES_sgml(FILE *fp, SCORES **scor, int *nscor, int maxn)
{
    char *proc="load_SCORES_sgml", *msg;
    int buf_len=100;
    TEXT *buf, *token;
    SGML sg;
    SCORES *tscor=(SCORES *)0;
    PATH *path = (PATH *)0;
    int spk=(-1), dbg = 0;
    int word_aux_fields[50];
    int num_word_aux = 0, max_word_aux = 50;

    alloc_singZ(buf,buf_len,TEXT,'\0');
    init_SGML(&sg);

	while (!feof(fp) && TEXT_ensure_fgets(&buf,&buf_len,fp) != NULL){
	if (*buf == '<' && *(buf+1) != '/'){
	    if (dbg) fprintf(scfp,"%s: Begin %s",proc,buf);
	    if (! add_SGML_tag(&sg,buf)) {
	      msg = rsprintf("Unable to parse SGML tag '%s'",buf);
		goto FAIL;
	    }
	    if (dbg) dump_SGML_tag(&sg,sg.num_tags-1,scfp);

	    /* switch on the reasons to begin */

	    if (TEXT_strcmp(sg.tags[sg.num_tags-1].name,(TEXT *)"SYSTEM")== 0){
		TEXT *sga; 

		if (maxn <= *nscor){
		    msg = rsprintf("SCORE array too small, increase size\n");
		    goto FAIL;
		}

		tscor = SCORES_init((char *)
				    get_SGML_attrib(&sg,sg.num_tags-1,
						    (TEXT *)"title"),10);
		sga = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"ref_fname");
		if (TEXT_strcmp(sga,(TEXT *)"") != 0)
		    tscor->ref_fname = (char *)TEXT_strdup(sga);

		sga = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"hyp_fname");
		if (TEXT_strcmp(sga,(TEXT *)"") != 0)
		    tscor->hyp_fname = (char *)TEXT_strdup(sga);

		sga = get_SGML_attrib(&sg,sg.num_tags-1,
				      (TEXT *)"creation_date");
		if (TEXT_strcmp(sga,(TEXT *)"") != 0)
		    tscor->creation_date = (char *)TEXT_strdup(sga);
		
		sga = get_SGML_attrib(&sg,sg.num_tags-1,
				      (TEXT *)"frag_corr");
		if (TEXT_strcmp(sga,(TEXT *)"TRUE") == 0)
		  tscor->frag_corr = TRUE;
		else
		  tscor->frag_corr = FALSE;
		
		sga = get_SGML_attrib(&sg,sg.num_tags-1,
				      (TEXT *)"opt_del");
		if (TEXT_strcmp(sga,(TEXT *)"TRUE") == 0)
		  tscor->opt_del = TRUE;
		else
		  tscor->opt_del = FALSE;

		sga = get_SGML_attrib(&sg,sg.num_tags-1,
				      (TEXT *)"weight_ali");
		if (TEXT_strcmp(sga,(TEXT *)"TRUE") == 0){
		  tscor->weight_ali = TRUE;
		  sga = get_SGML_attrib(&sg,sg.num_tags-1,
					(TEXT *)"weight_filename");
		  tscor->weight_file = (char *)TEXT_strdup(sga);
		} else
		  tscor->weight_ali = FALSE;
		
		scor[(*nscor)++] = tscor;
		spk = (-1);
	    } else if (TEXT_strcmp(sg.tags[sg.num_tags-1].name,
				   (TEXT *)"SPEAKER") ==0){
		if (tscor == (SCORES *)0) {
		    msg = rsprintf("SPEAKER tag '%s' found outside%s",
				   buf," of SYSTEM tag");

		    goto FAIL;
		}
		spk = SCORES_get_grp(tscor,(char *)
				           get_SGML_attrib(&sg,sg.num_tags-1,
							   (TEXT *)"id"));
	    } else if (TEXT_strcmp(sg.tags[sg.num_tags-1].name,
				   (TEXT *)"PATH") == 0){
		TEXT *l;
		if (spk == (-1)){
		    msg = rsprintf("PATH tag '%s' found outside%s",
				   buf," of SPEAKER tag");
		    goto FAIL;
		}
		path = PATH_alloc(atof((char*)
				       get_SGML_attrib(&sg,sg.num_tags-1,
						       (TEXT*)"word_cnt")));
		path->num = 0;
		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"labels");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		       PATH_add_label(path,(char *)l);

		PATH_add_utt_id(path,(char *)get_SGML_attrib(&sg,sg.num_tags-1,
							     (TEXT *)"id"));

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"file");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		       PATH_add_file(path,(char *)l);

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"channel");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		       PATH_add_channel(path,(char *)l);

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"sequence");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		    path->sequence = atoi((char *)l);
		else
		    path->sequence = -1;

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"R_T1");
		if (TEXT_strcmp(l,(TEXT *)"") != 0){
		    path->ref_t1 = atof((char *)l);
		    BF_SET(path->attrib,PA_REF_TIMES);
		}

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"R_T2");
		if (TEXT_strcmp(l,(TEXT *)"") != 0) {
		    path->ref_t2 = atof((char *)l);
		    BF_SET(path->attrib,PA_REF_TIMES);
		}
		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"H_T1");
		if (TEXT_strcmp(l,(TEXT *)"") != 0) {
		    path->hyp_t1 = atof((char *)l);
		    BF_SET(path->attrib,PA_HYP_TIMES);
		}
		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"H_T2");
		if (TEXT_strcmp(l,(TEXT *)"") != 0) {
		    path->hyp_t2 = atof((char *)l);
		    BF_SET(path->attrib,PA_HYP_TIMES);
		}
		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"char_align");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		    BF_SET(path->attrib,PA_CHAR_ALIGN);

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"case_sense");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		    BF_SET(path->attrib,PA_CASE_SENSE);

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"hyp_weight");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		    BF_SET(path->attrib,PA_HYP_WEIGHT);

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"ref_weight");
		if (TEXT_strcmp(l,(TEXT *)"") != 0)
		    BF_SET(path->attrib,PA_REF_WEIGHT);

		l = get_SGML_attrib(&sg,sg.num_tags-1,(TEXT *)"word_aux");
		if (TEXT_strcmp(l,(TEXT *)"") != 0){
		    TEXT *tp = l, *tp2;
		    /* now parse the information in the line */
		    num_word_aux = 0;
		    while (*tp != '\0'){
			if (num_word_aux == max_word_aux) {
			    msg = rsprintf("Too many auxillary word items"
					   " > %d, expand 'max_word_aux'"
					   " and recompile",max_word_aux);
			    goto FAIL;
			}
			if ((tp2 = TEXT_strchr(tp,','))==NULL_TEXT){
			    tp2 = tp + TEXT_strlen(tp);
			}
			if (dbg) fprintf(scfp,"%s: Aux str '%s'\n",proc,tp);
			if (TEXT_strCcmp(tp,(TEXT *)"r_t1+t2",7) == 0){
			    BF_SET(path->attrib,PA_REF_WTIMES);
			    word_aux_fields[num_word_aux++] = PA_REF_WTIMES;
			} else if (TEXT_strCcmp(tp,(TEXT *)"h_t1+t2",7) == 0){
			    BF_SET(path->attrib,PA_HYP_WTIMES);
			    word_aux_fields[num_word_aux++] = PA_HYP_WTIMES;
			} else if (TEXT_strCcmp(tp,(TEXT *)"r_conf",6) == 0){
			    BF_SET(path->attrib,PA_REF_CONF);
			    word_aux_fields[num_word_aux++] = PA_REF_CONF;
			} else if (TEXT_strCcmp(tp,(TEXT *)"h_conf",6) == 0){
			    BF_SET(path->attrib,PA_HYP_CONF);
			    word_aux_fields[num_word_aux++] = PA_HYP_CONF;
			} else if (TEXT_strCcmp(tp,(TEXT *)"r_weight",8) == 0){
			    BF_SET(path->attrib,PA_REF_WEIGHT);
			    word_aux_fields[num_word_aux++] = PA_REF_WEIGHT;
			} else if (TEXT_strCcmp(tp,(TEXT *)"h_weight",8) == 0){
			    BF_SET(path->attrib,PA_HYP_WEIGHT);
			    word_aux_fields[num_word_aux++] = PA_HYP_WEIGHT;
			} else if (TEXT_strCcmp(tp,(TEXT *)"h_spkr",6) == 0){
			    word_aux_fields[num_word_aux++] = PA_HYP_SPKR;
			} else if (TEXT_strCcmp(tp,(TEXT *)"h_isSpkrSub",11) == 0){
                            word_aux_fields[num_word_aux++] = PA_HYP_ISSPKRSUB;
			} else {
			    fprintf(scfp,"Warning: Unknown word auxillary inf"
				    "o type '%s'\n",tp);
			}
			if (*(tp=tp2) == ',') tp++;
		    }
		}
	    } else if ((TEXT_strcmp(sg.tags[sg.num_tags-1].name,
				   (TEXT *)"LABEL") == 0) ||
		       (TEXT_strcmp(sg.tags[sg.num_tags-1].name,
				   (TEXT *)"CATEGORY") == 0)){
		char *x;
		x = rsprintf(";; %s id=\"%s\" title=\"%s\" desc=\"%s\"\n",
			     sg.tags[sg.num_tags-1].name,
			     get_SGML_attrib(&sg,sg.num_tags-1,(TEXT*)"id"),
			     get_SGML_attrib(&sg,sg.num_tags-1,(TEXT*)"title"),
			     get_SGML_attrib(&sg,sg.num_tags-1,(TEXT*)"desc"));
		if (tscor == (SCORES *)0)
		    fprintf(stderr,"Warning: %s tag '%s' found outside of SYSTEM flag\n",
			    sg.tags[sg.num_tags-1].name, buf);
		else
		    parse_input_comment_line(tscor, (TEXT *)x);
	    }
	} else if (*buf == '<' && *(buf+1) == '/'){
	    TEXT *endtag;
	    if (dbg) fprintf(scfp,"%s: End   %s",proc,buf);
	    if ((endtag = delete_SGML_tag(&sg,buf)) == NULL){
		msg = rsprintf("Unable to delete SGML end tag '%s'",buf);
		goto FAIL;
	    }

	    /* switch on the reasons to END */
	    if (TEXT_strcmp(endtag,(TEXT *)"SYSTEM") == 0){
		tscor = (SCORES *)0;
		spk = (-1);
		path = (PATH *)0;
	    } else if (TEXT_strcmp(endtag,(TEXT *)"SPEAKER") ==0){
		spk = (-1);
		path = (PATH *)0;
	    } else if (TEXT_strcmp(endtag,(TEXT *)"PATH") == 0) {
		add_PATH_score(tscor,path,spk,1);
		path = (PATH *)0;
	    }
	    free_singarr(endtag,TEXT);
	} else if (path != (PATH *)0){
	    WORD *wd1, *wd2;
	    TEXT *p1, *p2, *p3;
	    if (dbg) fprintf(scfp,"%s: DATA %s",proc,buf);
	    token = TEXT_strqtok(buf, (TEXT *)":\n");
	    while (token != (TEXT *)NULL){
		if (dbg) fprintf(scfp,"%s:     token %s\n",proc,token);
		
		/* pick of the evaluation flag */
		switch (*token){
		  case 'I': { path->pset[path->num].eval = P_INS;  break; }
		  case 'D': { path->pset[path->num].eval = P_DEL;  break; }
		  case 'S': { path->pset[path->num].eval = P_SUB;  break; }
		  case 'C': { path->pset[path->num].eval = P_CORR; break; }
		  case 'M': { path->pset[path->num].eval = P_MRG;  break; }
		  case 'T': { path->pset[path->num].eval = P_SPL;  break; }
		  default:  { 
		      msg = rsprintf("Unknown Alignment type '%c'",*token);
		      goto FAIL;
		  }
		}

		/* allocate the memory */
		path->pset[path->num].a_ptr = wd1 = NULL_WORD;
		path->pset[path->num].b_ptr = wd2 = NULL_WORD;
		if (path->pset[path->num].eval != P_INS){
		  wd1 = new_WORD(NULL_TEXT, -1, 0.0, 0.0, 0.0, NULL_TEXT, NULL_TEXT,
				 0, 0, -1.0);
		  path->pset[path->num].a_ptr = wd1;
		}
		if (path->pset[path->num].eval != P_DEL){
		  wd2 = new_WORD(NULL_TEXT, -1, 0.0, 0.0, 0.0, NULL_TEXT, NULL_TEXT,
				 0, 0, -1.0);
		  path->pset[path->num].b_ptr = wd2;
		}
		    
		/* begin parsing elements of the line */
		p1 = token + 1;  /* skip over the evaluation id */

		/* extract the first word */
		if (path->pset[path->num].eval == P_INS){
		    /* skip the the first element */
		    if (*(p1+1) != ','){
			msg = rsprintf("Word parse error, at %s of %s",
				       p1,token);
			goto FAIL;
		    }
		    p1++;
		} else {
		    if (*(p1+1) != '"') {
			msg = rsprintf("Word parse error, at %s of %s",
				       p1,token);
			goto FAIL;
		    }
		    p2 = TEXT_strchr(p1+2,'"');
		    wd1->value = TEXT_strBdup(p1+2,p2 - (p1+2));
		    p1 = p2+1;
		    if (dbg)
			fprintf(scfp,"%s: first word '%s'\n",proc,wd1->value);
		}

		/* extract the second word */
		if (path->pset[path->num].eval == P_DEL){
		    /* skip the the second element */
		    if (*(++p1) != ',' && *p1 != '\0'){
			msg = rsprintf("Word parse error, at %s",token);
			goto FAIL;
		    }
		} else {
		    if (*(p1+1) != '"') {
			msg = rsprintf("Word parse error, at %s",token);
			goto FAIL;
		    }
		    p2 = TEXT_strchr(p1+2,'"');
		    wd2->value = TEXT_strBdup(p1+2,p2 - (p1+2));
		    p1 = p2+1;
		    if (dbg) fprintf(scfp,"%s: second word '%s'\n",
				     proc,wd2->value);
		}

		if (*p1 != '\0' && num_word_aux == 0){
		    fprintf(scfp,"Warning: Auxillary word values exist, but no"
			    "ne have been defined in the 'word_aux' field\n");
		}
		if (num_word_aux > 0){
		    int aux_val = 0;
		    while (*p1 != '\0') {
			if (aux_val == num_word_aux){
			    fprintf(scfp,"Warning: More auxillary word values "
				    "than defined\n");
			}
			/* handle the auxilliary values */
			if ((p2 = TEXT_strchr(p1+1,',')) == NULL_TEXT)
			    p2 = p1 + TEXT_strlen(p1);
			if (p2 == p1+1) {
			    p1++; /* an empty item */
			} else {
			    TEXT save = *p2;
			    *p2 = '\0';
			    switch(word_aux_fields[aux_val]){
			      case PA_REF_WTIMES:
				if (path->pset[path->num].eval != P_INS){
				    if ((p3 = TEXT_strchr(p1,'+'))==NULL_TEXT){
					msg = rsprintf("Illegal auxillary time"
						   " format '%s'",p1);
				    }
				    *p3 = '\0';
				    ((WORD *)path->pset[path->num].a_ptr)->T1 =
					atof((char *)p1+1);
				    ((WORD *)path->pset[path->num].a_ptr)->T2 =
					atof((char *)p3+1);
				    *p3 = '+';
				}
				break;
			      case PA_HYP_WTIMES:	
				if (path->pset[path->num].eval != P_DEL){
				    if ((p3 = TEXT_strchr(p1,'+'))==NULL_TEXT){
					msg = rsprintf("Illegal auxillary time"
						   " format '%s'",p1);
				    }
				    *p3 = '\0';
				    ((WORD *)path->pset[path->num].b_ptr)->T1 =
					atof((char *)p1+1);
				    ((WORD *)path->pset[path->num].b_ptr)->T2 =
					atof((char *)p3+1);
				    *p3 = '+';
				}
				break;
			      case PA_REF_CONF:
				if (path->pset[path->num].eval != P_INS)
				    ((WORD *)path->pset[path->num].a_ptr)->conf
					= atof((char *)p1+1);
				break;
			      case PA_HYP_CONF:
				if (path->pset[path->num].eval != P_DEL)
				    ((WORD *)path->pset[path->num].b_ptr)->conf
					= atof((char *)p1+1);
				break;
			      case PA_REF_WEIGHT:
				if (path->pset[path->num].eval != P_INS)
				    ((WORD *)path->pset[path->num].a_ptr)->weight
					= atof((char *)p1+1);
				break;
			      case PA_HYP_WEIGHT:
				if (path->pset[path->num].eval != P_DEL)
				    ((WORD *)path->pset[path->num].b_ptr)->weight
					= atof((char *)p1+1);
				break;
                              case PA_HYP_SPKR:
                                break;
			      case PA_HYP_ISSPKRSUB:
                                break;
			      default:
			       fprintf(stderr,"Warning: Unknown auxillary fiel"
				       "d type\n");
			    }
			    *p2 = save;
			    p1 = p2;
			}
			aux_val ++;
		    }
		}
		path->num++;
		if (dbg) printf("\n");
		token = TEXT_strqtok((TEXT *)0, (TEXT *)":\n");
	    }
	} else {
 	    printf("%s: Ignored data outside of PATH tag\n%s",proc,buf);
	}
    }
    
    free_singarr(buf,TEXT);
    return 1;

  FAIL:
    fprintf(scfp,"Error: %s %s\n",proc,msg);
    return(0);
}

int SCORES_get_grp(SCORES *sc, char *grpname){
    int i;
    GRP *tgrp;

    for (i=0; i<sc->num_grp; i++)
	if (strcmp(grpname,sc->grp[i].name) == 0)
	    return(i);
    
    /* add this one to the list, allocating a new group */
    if (sc->num_grp+1 >= sc->max_grp) {
	/* allocate extra space */
	alloc_singarr(tgrp,(int)(sc->max_grp * 1.5),GRP);
	memcpy(tgrp,sc->grp,sizeof(GRP) * sc->num_grp);
	free_singarr(sc->grp,GRP);
	sc->grp = tgrp;
	sc->max_grp = (int)(sc->max_grp * 1.5);
    }
    sc->grp[sc->num_grp].name   = (char *)TEXT_strdup((TEXT *)grpname);
    sc->grp[sc->num_grp].corr   = 0;
    sc->grp[sc->num_grp].ins    = 0;
    sc->grp[sc->num_grp].del    = 0;
    sc->grp[sc->num_grp].sub    = 0;
    sc->grp[sc->num_grp].merges = 0;
    sc->grp[sc->num_grp].splits = 0;

    sc->grp[sc->num_grp].weight_ref    = 0;
    sc->grp[sc->num_grp].weight_corr   = 0;
    sc->grp[sc->num_grp].weight_ins    = 0;
    sc->grp[sc->num_grp].weight_del    = 0;
    sc->grp[sc->num_grp].weight_sub    = 0;
    sc->grp[sc->num_grp].weight_merges = 0;
    sc->grp[sc->num_grp].weight_splits = 0;

    sc->grp[sc->num_grp].nsent  = 0;
    sc->grp[sc->num_grp].serr   = 0;

    sc->grp[sc->num_grp].max_path = 20;
    sc->grp[sc->num_grp].num_path = 0;
    alloc_singarr(sc->grp[sc->num_grp].path,
		  sc->grp[sc->num_grp].max_path,PATH *);
    sc->num_grp ++;
    return(sc->num_grp -1);
}

SCORES *SCORES_init(char *name, int ngrp){
    SCORES *sc;
    if (ngrp < 20)
	ngrp = 20;

    alloc_singarr(sc,1,SCORES);
    sc->title = (char *)TEXT_strdup((TEXT *)name);
    sc->max_grp = ngrp;
    sc->num_grp = 0;

    sc->ref_fname = NULL;
    sc->hyp_fname = NULL;
    sc->frag_corr = FALSE;
    sc->opt_del = FALSE;
    sc->weight_ali = FALSE;
    sc->weight_file = (char *)0;
    sc->creation_date = NULL;

    alloc_singarr(sc->grp,ngrp,GRP);

    sc->aset.max_plab = 10;
    sc->aset.num_plab = 0;
    alloc_singarr(sc->aset.plab,sc->aset.max_plab,PATHLABEL_ITEM);

    sc->aset.max_cat = 10;
    sc->aset.num_cat = 0;
    alloc_singarr(sc->aset.cat,sc->aset.max_cat,CATEGORY_ITEM);
    return(sc);
}


void SCORES_free(SCORES *scor){
    int i, p;
    GRP *gp;

    free_singarr(scor->title,char);
    if (scor->ref_fname) free_singarr(scor->ref_fname,char);
    if (scor->hyp_fname) free_singarr(scor->hyp_fname,char);
    if (scor->creation_date) free_singarr(scor->creation_date,char);

    for (i=0; i<scor->num_grp; i++){
	gp = &(scor->grp[i]);
	free_singarr(gp->name,char);
	for (p=0; p<gp->num_path; p++)
	    PATH_free(gp->path[p]);
	free_singarr(gp->path,PATH *)
    }
    free_singarr(scor->grp,GRP);

    for (i=0; i<scor->aset.num_plab; i++){
	if (scor->aset.plab[i].id != (char *)0) free_singarr(scor->aset.plab[i].id,char);
	if (scor->aset.plab[i].title != (char *)0)
	    free_singarr(scor->aset.plab[i].title,char);
	if (scor->aset.plab[i].desc != (char *)0)
	    free_singarr(scor->aset.plab[i].desc,char);
    }
    free_singarr(scor->aset.plab,PATHLABEL_ITEM);

    for (i=0; i<scor->aset.num_cat; i++){
	if (scor->aset.cat[i].id != (char *)0) free_singarr(scor->aset.cat[i].id,char);
	if (scor->aset.cat[i].title != (char *)0)
	    free_singarr(scor->aset.cat[i].title,char);
	if (scor->aset.cat[i].desc != (char *)0)
	    free_singarr(scor->aset.cat[i].desc,char);
    }
    free_singarr(scor->aset.cat,CATEGORY_ITEM);
    if (scor->weight_file != (char *)0) free_singarr(scor->weight_file,char);
    
    free_singarr(scor,SCORES);
}

int find_PATHLABEL_id(SCORES *sc, char *id){
    int i; 
    for (i=0; i<sc->aset.num_plab; i++){
	if (TEXT_strcasecmp((TEXT*)sc->aset.plab[i].id,(TEXT*)id) == 0)
	    return i;
    }
    return -1;
}



/* returns true if buf is a comment line.  If it is, check to see if */
/* there is any information in the comment line to extract!          */
int parse_input_comment_line(SCORES *sc, TEXT *buf){
    TEXT *bq, *eq;
    int dup, n;

    if (! TEXT_is_comment(buf))
	return(0);

    if (TEXT_strCcmp(buf,(TEXT *)";; LABEL ",9) == 0) {
	/* parse the label line */
	
	if (sc->aset.num_plab == sc->aset.max_plab){
	    expand_singarr(sc->aset.plab,sc->aset.num_plab,
			   sc->aset.max_plab,2,PATHLABEL_ITEM);
	}
	
	/* locate the id token */
	if ((bq = TEXT_strchr(buf,(TEXT)'"')) == NULL) goto FAILED;
	if ((eq = TEXT_strchr(bq+1,(TEXT)'"')) == NULL) goto FAILED;
	sc->aset.plab[sc->aset.num_plab].id = 
	  (char *)TEXT_strBdup(bq+1,eq-bq-1);
	
	/* the title */
	if ((bq = TEXT_strchr(eq+1,(TEXT)'"')) == NULL) goto FAILED;
	if ((eq = TEXT_strchr(bq+1,(TEXT)'"')) == NULL) goto FAILED;
	sc->aset.plab[sc->aset.num_plab].title = 
	  (char *)TEXT_strBdup(bq+1,eq-bq-1);
	
	/* the Description  */
	if ((bq = TEXT_strchr(eq+1,(TEXT)'"')) == NULL) goto FAILED;
	if ((eq = TEXT_strchr(bq+1,(TEXT)'"')) == NULL) goto FAILED;
	sc->aset.plab[sc->aset.num_plab].desc =
	  (char *)TEXT_strBdup(bq+1,eq-bq-1);

	/* before we increment the counter, let's make sure this is not
	   a duplicate entry */
	for (n=0, dup=0; n < sc->aset.num_plab; n++){
	  if ((TEXT_strcmp((TEXT*)sc->aset.plab[sc->aset.num_plab].id,
			   (TEXT*)sc->aset.plab[n].id) == 0) &&
	      (TEXT_strcmp((TEXT*)sc->aset.plab[sc->aset.num_plab].title,
			   (TEXT*)sc->aset.plab[n].title) == 0) &&
	      (TEXT_strcmp((TEXT*)sc->aset.plab[sc->aset.num_plab].desc,
			   (TEXT*)sc->aset.plab[n].desc) == 0))
	    dup = 1;
	}
	if (dup == 0)
	  sc->aset.num_plab++;	    
	else {
	  /* free what was just created */
	  TEXT_free((TEXT*)sc->aset.plab[sc->aset.num_plab].id);
	  TEXT_free((TEXT*)sc->aset.plab[sc->aset.num_plab].title);
	  TEXT_free((TEXT*)sc->aset.plab[sc->aset.num_plab].desc);
	}
    } else if (TEXT_strCcmp(buf,(TEXT *)";; CATEGORY ",12) == 0) {

	/* parse the category line */
	if (sc->aset.num_cat == sc->aset.max_cat){
	    printf("Expanding\n");
	    expand_singarr(sc->aset.cat,sc->aset.num_cat,
			   sc->aset.max_cat,2,CATEGORY_ITEM);
	}
	
	/* locate the id token */
	if ((bq = TEXT_strchr(buf,(TEXT)'"')) == NULL) goto FAILED;
	if ((eq = TEXT_strchr(bq+1,(TEXT)'"')) == NULL) goto FAILED;
	sc->aset.cat[sc->aset.num_cat].id =
	  (char *)TEXT_strBdup(bq+1,eq-bq-1);
	
	/* the title */
	if ((bq = TEXT_strchr(eq+1,(TEXT)'"')) == NULL) goto FAILED;
	if ((eq = TEXT_strchr(bq+1,(TEXT)'"')) == NULL) goto FAILED;
	sc->aset.cat[sc->aset.num_cat].title =
	  (char *)TEXT_strBdup(bq+1,eq-bq-1);
	
	/* the Description  */
	if ((bq = TEXT_strchr(eq+1,(TEXT)'"')) == NULL) goto FAILED;
	if ((eq = TEXT_strchr(bq+1,(TEXT)'"')) == NULL) goto FAILED;
	sc->aset.cat[sc->aset.num_cat].desc = 
	  (char *)TEXT_strBdup(bq+1,eq-bq-1);
	
	/* before we increment the counter, let's make sure this is not
	   a duplicate entry */
	for (n=0, dup=0; n < sc->aset.num_cat; n++){
	  if ((TEXT_strcmp((TEXT*)sc->aset.cat[sc->aset.num_cat].id,
			   (TEXT*)sc->aset.cat[n].id) == 0) &&
	      (TEXT_strcmp((TEXT*)sc->aset.cat[sc->aset.num_cat].title,
			   (TEXT*)sc->aset.cat[n].title) == 0) &&
	      (TEXT_strcmp((TEXT*)sc->aset.cat[sc->aset.num_cat].desc,
			   (TEXT*)sc->aset.cat[n].desc) == 0))
	    dup = 1;
	}
	if (dup == 0)
	  sc->aset.num_cat++;	    
	else {
	  /* free what was just created */
	  TEXT_free((TEXT*)sc->aset.cat[sc->aset.num_cat].id);
	  TEXT_free((TEXT*)sc->aset.cat[sc->aset.num_cat].title);
	  TEXT_free((TEXT*)sc->aset.cat[sc->aset.num_cat].desc);
	}
    }

    goto FINISHED;

  FAILED:
  FINISHED:

    return(1);
}

void load_comment_labels_from_file(SCORES *scor, char *labfile){
    FILE *fp;
    int len=2000;
    TEXT *buf;

    alloc_singZ(buf,len,TEXT,(TEXT)0);

    if ((fp = fopen(labfile,"r")) == NULL)
	return;

    while (TEXT_ensure_fgets(&buf, &len, fp) != NULL) 
	parse_input_comment_line(scor, buf);

    fclose(fp);
    free_singarr(buf,TEXT);
}

void add_PATH_score(SCORES *sc, PATH *path, int g, int keep_path){
    int i, err=0;

    for (i=0; i<path->num; i++){
	if (path->pset[i].eval == P_INS){
	    sc->grp[g].ins ++;
	    sc->grp[g].weight_ins += ((WORD*)(path->pset[i].b_ptr))->weight;
	    err++;
	} else if (path->pset[i].eval == P_DEL){
	    sc->grp[g].del ++;
	    sc->grp[g].weight_del += ((WORD*)(path->pset[i].a_ptr))->weight;
	    sc->grp[g].weight_ref += ((WORD*)(path->pset[i].a_ptr))->weight;
	    err++;
	} else if (path->pset[i].eval == P_CORR){
	    sc->grp[g].corr ++;
	    sc->grp[g].weight_ref += ((WORD*)(path->pset[i].a_ptr))->weight;
	    sc->grp[g].weight_corr += ((WORD*)(path->pset[i].b_ptr))->weight;
	} else if (path->pset[i].eval == P_SUB){
	    sc->grp[g].sub ++;
	    sc->grp[g].weight_sub += ((WORD*)(path->pset[i].a_ptr))->weight + 
				      ((WORD*)(path->pset[i].b_ptr))->weight;
	    sc->grp[g].weight_ref += ((WORD*)(path->pset[i].a_ptr))->weight;
	    err++;
	} else {
	    printf("Warning: Unknown Alignment Evaluation %d\n",
		   path->pset[i].eval);
	}
    }
    sc->grp[g].nsent ++;
    if (err > 0)
	sc->grp[g].serr ++;

    if (keep_path){
	if (sc->grp[g].num_path + 1 >= sc->grp[g].max_path){
	    /* Expand the array */
	    expand_singarr(sc->grp[g].path,sc->grp[g].num_path,
			   sc->grp[g].max_path,2,PATH *)
	}
	sc->grp[g].path[sc->grp[g].num_path++] = path;
    }
}

/************************************************************************/
/*   print the report for the entire system.                            */
/************************************************************************/
void print_system_summary(SCORES *sc, char *sys_root_name, int do_sm, int do_raw, int do_weighted, int feedback)
{
    int spkr, tot_corr=0, tot_sub=0, tot_del=0;
    int total_sents=0, char_align=0, np=0;
    FILE *fp;
    char *fname;

    /* added by Brett to compute mean, variance, and standard dev */
    double *corr_arr, *sub_arr, *del_arr, *ins_arr, *err_arr, *serr_arr;
    double *spl_arr, *mrg_arr, *nce_arr;

    int tot_ins=0, tot_ref =0, tot_st=0, tot_st_er=0;
    int tot_spl=0, tot_mrg=0, tot_word=0;
    double median_corr, median_del, median_ins, median_sub, median_err;
    double median_spl, median_mrg, median_serr, median_nce;
    double mean_corr, mean_del, mean_ins, mean_sub, mean_err, mean_serr;
    double mean_spl, mean_mrg, mean_nce;
    double var_corr, var_del, var_ins, var_sub, var_err, var_serr;
    double var_spl, var_mrg, var_nce;
    double sd_corr, sd_del, sd_ins, sd_sub, sd_err, sd_serr;
    double sd_spl, sd_mrg, sd_nce;
    double Z_stat;
    int *sent_num_arr, *word_num_arr;
    double mean_sent_num, var_sent_num, sd_sent_num, median_sent_num, Z_stat_fl;
    double mean_word_num, var_word_num, sd_word_num, median_word_num;
    char *pct_fmt, *tot_pct_fmt, *spkr_fmt=" %s ", *sent_fmt="%5d", *Zpct_fmt;
    char *tot_Zpct_fmt, *tpct_fmt;
    int prec, tprec;
    int Zero_spkr = 0;
    char *Znce_fmt;
    char *nce_fmt; int nce_prec;

    double nce_system = 0.0;

    int has_hyp_conf = 0, not_has_hyp_conf = 0;   

    /* scan through the set of results, looking at the path */
    /* attributes. If any of the paths have the confidence attribute */
    /* set, set the variable 'has_hyp_conf' to 1.  This adds */ 
    /* the normalized cross-entropy statistic to the output report */
    for (spkr=0; spkr<sc->num_grp; spkr++){
        for (np=0; np<sc->grp[spkr].num_path; np++){
	    if (BF_isSET(sc->grp[spkr].path[np]->attrib,PA_HYP_CONF))
	        has_hyp_conf=1;
	    else 
	        not_has_hyp_conf = 1;
	}
    }
    /* if some segments have confidences, and some do not, ignore the */
    /* confidence scores */
    if (has_hyp_conf == 1 && not_has_hyp_conf == 1){
        has_hyp_conf = 0;
	fprintf(stderr,"Warning: Ignoring confidence scores in output\n");
    }

    /* allocate memory */
    alloc_singZ(corr_arr,sc->num_grp,double,0.0);
    alloc_singZ(sub_arr,sc->num_grp,double,0.0);
    alloc_singZ(del_arr,sc->num_grp,double,0.0);
    alloc_singZ(ins_arr,sc->num_grp,double,0.0);
    alloc_singZ(err_arr,sc->num_grp,double,0.0);
    alloc_singZ(serr_arr,sc->num_grp,double,0.0);
    alloc_singZ(spl_arr,sc->num_grp,double,0.0);
    alloc_singZ(mrg_arr,sc->num_grp,double,0.0);
    alloc_singZ(nce_arr,sc->num_grp,double,0.0);
    alloc_singZ(sent_num_arr,sc->num_grp,int,0);
    alloc_singZ(word_num_arr,sc->num_grp,int,0.0);

    for (spkr=0; spkr<sc->num_grp; spkr++){
      double Trefs;
      if (! do_weighted)
	Trefs = sc->grp[spkr].sub + sc->grp[spkr].corr + 
	  sc->grp[spkr].del + sc->grp[spkr].merges + sc->grp[spkr].splits;
      else
	Trefs = sc->grp[spkr].weight_ref;

      if (do_raw || Trefs == 0){
	corr_arr[spkr] = sc->grp[spkr].corr;
	sub_arr[spkr]  = sc->grp[spkr].sub;
	ins_arr[spkr]  = sc->grp[spkr].ins;
	del_arr[spkr]  = sc->grp[spkr].del;
	mrg_arr[spkr]  = sc->grp[spkr].merges;
	spl_arr[spkr]  = sc->grp[spkr].splits;
	err_arr[spkr]  = sc->grp[spkr].sub+sc->grp[spkr].ins+
	  sc->grp[spkr].del+sc->grp[spkr].merges+sc->grp[spkr].splits;
      } else {
	if (! do_weighted){
	  corr_arr[spkr] = pct(sc->grp[spkr].corr,Trefs);
	  sub_arr[spkr]  = pct(sc->grp[spkr].sub,Trefs);
	  ins_arr[spkr]  = pct(sc->grp[spkr].ins,Trefs);
	  del_arr[spkr]  = pct(sc->grp[spkr].del,Trefs);
	  mrg_arr[spkr]  = pct(sc->grp[spkr].merges,Trefs);
	  spl_arr[spkr]  = pct(sc->grp[spkr].splits,Trefs);
	  err_arr[spkr]  = pct(sc->grp[spkr].sub+sc->grp[spkr].ins+
			       sc->grp[spkr].del+sc->grp[spkr].merges+
			       sc->grp[spkr].splits,Trefs);
	} else {
	  corr_arr[spkr] = pct(sc->grp[spkr].weight_corr,Trefs);
	  sub_arr[spkr]  = pct(sc->grp[spkr].weight_sub,Trefs);
	  ins_arr[spkr]  = pct(sc->grp[spkr].weight_ins,Trefs);
	  del_arr[spkr]  = pct(sc->grp[spkr].weight_del,Trefs);
	  mrg_arr[spkr]  = pct(sc->grp[spkr].weight_merges,Trefs);
	  spl_arr[spkr]  = pct(sc->grp[spkr].weight_splits,Trefs);
	  err_arr[spkr]  = pct(sc->grp[spkr].weight_sub+sc->grp[spkr].weight_ins+
			       sc->grp[spkr].weight_del+sc->grp[spkr].weight_merges+
			       sc->grp[spkr].weight_splits,Trefs);
	}
      }
      if (do_raw)
	serr_arr[spkr] = sc->grp[spkr].serr;
      else
	serr_arr[spkr] = pct(sc->grp[spkr].serr,sc->grp[spkr].nsent);

      sent_num_arr[spkr]=sc->grp[spkr].nsent;
      word_num_arr[spkr] =sc->grp[spkr].corr+ sc->grp[spkr].sub + 
	sc->grp[spkr].del;

      tot_corr+=  (! do_weighted) ? sc->grp[spkr].corr : sc->grp[spkr].weight_corr;
      tot_sub+=   (! do_weighted) ? sc->grp[spkr].sub : sc->grp[spkr].weight_sub;
      tot_del+=   (! do_weighted) ? sc->grp[spkr].del : sc->grp[spkr].weight_del;
      tot_ins+=   (! do_weighted) ? sc->grp[spkr].ins : sc->grp[spkr].weight_ins;
      tot_ref+=   Trefs; 
      tot_spl+=   (! do_weighted) ? sc->grp[spkr].splits : sc->grp[spkr].weight_splits;
      tot_mrg+=   (! do_weighted) ? sc->grp[spkr].merges : sc->grp[spkr].weight_merges;
      tot_st_er+= sc->grp[spkr].serr;
      tot_st+=    sc->grp[spkr].nsent;
      tot_word+=  word_num_arr[spkr];
      total_sents+=sc->grp[spkr].nsent;

	/* test to see if any if the paths are character alignments */
      for (np=0; np<sc->grp[spkr].num_path && char_align==0; np++)
	if (BF_isSET(sc->grp[spkr].path[np]->attrib,PA_CHAR_ALIGN))
	  char_align=1;
    }

    if (has_hyp_conf == 1)
        compute_SCORE_nce(sc, &nce_system, nce_arr);

    nce_fmt = "%7.3f ";  nce_prec = 3;
    Znce_fmt = "%7.3f#";
    Zpct_fmt = "%5.0f*";
    tot_Zpct_fmt = "%5.1f+";
    if (!do_raw){
	pct_fmt = tot_pct_fmt = "%5.1f ";
	prec = tprec = 1;
    } else {
	pct_fmt="%5.0f ";    	prec = 0;
	tot_pct_fmt="%5.1f ";  tprec = 1;
    }

    /* open an output file, if it fails, write to stdout */
    if (strcmp(sys_root_name,"-") == 0 ||
	(fp = fopen(fname=rsprintf("%s.%s",sys_root_name,
				   (do_weighted) ? "wws" : 
				   ((do_raw) ? "raw" : "sys")),"w")) == NULL)
	fp = stdout;
    else
	if (feedback >= 1)
	    printf("    Writing %sscoring report to '%s'\n",
		   do_raw ? "raw " : "", fname);
    
    fprintf(fp,"\n\n\n%s\n\n",
	    center("SYSTEM SUMMARY PERCENTAGES by SPEAKER",SCREEN_WIDTH));
    if (do_weighted){
      fprintf(fp,"\n\n%s\n",
	      center("**********************************************************************",
		     SCREEN_WIDTH));
      fprintf(fp,"%s\n",
	      center("*****     Word Percentages Computed using Weighted Word Scoring  *****",
		     SCREEN_WIDTH));
      fprintf(fp,"%s\n\n",
	      center("**********************************************************************",
		     SCREEN_WIDTH));
      fprintf(fp,"%s\n\n",
	      center(rsprintf("** Weights defined by file: '%s'",
			      sc->weight_file),SCREEN_WIDTH));

    }
    if (!do_sm && (tot_spl + tot_mrg) > 0)
	fprintf(fp,"\nWarning: Split and/or Merges found, but not reported\n");

    Desc_erase();
    Desc_set_page_center(SCREEN_WIDTH);
    Desc_add_row_values("c",sc->title);
    Desc_add_row_separation('-',BEFORE_ROW);
    if (has_hyp_conf == 0){
      if (do_sm)
	Desc_add_row_values("l|cc|cccccccc"," SPKR"," # Snt",
			    (char_align) ? "# Chr" : "# Wrd","Corr",
			    " Sub"," Del"," Ins"," Mrg"," Spl"," Err","S.Err");
      else
	Desc_add_row_values("l|cc|cccccc"," SPKR"," # Snt",
			    (char_align) ? "# Chr" : "# Wrd","Corr",
			    " Sub"," Del"," Ins"," Err","S.Err");
    } else {
      if (do_sm)
	Desc_add_row_values("l|cc|cccccccc|c"," SPKR"," # Snt",
			    (char_align) ? "# Chr" : "# Wrd","Corr",
			    " Sub"," Del"," Ins"," Mrg"," Spl"," Err","S.Err",
			    "NCE");
      else
	Desc_add_row_values("l|cc|cccccc|c"," SPKR"," # Snt",
			    (char_align) ? "# Chr" : "# Wrd","Corr",
			    " Sub"," Del"," Ins"," Err","S.Err","NCE");
    }
    for (spkr=0; spkr<sc->num_grp; spkr++){
	Desc_add_row_separation('-',BEFORE_ROW);

	if (! has_hyp_conf){
	    if (do_sm)
	        Desc_set_iterated_format("l|cc|cccccccc");
	    else
	        Desc_set_iterated_format("l|cc|cccccc");
	} else {
	    if (do_sm)
	        Desc_set_iterated_format("l|cc|cccccccc|c");
	    else
	        Desc_set_iterated_format("l|cc|cccccc|c");
	}
	Desc_set_iterated_value(rsprintf(spkr_fmt,sc->grp[spkr].name));
	Desc_set_iterated_value(rsprintf(sent_fmt,sent_num_arr[spkr]));
	Desc_set_iterated_value(rsprintf(sent_fmt,word_num_arr[spkr]));
	if ((word_num_arr[spkr] > 0 && !do_raw) || do_raw){
	  Desc_set_iterated_value(rsprintf(pct_fmt,
					   F_ROUND(corr_arr[spkr],prec)));
	  Desc_set_iterated_value(rsprintf(pct_fmt,
					   F_ROUND(sub_arr[spkr],prec)));
	  Desc_set_iterated_value(rsprintf(pct_fmt,
					   F_ROUND(del_arr[spkr],prec)));
	  Desc_set_iterated_value(rsprintf(pct_fmt,
					   F_ROUND(ins_arr[spkr],prec)));
	  if (do_sm){
	    Desc_set_iterated_value(rsprintf(pct_fmt,
					     F_ROUND(mrg_arr[spkr],prec)));
	    Desc_set_iterated_value(rsprintf(pct_fmt,
					     F_ROUND(spl_arr[spkr],prec)));
	  }
	  Desc_set_iterated_value(rsprintf(pct_fmt,
					   F_ROUND(err_arr[spkr],prec)));
	} else {
	  Desc_set_iterated_value(rsprintf(Zpct_fmt,
					   F_ROUND(corr_arr[spkr],0)));
	  Desc_set_iterated_value(rsprintf(Zpct_fmt,
					   F_ROUND(sub_arr[spkr],0)));
	  Desc_set_iterated_value(rsprintf(Zpct_fmt,
					   F_ROUND(del_arr[spkr],0)));
	  Desc_set_iterated_value(rsprintf(Zpct_fmt,
					   F_ROUND(ins_arr[spkr],0)));
	  if (do_sm){
	    Desc_set_iterated_value(rsprintf(Zpct_fmt,
					     F_ROUND(mrg_arr[spkr],0)));
	    Desc_set_iterated_value(rsprintf(Zpct_fmt,
					     F_ROUND(spl_arr[spkr],0)));
	  }
	  Desc_set_iterated_value(rsprintf(Zpct_fmt,
					   F_ROUND(err_arr[spkr],0)));
	  Zero_spkr++;
	}
	Desc_set_iterated_value(rsprintf(pct_fmt,
					 F_ROUND(serr_arr[spkr],prec)));
	if (has_hyp_conf)
	  if (word_num_arr[spkr] > 0){
 	    Desc_set_iterated_value(rsprintf(nce_fmt,
					     F_ROUND(nce_arr[spkr],nce_prec)));
	  } else {
 	    Desc_set_iterated_value(rsprintf("# "));
	  }
	Desc_flush_iterated_row();
    }
    Desc_add_row_separation('=',BEFORE_ROW); 

    if (! has_hyp_conf){
        if (do_sm)
	    Desc_set_iterated_format("l|cc|cccccccc");
	else
	    Desc_set_iterated_format("l|cc|cccccc");
    } else {
        if (do_sm)
	    Desc_set_iterated_format("l|cc|cccccccc|c");
	else
	    Desc_set_iterated_format("l|cc|cccccc|c");
    }
    if (!do_raw){
	Desc_set_iterated_value(" Sum/Avg");
	Desc_set_iterated_value(rsprintf(sent_fmt,tot_st));
	Desc_set_iterated_value(rsprintf(sent_fmt,tot_word));
	Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(pct(tot_corr,tot_ref),tprec)));
	Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(pct(tot_sub,tot_ref),tprec)));
	Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(pct(tot_del,tot_ref),tprec)));
	Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(pct(tot_ins,tot_ref),tprec)));
	if (do_sm){
	    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(
					     pct(tot_mrg,tot_ref),tprec)));
	    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(
					     pct(tot_spl,tot_ref),tprec)));
	}
	Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(
					 pct(tot_sub + tot_ins + tot_del +
					     tot_spl + tot_mrg,tot_ref),tprec)));
	Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(pct(tot_st_er,tot_st),tprec)));
	if (has_hyp_conf) 
  	    Desc_set_iterated_value(rsprintf(nce_fmt,F_ROUND(nce_system,nce_prec)));
	Desc_flush_iterated_row();
    } else {
	Desc_set_iterated_value(" Sum");
	Desc_set_iterated_value(rsprintf(sent_fmt,tot_st));
	Desc_set_iterated_value(rsprintf(sent_fmt,tot_word));
	Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)tot_corr,prec)));
	Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)tot_sub,prec)));
	Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)tot_del,prec)));
	Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)tot_ins,prec)));
	if (do_sm){
	    Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)tot_mrg,prec)));
	    Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)tot_spl,prec)));
	}
	Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)(tot_sub + tot_ins +
								 tot_del + tot_spl +
								 tot_mrg),prec)));
	Desc_set_iterated_value(rsprintf(pct_fmt,F_ROUND((double)tot_st_er,prec)));
	if (has_hyp_conf)
  	    Desc_set_iterated_value(rsprintf(nce_fmt,F_ROUND(nce_system,nce_prec)));
	Desc_flush_iterated_row();
    }

    /* added by Brett to compute mean, variance, and standard dev */
    if (Zero_spkr > 0) {
      /* remove any speakers data if word_num_arr[spkr] is 0 */
      for (spkr=0; spkr<sc->num_grp; spkr++){
	if (word_num_arr[spkr] == 0) {
	  int s2;
	  /* shift up the speakers data to fill this one */
	  for (s2=spkr; s2 < sc->num_grp-1; s2++){
	    corr_arr[s2] = corr_arr[s2+1];
	    sub_arr[s2]  = sub_arr[s2+1];
	    ins_arr[s2]  = ins_arr[s2+1];
	    del_arr[s2]  = del_arr[s2+1];
	    err_arr[s2]  = err_arr[s2+1];
	  }
	  corr_arr[sc->num_grp-1] =  -1000.0;
	  sub_arr[sc->num_grp-1]  =  -1000.0;
	  ins_arr[sc->num_grp-1]  =  -1000.0;
	  del_arr[sc->num_grp-1]  =  -1000.0;
	  err_arr[sc->num_grp-1]  =  -1000.0;
	}
      }
    }
    calc_mean_var_std_dev_Zstat(sent_num_arr,sc->num_grp,
				&mean_sent_num,&var_sent_num,&sd_sent_num,
				&median_sent_num, &Z_stat_fl);
    calc_mean_var_std_dev_Zstat(word_num_arr,sc->num_grp,
				&mean_word_num,&var_word_num,&sd_word_num,
				&median_word_num, &Z_stat_fl);
    calc_mean_var_std_dev_Zstat_double(corr_arr,sc->num_grp - Zero_spkr,
                                &mean_corr,&var_corr,&sd_corr,&median_corr,
				&Z_stat);
    calc_mean_var_std_dev_Zstat_double(sub_arr,sc->num_grp - Zero_spkr,
                                &mean_sub,&var_sub,&sd_sub,&median_sub,
                                &Z_stat);
    calc_mean_var_std_dev_Zstat_double(ins_arr,sc->num_grp - Zero_spkr,
		        &mean_ins,&var_ins,&sd_ins,&median_ins,&Z_stat);
    calc_mean_var_std_dev_Zstat_double(del_arr,sc->num_grp - Zero_spkr,
			&mean_del,&var_del,&sd_del,&median_del,&Z_stat);
    calc_mean_var_std_dev_Zstat_double(spl_arr,sc->num_grp - Zero_spkr,
			&mean_spl,&var_spl,&sd_spl,&median_spl,&Z_stat);
    calc_mean_var_std_dev_Zstat_double(mrg_arr,sc->num_grp - Zero_spkr,
			&mean_mrg,&var_mrg,&sd_mrg,&median_mrg,&Z_stat);
    calc_mean_var_std_dev_Zstat_double(err_arr,sc->num_grp - Zero_spkr,
			&mean_err,&var_err,&sd_err,&median_err,&Z_stat);
    calc_mean_var_std_dev_Zstat_double(serr_arr,sc->num_grp,
		        &mean_serr,&var_serr,&sd_serr,&median_serr,&Z_stat);
    calc_mean_var_std_dev_Zstat_double(nce_arr,sc->num_grp,
		        &mean_nce,&var_nce,&sd_nce,&median_nce,&Z_stat);

    Desc_add_row_separation('=',BEFORE_ROW);
    if (has_hyp_conf == 0){
        if (do_sm)
	    Desc_set_iterated_format("c|cc|cccccccc"); 
	else
	    Desc_set_iterated_format("c|cc|cccccc");
    } else {
        if (do_sm)
	    Desc_set_iterated_format("c|cc|cccccccc|c"); 
	else
	    Desc_set_iterated_format("c|cc|cccccc|c");
    }
    Desc_set_iterated_value(" Mean ");
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(mean_sent_num,tprec)));
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(mean_word_num,tprec)));
    tpct_fmt = tot_pct_fmt;
    if (Zero_spkr > 0) tpct_fmt = tot_Zpct_fmt;
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(mean_corr,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(mean_sub,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(mean_del,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(mean_ins,tprec)));
    if (do_sm){
      Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(mean_spl,tprec)));
      Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(mean_mrg,tprec)));
    }
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(mean_err,tprec)));

    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(mean_serr,tprec)));
    if (has_hyp_conf) 
        Desc_set_iterated_value(rsprintf((Zero_spkr == 0) ? nce_fmt : Znce_fmt,F_ROUND(mean_nce,nce_prec)));
    Desc_flush_iterated_row();

    if (has_hyp_conf == 0){
        if (do_sm)
	    Desc_set_iterated_format("c|cc|cccccccc"); 
	else
	    Desc_set_iterated_format("c|cc|cccccc");
    } else {
        if (do_sm)
	    Desc_set_iterated_format("c|cc|cccccccc|c"); 
	else
	    Desc_set_iterated_format("c|cc|cccccc|c");
    }
    Desc_set_iterated_value(" S.D. ");
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(sd_sent_num,tprec)));
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(sd_word_num,tprec)));
    tpct_fmt = tot_pct_fmt;
    if (Zero_spkr > 0) tpct_fmt = tot_Zpct_fmt;
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(sd_corr,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(sd_sub,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(sd_del,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(sd_ins,tprec)));
    if (do_sm){
	Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(sd_spl,tprec)));
	Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(sd_mrg,tprec)));
    }
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(sd_err,tprec)));
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(sd_serr,tprec)));
    if (has_hyp_conf) 
        Desc_set_iterated_value(rsprintf((Zero_spkr == 0) ? nce_fmt : Znce_fmt,F_ROUND(sd_nce,nce_prec)));
    Desc_flush_iterated_row();

    if (has_hyp_conf == 0){
        if (do_sm)
	    Desc_set_iterated_format("c|cc|cccccccc"); 
	else
	    Desc_set_iterated_format("c|cc|cccccc");
    } else {
        if (do_sm)
	    Desc_set_iterated_format("c|cc|cccccccc|c"); 
	else
	    Desc_set_iterated_format("c|cc|cccccc|c");
    }
    Desc_set_iterated_value("Median");
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(median_sent_num,tprec)));
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(median_word_num,tprec)));
    tpct_fmt = tot_pct_fmt;
    if (Zero_spkr > 0) tpct_fmt = tot_Zpct_fmt;
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(median_corr,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(median_sub,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(median_del,tprec)));
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(median_ins,tprec)));
    if (do_sm){
	Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(median_spl,tprec)));
	Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(median_mrg,tprec)));
    }
    Desc_set_iterated_value(rsprintf(tpct_fmt,F_ROUND(median_err,tprec)));
    Desc_set_iterated_value(rsprintf(tot_pct_fmt,F_ROUND(median_serr,tprec)));
    if (has_hyp_conf) 
      Desc_set_iterated_value(rsprintf((Zero_spkr == 0) ? nce_fmt : Znce_fmt,F_ROUND(median_nce,nce_prec)));

    Desc_flush_iterated_row();

    Desc_dump_report(0,fp);

    if (Zero_spkr && !do_raw)
      fprintf(fp,"\n"
      "* No Reference words for this/these speaker(s).  Word counts supplied\n"
      "  rather than percents.\n"
      "# No Reference words for this/these speaker(s).  NCE not computable.\n"
      "+ Speaker(s) with no reference data is ignored\n");


    if (fp != stdout)
       fclose(fp);

    free_singarr(corr_arr,double);
    free_singarr(sub_arr,double);
    free_singarr(del_arr,double);
    free_singarr(ins_arr,double);
    free_singarr(err_arr,double);
    free_singarr(serr_arr,double);
    free_singarr(spl_arr,double);
    free_singarr(mrg_arr,double);
    free_singarr(nce_arr,double);
    free_singarr(sent_num_arr,int);
    free_singarr(word_num_arr,int);
}

 

/************************************************************************/
/*   print the report for the entire system.                            */
/************************************************************************/
void print_N_system_summary(SCORES *sc[], int nsc, char *out_root_name, char *test_name, int do_raw, int feedback)
{
    int s, g;
    FILE *fp;
    char *fname, lfmt[50], dfmt[50], *tot_pct_fmt, *pct_fmt;
    int prec, tprec, l;
    int **wcount;
    double **errs;
    
    if (!do_raw){
	pct_fmt=tot_pct_fmt="%6.1f ";
	prec = tprec = 1;
    } else {
	pct_fmt="%5.0f ";    	prec = 0;
	tot_pct_fmt="%5.1f ";  tprec = 1;
    }

    /* open an output file, if it fails, write to stdout */
    if (strcmp(out_root_name,"-") == 0 ||
	(fp = fopen(fname=rsprintf("%s.%s",out_root_name,
				   (do_raw) ? "raw" : "sys"),"w")) == NULL)
	fp = stdout;
    else
	if (feedback >= 1)
	    printf("    Writing %sscoring report to '%s'\n",
		   do_raw ? "raw " : "", fname);

    /* allocate memory, make enough room to contain the mean, variance, stddev,
       median and Z_stat */
    alloc_2dimZ(errs,nsc,sc[0]->num_grp + 5,double,0.0);
    alloc_2dimZ(wcount,nsc,2,int,0);

    /* calculate the percentages */
    for (s=0; s<nsc; s++)
	for (g=0; g<sc[s]->num_grp; g++){
	    GRP *gr = &(sc[s]->grp[g]);
	    if (!do_raw)
		errs[s][g] = pct((gr->ins + gr->sub + gr->del + 
				  gr->splits + gr->merges),
				 (gr->corr + gr->sub + gr->del +
				  gr->splits + gr->merges));
	    else
		errs[s][g] = (gr->ins + gr->sub + gr->del + 
			      gr->splits + gr->merges);
	    wcount[s][0] += (gr->corr + gr->sub + gr->del + 
			     gr->splits + gr->merges);
	    wcount[s][1] += (gr->ins + gr->sub + gr->del + 
			     gr->splits + gr->merges);
	}
    
    /* calculate the statistics.  The errs array has enough room in it */
    /* to containg the mean, variancs, std_dev, median, std_dev and Z_STAT*/
    for (s=0; s<nsc; s++)
	calc_mean_var_std_dev_Zstat_double(errs[s],sc[0]->num_grp,
					   &(errs[s][sc[0]->num_grp]),
					   &(errs[s][sc[0]->num_grp+1]),
					   &(errs[s][sc[0]->num_grp+2]),
					   &(errs[s][sc[0]->num_grp+3]),
					   &(errs[s][sc[0]->num_grp+4]));

    /* build the format statements */
    sprintf(lfmt,"c");    sprintf(dfmt,"l");
    for (s=0; s<nsc; s++){
	strcat(lfmt,"|c");    strcat(dfmt,"|c");
    }
	    
    /* Write the report */
    Desc_erase();
    Desc_set_page_center(SCREEN_WIDTH);
    Desc_add_row_separation(' ',BEFORE_ROW);
    if (*test_name == (char)'\0')
      Desc_add_row_separation(' ',AFTER_ROW);
    if (do_raw)
      Desc_add_row_values("c","Scoring Summmary of Word Errors by Speaker");
    else
      Desc_add_row_values("c","Scoring Summmary of Percent Word Error by Speaker");

    

    if (*test_name != (char)'\0'){
      Desc_add_row_separation(' ',AFTER_ROW);
      Desc_add_row_values("c",rsprintf("%s Test",test_name));
    }

    /* set the titles */
    Desc_add_row_separation('-',BEFORE_ROW);
    Desc_set_iterated_format(lfmt); 
    Desc_set_iterated_value("Spkr");
    for (s=0; s<nsc; s++)
	Desc_set_iterated_value(sc[s]->title);
    Desc_flush_iterated_row();

    /* Set the data rows */	
    for (g=0; g<sc[0]->num_grp; g++){
	Desc_add_row_separation('-',BEFORE_ROW);
	Desc_set_iterated_format(dfmt); 
	Desc_set_iterated_value(rsprintf(" %s ",sc[0]->grp[g].name));
	for (s=0; s<nsc; s++)
	    Desc_set_iterated_value(rsprintf(pct_fmt,
					     F_ROUND(errs[s][g],prec)));
	Desc_flush_iterated_row();
    }

    /* set the overall summation rows */
    Desc_add_row_separation('=',BEFORE_ROW); 
    Desc_set_iterated_format(lfmt);
    if (!do_raw){
	Desc_set_iterated_value(" Average ");
	for (s=0; s<nsc; s++)
	    Desc_set_iterated_value(rsprintf(pct_fmt,
					     F_ROUND(pct(wcount[s][1],wcount[s][0]),
						     tprec)));
    } else {
	Desc_set_iterated_value(" Sum ");
	for (s=0; s<nsc; s++)
	    Desc_set_iterated_value(rsprintf(pct_fmt,
					     F_ROUND(wcount[s][1],tprec)));
    }	
    Desc_flush_iterated_row();
    Desc_add_row_separation('-',BEFORE_ROW);

    for (l=0, g=sc[0]->num_grp; g<sc[0]->num_grp+5; l++, g++){
	if (l == 1 || l == 4)
	    continue;
	Desc_set_iterated_format(lfmt); 
	if (l == 0) Desc_set_iterated_value(" Mean ");
	else if (l == 2) Desc_set_iterated_value(" StdDev ");
	else if (l == 3) {
	    Desc_add_row_separation(' ',BEFORE_ROW);
	    Desc_set_iterated_value(" Median ");
	}
	for (s=0; s<nsc; s++)
	    Desc_set_iterated_value(rsprintf(tot_pct_fmt,
					     F_ROUND(errs[s][g],tprec)));
	Desc_flush_iterated_row();
    }

    Desc_dump_report(0,fp);

    if (fp != stdout)
       fclose(fp);

    free_2dimarr(errs,nsc,double);
    free_2dimarr(wcount,nsc,int);
}


/************************************************************************/
/*   print the report for the entire system.                            */
/************************************************************************/
void print_N_system_executive_summary(SCORES *sc[], int nsc, char *out_root_name, char *test_name, int do_raw, int feedback)
{
    int s, g, one_with_conf = 0;
    FILE *fp;
    char *fname, lfmt[50], dfmt[50], pfmt[50];

    /* open an output file, if it fails, write to stdout */
    if (strcmp(out_root_name,"-") == 0 ||
	(fp = fopen(fname=rsprintf("%s.%s",out_root_name,
				   (do_raw) ? "res" : "es"),"w")) == NULL)
	fp = stdout;
    else
	if (feedback >= 1)
	    printf("    Writing %sexecutive scoring report to '%s'\n",
		   do_raw ? "raw " : "", fname);

    /* | sys | nUTT nRef | Corr    Sub    Del    Ins    Err  S.Err | NCE */

    /* see if any of the systems has confidence scores */
    for (s=0; s<nsc; s++)
      if (hyp_confidences_available(sc[s]))
	one_with_conf = 1;

    /* build the format statements */
    sprintf(pfmt,"c|cc|caaaaa");
    sprintf(lfmt,"c|cc|cccccc");
    sprintf(dfmt,"l|cc|llllll");
    if (one_with_conf){
      strcat(pfmt,"|c");
      strcat(lfmt,"|c");
      strcat(dfmt,"|c");
    }
	    
    /* Write the report */
    Desc_erase();
    Desc_set_page_center(SCREEN_WIDTH);
    Desc_add_row_separation(' ',BEFORE_ROW);
    if (*test_name == (char)'\0')
      Desc_add_row_separation(' ',AFTER_ROW);
    if (do_raw)
      Desc_add_row_values("c","Executive Scoring Summary by Word Tokens");
    else
      Desc_add_row_values("c","Executive Scoring Summary by Percentages");

    if (*test_name != (char)'\0'){
      Desc_add_row_separation(' ',AFTER_ROW);
      Desc_add_row_values("c",rsprintf("%s Test",test_name));
    }
    
    Desc_set_iterated_format(lfmt); 
      Desc_set_iterated_value("System"); 
      Desc_set_iterated_value("# Snt"); 
      Desc_set_iterated_value("# Ref"); 
      Desc_set_iterated_value("Corr"); 
      Desc_set_iterated_value("Sub"); 
      Desc_set_iterated_value("Del"); 
      Desc_set_iterated_value("Ins"); 
      Desc_set_iterated_value("Err"); 
      Desc_set_iterated_value("S.Err"); 
      if (one_with_conf)    Desc_set_iterated_value("NCE");
    Desc_flush_iterated_row();

    Desc_add_row_separation('-',BEFORE_ROW);

    for (s=0; s<nsc; s++){
      int tcor, tsub, tins, tdel, tsnt, tserr, tref, terr;

      tcor = tsub = tins = tdel = tsnt = tserr = 0;
      
      Desc_set_iterated_format(lfmt);
      Desc_set_iterated_value(sc[s]->title);

      /* calc the scores */
      for (g=0; g<sc[s]->num_grp; g++){
	GRP *gr = &(sc[s]->grp[g]);
	
	tcor += gr->corr;
	tsub += gr->sub;
	tins += gr->ins;
	tdel += gr->del;
	tsnt += gr->nsent;
	tserr += gr->serr;
      }
      tref = tcor + tsub + tdel;
      terr = tsub + tdel + tins;

      Desc_set_iterated_value(rsprintf("%d",tsnt));
      Desc_set_iterated_value(rsprintf("%d",tref));
      if (!do_raw){
	Desc_set_iterated_value(rsprintf("%.1f",
					 F_ROUND(pct(tcor,tref),1)));
	Desc_set_iterated_value(rsprintf("%.1f",
					 F_ROUND(pct(tsub,tref),1)));
	Desc_set_iterated_value(rsprintf("%.1f",
					 F_ROUND(pct(tdel,tref),1)));
	Desc_set_iterated_value(rsprintf("%.1f",
					 F_ROUND(pct(tins,tref),1)));
	Desc_set_iterated_value(rsprintf("%.1f",
					 F_ROUND(pct(terr,tref),1)));
	Desc_set_iterated_value(rsprintf("%.1f",
					 F_ROUND(pct(tserr,tsnt),1)));
      } else {
	Desc_set_iterated_value(rsprintf("%d",tcor));
	Desc_set_iterated_value(rsprintf("%d",tsub));
	Desc_set_iterated_value(rsprintf("%d",tdel));
	Desc_set_iterated_value(rsprintf("%d",tins));
	Desc_set_iterated_value(rsprintf("%d",terr));
	Desc_set_iterated_value(rsprintf("%d",tserr));
      }
      if (one_with_conf){
	if (hyp_confidences_available(sc[s])){
	  double nce;
	  compute_SCORE_nce(sc[s], &nce, (double *)0);
	  Desc_set_iterated_value(rsprintf("%.3f",F_ROUND(nce,3)));
	} else 
	  Desc_set_iterated_value("---");
      }
      Desc_flush_iterated_row();
    }
    Desc_dump_report(1,fp);

    if (fp != stdout)
       fclose(fp);
}

char *get_date(void){
    time_t t;
    TEXT *tp;
    (void) time(&t);
    tp = TEXT_strdup((TEXT *)ctime(&t));
    TEXT_xnewline(tp)
    return((char *)tp);
}

void compute_SCORE_nce(SCORES *sc, double *nce_system, double *nce_arr){

    /* The single metric for confidence scores is a "normalized cross-entropy":

                        ^                ^
       [H(C) + SUM(log Pc) + SUM (log(1-Pc)) ] / H(c)
                   correct       incorrect

		   where:

       H(C) = -[n log p + (N-n)log(1-p)]
       n = number of correct HYP words
       N = total number of HYP words
       p = probability that a given HYP word is correct = n/N
       */
    double phyp_corr;
    double H_of_C;
    double sum_c_sys = 0.0, sum_i_sys = 0.0;
    double sum_c_spk, sum_i_spk;
    int conf_not_in_range = 0;
    int spkr, np, tot_corr=0, tot_sub=0, tot_ins=0;

    /* compute some word count totals */
    for (spkr=0; spkr<sc->num_grp; spkr++){
	tot_corr+=  sc->grp[spkr].corr;
	tot_sub+=   sc->grp[spkr].sub;
	tot_ins+=   sc->grp[spkr].ins;
    }

    for (spkr=0; spkr<sc->num_grp; spkr++){
	int n_words;
	n_words = 0;
	sum_c_spk = sum_i_spk = 0.0;
	for (np=0; np<sc->grp[spkr].num_path; np++){
	    PATH *path = sc->grp[spkr].path[np];
	    int wd;
	    double conf;
	    for (wd=0; wd < path->num; wd++){
	      if ((path->pset[wd].eval != P_DEL) &&
		  (*(((WORD*)path->pset[wd].b_ptr)->value) != (TEXT)'\0')){
		    conf = ((WORD *)path->pset[wd].b_ptr)->conf;
		    if (conf < 0.0 || conf > 1.0)
			conf_not_in_range ++;
		    n_words ++;
		    if (conf < 0.0000001) /* make sure there's a floor */
			conf = 0.0000001;
		    /* added by JGF, cant take the log of (1.0 - 1.0) */
		    if (conf > 0.9999999) /* make sure there's a ceiling */
			conf = 0.9999999;
		    if (path->pset[wd].eval == P_CORR)
			sum_c_spk += log(conf) * M_LOG2E;
		    else 
			sum_i_spk += log(1.0 - conf) * M_LOG2E;
		}
	    }
	}
	if (n_words > 0){
	    phyp_corr = (double)(sc->grp[spkr].corr) / 
		(double)(sc->grp[spkr].sub + sc->grp[spkr].ins + 
			 sc->grp[spkr].corr);
	    H_of_C = - ( (sc->grp[spkr].corr * log(phyp_corr) * M_LOG2E) + 
			 ( (sc->grp[spkr].sub + sc->grp[spkr].ins )
			   * log(1.0 - phyp_corr) * M_LOG2E) );
	    if (nce_arr != (double *)0)
	      nce_arr[spkr] = (H_of_C + sum_c_spk + sum_i_spk) / H_of_C;
	    sum_c_sys += sum_c_spk;
	    sum_i_sys += sum_i_spk;
	} else {
	    if (nce_arr != (double *)0)
	      nce_arr[spkr] = 0.0;
	}
    }
    phyp_corr = (double)(tot_corr) / 
	(double)(tot_sub + tot_ins + tot_corr);
    H_of_C = - ( tot_corr * log(phyp_corr) * M_LOG2E + 
		 ((tot_sub + tot_ins) * log(1.0 - phyp_corr) * M_LOG2E )); 
    *nce_system = (H_of_C + sum_c_sys + sum_i_sys) / H_of_C; 
    if (conf_not_in_range > 0)
	fprintf(stderr,"Warning: %d of %d confidence scores were"
		" not in the range (0.0,1.0)\n",conf_not_in_range,
		tot_corr+tot_sub+tot_ins);
}

void print_N_SCORE(SCORES *scor[], int nscor, char *outname, int max, int feedback, int score_diff){
  SC_CORRESPONDENCE *sc;
  char *fname;
  FILE *fp;
    
  /* open an output file, if it fails, write to stdout */
  if (strcmp(outname,"-") == 0 ||
      (fp = fopen(fname=rsprintf("%s.prn",outname),"w")) == NULL)
    fp = stdout;
  else
    if (feedback >= 1)
      printf("    Writing Multi-system alignments to '%s'\n",fname);


  sc = alloc_SC_CORRESPONDENCE(scor, nscor);
  locate_matched_data(scor, nscor, &sc);

  dump_paths_of_SC_CORRESPONDENCE(sc, max, fp, score_diff);

  if (fp != stdout && fp != (FILE *)0) fclose(fp);
}

