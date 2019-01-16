#include "sctk.h"


void remove_id(TEXT *utt, TEXT **id, int *len);
int extract_speaker(TEXT *hyp_id, TEXT *sname, enum id_types id);
void load_refs(SCORES *sc, char *hyp_file, char *ref_file, TEXT ***rset, TEXT ***refid, int *refcnt, int case_sense);
void free_refs(TEXT ***rset, TEXT ***ref_id, int refcnt);
void convert_text_to_word_list(char *file, char *words,int case_sense);
void expand_words_to_chars(ARC *arc, void *ptr);
void decode_opt_del(ARC *arc, void *ptr);
void decode_fragment(ARC *arc, void *ptr);
void lookup_word_weight(ARC *arc, void *ptr);
void set_word_duration(ARC *arc, void *ptr);
void set_word_start(ARC *arc, void *ptr);
void set_word_conf(ARC *arc, void *ptr);
void set_word_opt_del(ARC *arc, void *ptr);


#define LOAD_BUFF_LEN 1000

void remove_id(TEXT *utt, TEXT **id, int *id_len){
    TEXT *L_paren, *R_paren;
    
    R_paren = TEXT_strrchr(utt,')');
    L_paren = TEXT_strrchr(utt,'(');

    **id = NULL_TEXT;

    if (R_paren == NULL_TEXT && L_paren == NULL_TEXT)	
	return;

    if (R_paren == NULL_TEXT || L_paren == NULL_TEXT){
	fprintf(stderr,"Error: Unparsable utterance id %s\n",utt);
	exit(1);
    }

    /* make sure there is enough space */
    if (R_paren - L_paren + 1 + 1 > *id_len){
	*id_len = R_paren - L_paren + 1 + 1 + 10; 
	free_singarr(*id,TEXT);
	alloc_singZ((*id),*id_len,TEXT,'\0');
    }
    TEXT_strBcpy(*id,L_paren,R_paren - L_paren + 1);
    *(*id + (R_paren - L_paren + 1)) = NULL_TEXT;
    *L_paren = NULL_TEXT;
}

int extract_speaker(TEXT *id, TEXT *sname, enum id_types idt){
    char *proc = "extract_speaker";
    TEXT *p;
    switch (idt){
      case SP:
	TEXT_strcpy(sname, id);
	return(0);
      case ATIS:
      case WSJ:
	TEXT_strBcpy(sname, id + 1, 3);
	sname[3] = NULL_TEXT;
	return(0);
      case RM:
      case SPUID:
      case SWB:
	if (((p = TEXT_strchr(id,'-')) == NULL_TEXT) &&
	    ((p = TEXT_strchr(id,'_')) == NULL_TEXT)){
	    fprintf(stderr,"Error: %s can't locate ",proc);
	    switch (idt){
	      case RM: 	   fprintf(stderr,"%s id %s\n","RM",id); break;
	      case SPUID:  fprintf(stderr,"%s id %s\n","SPU_ID",id); break;
	      case SWB:	   fprintf(stderr,"%s id %s\n","SWB",id); break;
	      case WSJ:
	      case ATIS:
	      default:     fprintf(stderr,"*** internal error\n");
	    }
	    return(1);
        }
	TEXT_strBcpy(sname, id + 1, p - (id+1)); 
	sname[p - (id+1)] = NULL_TEXT; 
	return(0);
      default:
	fprintf(stderr,"Error: %s unknown id type\n",proc);
	return(1);
    }
}

/* free the reference transcripts */
void free_refs(TEXT ***rset, TEXT ***ref_id, int refcnt){
    free_2dimarr((*rset),refcnt,TEXT); 
    free_2dimarr((*ref_id),refcnt,TEXT);
}

void load_refs(SCORES *sc, char *hyp_file, char *ref_file, TEXT ***rset, TEXT ***refid, int *refcnt, int case_sense){
    TEXT *in_buf, *id_buf;
    int in_buf_len=LOAD_BUFF_LEN, id_buf_len=LOAD_BUFF_LEN;
    FILE *fp;
    TEXT **idset, **tidset, **refset, **refids;
    int max_ids, num_ids, i, num_ref, failure=0;

    max_ids = 3; num_ids = 0; num_ref = 0;
    alloc_singZ(idset,max_ids,TEXT *,(TEXT *)0);
    alloc_singZ(in_buf,in_buf_len,TEXT,'\0');
    alloc_singZ(id_buf,id_buf_len,TEXT,'\0');

    /* Pre-Read the entire hypothesis file */
    if ((hyp_file == NULL) || (*hyp_file == '\0') || 
        ((fp = fopen(hyp_file,"r")) == NULL)){
        fprintf(stderr,"Can't open input hypothesis file %s\n",hyp_file);
        exit(1);
    }
    while (!feof(fp)){
        *in_buf  = *(in_buf+1) = COMMENT_CHAR; 
        *(in_buf+2) = NULL_TEXT;
        while (!feof(fp) && (parse_input_comment_line(sc,in_buf) ||
			     TEXT_is_comment_info(in_buf) ||
			     TEXT_is_empty(in_buf))){
            if (TEXT_ensure_fgets(&in_buf,&in_buf_len,fp) == NULL)
               *in_buf = NULL_TEXT;
	}
	if (!feof(fp)){
	    if (!case_sense)
		TEXT_str_case_change_with_mem_expand(&in_buf, &in_buf_len, 1);
	    remove_id(in_buf,&id_buf,&id_buf_len);
	    if (num_ids + 1 >= max_ids) {
		/* expanding the array */
		alloc_singZ(tidset,max_ids*2,TEXT *, (TEXT *)0);
		memcpy(tidset,idset,sizeof(TEXT *) * num_ids);
		free_singarr(idset,TEXT *);
		idset = tidset;
		max_ids *= 2;
	    }
	    idset[num_ids++] = TEXT_strdup(id_buf);
	}
    }
    fclose(fp);

    /* sort the id's in-place */
    qsort((void *)idset,num_ids,sizeof(TEXT *),qsort_TEXT_strcmp);

    /* space for the reference transcripts allocate */
    alloc_singZ(refset,num_ids,TEXT *,(TEXT *)0);  
    
    /* Read the reference file, saving the needed texts */
    if ((ref_file == NULL) || (*ref_file == '\0') || 
        ((fp = fopen(ref_file,"r")) == NULL)){
        fprintf(stderr,"Can't open input Reference file %s\n",ref_file);
        exit(1);
    }
    while (!feof(fp)){
	TEXT **ind;
	int rind;

        *in_buf  = *(in_buf+1) = COMMENT_CHAR; 
        *(in_buf+2) = NULL_TEXT;
        while (!feof(fp) && (parse_input_comment_line(sc,in_buf) ||
			     TEXT_is_comment_info(in_buf) ||
			     TEXT_is_empty(in_buf))){
            if (TEXT_ensure_fgets(&in_buf,&in_buf_len,fp) == NULL)
               *in_buf = '\0';
	}
        if (feof(fp))
            break;

	if (!case_sense)
	    TEXT_str_case_change_with_mem_expand(&in_buf, &in_buf_len, 1);

	remove_id(in_buf,&id_buf,&id_buf_len);
	/* Search for the id in idset */
	if ((ind = (TEXT **)bsearch(id_buf,idset,num_ids,sizeof(TEXT *),
				    bsearch_TEXT_strcmp)) != NULL){
	    rind = ind - idset;
	    if (refset[rind] != (TEXT *)0){
		if (TEXT_strBcmp(in_buf,refset[rind],TEXT_strlen(in_buf))!= 0){
		    fprintf(stderr,"Error: double reference text for id '%s'\n",
			    id_buf);
		    failure=1;
		} 
	    } else {
		refset[rind] = TEXT_strdup(in_buf);
		num_ref++;
	    }
	}
    }
    fclose(fp);
    
    if (num_ref != num_ids){
	failure=1;
	fprintf(stderr,"Error: Not enough Reference files loaded\nMissing:\n");
	for (i=0; i<num_ref; i++)
	    if (refset[i] == (TEXT *)0)
	      fprintf(stderr,"    %s\n",idset[i]);
    }

    /* space for the reference id's */
    alloc_singZ(refids,num_ref,TEXT *,(TEXT *)0);  
    /* transfer the id's to the new array */
    memcpy(refids,idset,sizeof(TEXT *) * num_ref);
    /* free the idset array */
    free_singarr(idset, TEXT *)

    free_singarr(in_buf,TEXT);
    free_singarr(id_buf,TEXT);

    *rset = refset;
    *refcnt = num_ref;
    *refid = refids;

    if (failure)
	exit(1);
}

PATH *infer_word_seg_algo2(TEXT *ref, TEXT *hyp, NETWORK *hnet, int case_sense,char *id, char *lex_fname, int fcorr, int opt_del, int flags)
{
    char *proc="infer_word_seg_algo2";
    NETWORK *refnet, *hypnet;
    PATH *path;
    static TEXT_LIST *tl = (TEXT_LIST *)0;
    static int max_word_len=1;
    int char_align = (flags == INF_ASCII_TOO) ? CALI_ON : CALI_ON+CALI_NOASCII;

    if (hyp != (TEXT *)0 && hnet != (NETWORK *)0){
	fprintf(scfp,"Error: %s passed ambiguous arguments "
		"for 'hyp' and 'hnet'\n",proc);
	exit(1);
    }

    if (tl == (TEXT_LIST *)0){
	int i, s;
	if ((tl = load_TEXT_LIST(lex_fname,0)) == (TEXT_LIST *)0) {
	    fprintf(scfp,"Error: Can't load lexicon file '%s'\n",lex_fname);
	    exit(0);
	}
	/* compute the largest word, (in characters) in the TEXT_LIST */
	for (i=0; i<tl->num; i++)
	    if ((s = TEXT_chrlen(tl->elem[i])) > max_word_len)
		max_word_len = s;
    }

    {
        TEXT *refCC = (case_sense ? TEXT_strdup(TEXT_str_to_master(ref, 1)) : ref);
        if ((refnet = Network_create_from_TEXT(ref, "Reference Net",
					   print_WORD,
					   equal_WORD2,
					   release_WORD, null_alt_WORD,
					   opt_del_WORD,
					   copy_WORD, make_empty_WORD,
					   use_count_WORD))
	== NULL_NETWORK){
            fprintf(stderr,"Network_create_from_TEXT failed\n");
	    exit(1);
        }
        if (case_sense){
            free_singarr(refCC, TEXT);
        }
    }
    
    if (hyp != (TEXT *)0){
	TEXT *hypCC = (case_sense ? TEXT_strdup(TEXT_str_to_master(hyp, 1)) : hyp);
	if ((hypnet = Network_create_from_TEXT(hyp,"Hypothesis Net",
					       print_WORD,
					       equal_WORD2,
					       release_WORD, null_alt_WORD,
					       opt_del_WORD,
					       copy_WORD, make_empty_WORD,
					       use_count_WORD))
	    == NULL_NETWORK){
    fprintf(stderr,"Network_create_from_TEXT failed\n");
	    exit(1);
	}
        if (case_sense){
           free_singarr(hypCC, TEXT);
        }
    } else 
	hypnet = hnet;

    if (opt_del){
      Network_traverse(hypnet,NULL,0,decode_opt_del,0,0);
      Network_traverse(refnet,NULL,0,decode_opt_del,0,0);
    }

    if (fcorr){
      Network_traverse(hypnet,NULL,0,decode_fragment,0,0);
      Network_traverse(refnet,NULL,0,decode_fragment,0,0);
    }

    Network_traverse(refnet,NULL,0,expand_words_to_chars,&char_align,0);

    Network_fully_connect_cond(refnet, max_word_len, append_WORD_no_NULL,
			       WORD_in_TEXT_LIST, tl);

    /* printf("-----------------\n");
       Network_traverse(hypnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);
       Network_traverse(refnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);  */

    Network_dpalign(refnet,hypnet,wwd_WORD,&path,FALSE);
    PATH_add_utt_id(path,(char *)id);
    PATH_set_sequence(path);
    if (case_sense) BF_SET(path->attrib,PA_CASE_SENSE);
    
    Network_destroy(refnet);
    Network_destroy(hypnet);
    
    /* free_TEXT_LIST (&tl); */
    return(path);
}

PATH *infer_word_seg_algo1(TEXT *ref, TEXT *hyp, NETWORK *hnet, int case_sense,char *id, char *lex_fname, int fcorr, int opt_del, int flags)
{
    char *proc="infer_word_seg_algo1";
    NETWORK *refnet, *hypnet;
    PATH *path;
    static TEXT_LIST *tl = (TEXT_LIST *)0;
    static int max_word_len=1;
    int char_align = (flags == INF_ASCII_TOO) ? CALI_ON : CALI_ON+CALI_NOASCII;

    if (hyp != (TEXT *)0 && hnet != (NETWORK *)0){
	fprintf(scfp,"Error: %s passed ambiguous arguments "
		"for 'hyp' and 'hnet'\n",proc);
	exit(1);
    }

    if (tl == (TEXT_LIST *)0){
	int i, s;
	if ((tl = load_TEXT_LIST(lex_fname,0)) == (TEXT_LIST *)0) {
	    fprintf(scfp,"Error: Can't load lexicon file '%s'\n",lex_fname);
	    exit(0);
	}
	/* compute the largest word, (in characters) in the TEXT_LIST */
	for (i=0; i<tl->num; i++)
	    if ((s = TEXT_chrlen(tl->elem[i])) > max_word_len)
		max_word_len = s;
    }

    TEXT *refCC = (! case_sense ? TEXT_strdup(TEXT_str_to_master(ref, 1)) : ref);
//db=20;
    if ((refnet = Network_create_from_TEXT(refCC, "Reference Net",
					   print_WORD,
					   equal_WORD2,
					   release_WORD, null_alt_WORD,
					   opt_del_WORD,
					   copy_WORD, make_empty_WORD,
					   use_count_WORD))
	== NULL_NETWORK){
	fprintf(stderr,"Network_create_from_TEXT failed\n");
	exit(1);
    }

    if (! case_sense){
        free_singarr(refCC, TEXT);
    }    
    if (hyp != (TEXT *)0){
	TEXT *hypCC = (! case_sense ? TEXT_strdup(TEXT_str_to_master(hyp, 1)) : hyp);
	if ((hypnet = Network_create_from_TEXT(hypCC,"Hypothesis Net",
					       print_WORD,
					       equal_WORD2,
					       release_WORD, null_alt_WORD,
					       opt_del_WORD,
					       copy_WORD, make_empty_WORD,
					       use_count_WORD))
	    == NULL_NETWORK){
    fprintf(stderr,"Network_create_from_TEXT failed\n");
	    exit(1);
	}
        if (! case_sense){
            free_singarr(hypCC, TEXT);
        }
    } else 
	hypnet = hnet;

//       Network_traverse(hypnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);

    if (opt_del){
      Network_traverse(hypnet,NULL,0,decode_opt_del,0,0);
      Network_traverse(refnet,NULL,0,decode_opt_del,0,0);
    }

    if (fcorr){
      Network_traverse(hypnet,NULL,0,decode_fragment,0,0);
      Network_traverse(refnet,NULL,0,decode_fragment,0,0);
    }

//    Network_traverse(hypnet,NULL,0,0c,0,NT_CA_For+NT_Verbose);
    Network_traverse(hypnet,NULL,0,expand_words_to_chars,&char_align,0);
//    Network_traverse(hypnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);

    Network_fully_connect_cond(hypnet, max_word_len, append_WORD_no_NULL,
			       WORD_in_TEXT_LIST, tl);

//     printf("-----------------\n");
//       Network_traverse(hypnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);
//       Network_traverse(refnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);  

    Network_dpalign(refnet,hypnet,wwd_WORD,&path,FALSE);
    PATH_add_utt_id(path,(char *)id);
    PATH_set_sequence(path);
    if (case_sense) BF_SET(path->attrib,PA_CASE_SENSE);
    
    Network_destroy(refnet);
    Network_destroy(hypnet);
    
    /* free_TEXT_LIST (&tl); */
    return(path);
}

PATH *network_dp_align_texts(TEXT *ref, NETWORK *rnet, TEXT *hyp, NETWORK *hnet, int char_align, int case_sense, char *id, int fcorr, int opt_del, int time_align, WWL *wwl, char *lm_file)
{
    char *proc="network_dp_align_texts";
    NETWORK *refnet, *hypnet;
    PATH *path;

    if (hyp != (TEXT *)0 && hnet != (NETWORK *)0){
	fprintf(scfp,"Error: %s passed ambiguous arguments "
		"for 'hyp' and 'hnet'\n",proc);
	exit(1);
    }

    if (ref != (TEXT *)0 && rnet != (NETWORK *)0){
	fprintf(scfp,"Error: %s passed ambiguous arguments "
		"for 'ref' and 'rnet'\n",proc);
	exit(1);
    }
//    printf("\nREF -> %s\n",ref);
//    printf("HYP -> %s\n",hyp);
    if (ref != (TEXT *)0){
      TEXT *refCC = (! case_sense ? TEXT_strdup(TEXT_str_to_master(ref, 1)) : ref);
      
      if ((refnet = Network_create_from_TEXT(refCC, "Reference Net",print_WORD,
					     equal_WORD2,
					     release_WORD, null_alt_WORD,
					     opt_del_WORD,
					     copy_WORD, make_empty_WORD,
					     use_count_WORD))
	  == NULL_NETWORK){
	fprintf(stderr,"Network_create_from_TEXT failed\n");
	exit(1);
      }
      if (! case_sense){
        free_singarr(refCC, TEXT);
      }
    } else {
      refnet = rnet;
    }

    if (hyp != (TEXT *)0){
        TEXT *hypCC = (!case_sense ? TEXT_strdup(TEXT_str_to_master(hyp, 1)) : hyp);

	if ((hypnet = Network_create_from_TEXT(hypCC,"Hypothesis Net",
					       print_WORD,
					       equal_WORD2,
					       release_WORD, null_alt_WORD,
					       opt_del_WORD,
					       copy_WORD, make_empty_WORD,
					       use_count_WORD))
	    == NULL_NETWORK){
	    fprintf(stderr,"Network_create_from_TEXT failed\n");
	    exit(1);
	}
      if (! case_sense){
        free_singarr(hypCC, TEXT);
      }
    } else 
	hypnet = hnet;

    if (opt_del){
      Network_traverse(hypnet,NULL,0,decode_opt_del,0,0);
      Network_traverse(refnet,NULL,0,decode_opt_del,0,0);
    }

    if (fcorr){
      Network_traverse(hypnet,NULL,0,decode_fragment,0,0);
      Network_traverse(refnet,NULL,0,decode_fragment,0,0);
    }

    if (char_align){
      Network_traverse(hypnet,NULL,0,expand_words_to_chars,&char_align,0);
      Network_traverse(refnet,NULL,0,expand_words_to_chars,&char_align,0);
    }

    if (wwl != (WWL *)0){
      Network_traverse(hypnet,NULL,0,lookup_word_weight,wwl,0);
      Network_traverse(refnet,NULL,0,lookup_word_weight,wwl,0);
    }

    if (lm_file != (char *)0){
      Network_traverse(hypnet,NULL,0,lookup_lm_word_weight,lm_file,0);
      Network_traverse(refnet,NULL,0,lookup_lm_word_weight,lm_file,0);
    }

//        Network_traverse(hypnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);
//        Network_traverse(refnet,NULL,0,print_arc,0,NT_CA_For+NT_Verbose);  
    Network_dpalign(refnet,hypnet,
		    ( (time_align) ? wwd_time_WORD : 
		      ( ((wwl != (WWL *)0) || (lm_file != (char *)0)) ?
			 wwd_weight_WORD : wwd_WORD) ),
		    &path,FALSE);
    PATH_add_utt_id(path,(char *)id);
    PATH_set_sequence(path);
    if (case_sense) BF_SET(path->attrib,PA_CASE_SENSE);
    if (char_align) BF_SET(path->attrib,PA_CHAR_ALIGN);
    if ((wwl != (WWL *)0) || (lm_file != (char *)0)) {
      BF_SET(path->attrib,PA_HYP_WEIGHT);
      BF_SET(path->attrib,PA_REF_WEIGHT);
    }

    Network_destroy(refnet);
    Network_destroy(hypnet);
    
    return(path);
}

SCORES *align_trans_mode_dp(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, int char_align, enum id_types idt, int infer_word_seg, char *lexicon, int fcorr, int opt_del, int inf_no_ascii, WWL *wwl, char *lm_file){
    FILE *fp_hyp;
    TEXT *hyp_buff, *hyp_id, *spkr_id;
    int hyp_buff_len=LOAD_BUFF_LEN, hyp_id_len=LOAD_BUFF_LEN;
    int spkr_id_len=LOAD_BUFF_LEN;
    PATH *path;
    SCORES *scor;
    int spk, rind, last_spk=0;
    TEXT **reftran, **refid, **ind;
    int refcnt;

    alloc_singZ(hyp_buff,hyp_buff_len,TEXT,'\0');
    alloc_singZ(hyp_id,hyp_id_len,TEXT,'\0');
    alloc_singZ(spkr_id,spkr_id_len,TEXT,'\0');

    if ((hyp_file == NULL) || (*hyp_file == '\0') || 
        ((fp_hyp = fopen(hyp_file,"r")) == NULL)){
        fprintf(stderr,"Can't open input hypothesis file %s\n",hyp_file);
        exit(1);
    }

    scor = SCORES_init(set_title,20);
    scor->ref_fname = (char *)TEXT_strdup((TEXT *)ref_file);
    scor->hyp_fname = (char *)TEXT_strdup((TEXT *)hyp_file);
    scor->creation_date = get_date();
    scor->frag_corr = fcorr;
    scor->opt_del = opt_del;
    if (wwl != (WWL*)0) {
      scor->weight_ali = 1;
      scor->weight_file =
	(char *)TEXT_strdup((TEXT*)rsprintf("%s Column \"%s\"",wwl->filename,
					    wwl->weight_desc[0]));
    } else if (lm_file != (char *)0) {
      scor->weight_ali = 1;
      scor->weight_file = (char *)TEXT_strdup((TEXT*)lm_file);
    } else
      scor->weight_ali = 0;

    load_refs(scor, hyp_file, ref_file, &reftran, &refid, &refcnt, case_sense); 

    while (!feof(fp_hyp)){
        *hyp_buff  = *(hyp_buff+1) = COMMENT_CHAR;
        *(hyp_buff+2) = NULL_TEXT;
        while (!feof(fp_hyp) && (parse_input_comment_line(scor,hyp_buff) ||
                                 TEXT_is_comment_info(hyp_buff) ||
                                 TEXT_is_empty(hyp_buff)))
            if (TEXT_ensure_fgets(&hyp_buff,&hyp_buff_len,fp_hyp) == NULL)
               *hyp_buff = '\0';
        if (feof(fp_hyp))
            break;

	remove_id(hyp_buff,&hyp_id,&hyp_id_len);
	if (!case_sense)  TEXT_str_case_change_with_mem_expand(&hyp_id, &hyp_id_len, 1);

	extract_speaker(hyp_id,spkr_id,idt);
	spk = SCORES_get_grp(scor,(char *)spkr_id);
	
	if (feedback >= 1){
	    if (spk != last_spk) {  printf("\n"); last_spk = spk; }
	    printf("\r    Alignment# %d for speaker %s          ",
		   scor->grp[spk].nsent+1,spkr_id);
	    fflush(stdout);
	}

	if ((ind = (TEXT **)bsearch(hyp_id,refid,refcnt,sizeof(TEXT *),
				    bsearch_TEXT_strcmp)) == NULL){
	    fprintf(stderr,"Error: Unable to locate Ref transcript for '%s'\n",
		    hyp_id);
	    exit(1);
	}
	rind = ind - refid;

	if (feedback >= 100){
            printf("\n        HYP -> %s\n",hyp_buff);
            printf("        REF -> %s\n",reftran[rind]);
	    fflush(stdout);
        }
	if (infer_word_seg == 0)
	    path = network_dp_align_texts(reftran[rind], (NETWORK *)0,
					  hyp_buff, (NETWORK *)0,
					  char_align, case_sense, 
					  (char *)hyp_id, fcorr, opt_del, FALSE, wwl, lm_file);
	else if (infer_word_seg == INF_SEG_ALGO1)
	    path = infer_word_seg_algo1(reftran[rind], hyp_buff, (NETWORK *)0,
					case_sense, (char *)hyp_id, lexicon,
					fcorr, opt_del, inf_no_ascii);
	else if (infer_word_seg == INF_SEG_ALGO2)
	    path = infer_word_seg_algo2(reftran[rind], hyp_buff, (NETWORK *)0,
					case_sense, (char *)hyp_id, lexicon,
					fcorr, opt_del, inf_no_ascii);
	else{
	    fprintf(scfp,"Error: unknown alignment procedure\n");
	    exit(1);
	}

	if (feedback >= 2){
	    printf("\n"); 
	    PATH_print(path,stdout,199);
	    printf("\n");
	}

	add_PATH_score(scor,path,spk, keep_path);

	if (!keep_path)
	    PATH_free(path);
    }
    if (feedback >= 1)
	printf("\n\n");

    free_singarr(hyp_buff,TEXT);
    free_singarr(hyp_id,TEXT);
    free_singarr(spkr_id,TEXT);
    free_refs(&reftran, &refid, refcnt);
    fclose(fp_hyp);

    return(scor);
}


/*  This function, called by the network travers function, will set the */
/*  duration of the word to *(double *)ptr */
void set_word_duration(ARC *arc, void *ptr){
    WORD *tw = (WORD *)(arc->data);
    tw->T_dur = *(double *)ptr;
}

/*  This function, called by the network traverse function, will set the */
/*  words start to *(double *)ptr, then set T2 to be T1+dur, then */
/*  increment *(double *)ptr by dur. */
void set_word_start(ARC *arc, void *ptr){
    WORD *tw = (WORD *)(arc->data);
    tw->T1 = *(double *)ptr;
    tw->T2 = tw->T_dur + tw->T1;
    *(double *)ptr += tw->T_dur;
}

/*  This function, called by the network traverse function, will set the */
/*  word's confidence score to *(double *)ptr */
void set_word_conf(ARC *arc, void *ptr){
    WORD *tw = (WORD *)(arc->data);
    tw->conf = *(double *)ptr;
}

/*  This function, called by the network traverse function, will set the */
/*  word's optionally deletable flag to *(int *)ptr */
void set_word_opt_del(ARC *arc, void *ptr){
    WORD *tw = (WORD *)(arc->data);
    TEXT *ot;

    if (*(int *)ptr){
      /* re-make the optionally deletable text */
      ot = tw->value;
      tw->value = TEXT_strdup((TEXT *)rsprintf(WORD_OPT_DEL_PRE_STR "%s"
					       WORD_OPT_DEL_POST_STR,ot));
      TEXT_free(ot);
    }
}

void expand_words_to_chars(ARC *arc, void *ptr){
    char *proc = "expand_words_to_chars";
    WORD *tw = (WORD *)(arc->data);
    static TEXT *chars=(TEXT *)0, *charsEsc;
    static int chars_len=100;
    NETWORK *subnet;
    char buf[20];
    static int alt=1;

    if (chars == (TEXT *)0) alloc_singZ(chars,chars_len,TEXT,'\0');
    /* it the string is already a single letter, return */
    if (TEXT_strlen(tw->value) <= 1)
	return;

    if (db >= 5) {
	printf("%s: String length > 1\n",proc);
	arc->net->arc_func.print(arc->data);
    }
//    arc->net->arc_func.print(arc->data);
    sprintf(buf,"expand-net %d",alt);
    TEXT_separate_chars((!tw->opt_del) ? tw->value : tw->intern_value,
			&chars,&chars_len,*((int *)ptr));
			
//    printf("   from /%s/ comes /%s/\n",(!tw->opt_del) ? tw->value : tw->intern_value, chars);
    
    // Escape the semicolon
    alloc_singZ(charsEsc, chars_len*2+1, TEXT, NULL_TEXT);
    TEXT_strcpy_escaped(charsEsc, chars, ';');
//    printf("   and escaped /%s/\n", charsEsc);
    
    /* Fix, it wasn't enough check to see if the there were any spaces */
    if (TEXT_strcmp(chars,tw->value) != 0){
	subnet = Network_create_from_TEXT(charsEsc, buf, arc->net->arc_func.print,
					  arc->net->arc_func.equal,
					  arc->net->arc_func.destroy,
					  arc->net->arc_func.is_null_alt,
					  arc->net->arc_func.is_opt_del,
					  arc->net->arc_func.copy,
					  arc->net->arc_func.make_empty,
					  arc->net->arc_func.use_count);
	if (subnet == NULL_NETWORK){
	    fprintf(scfp,"proc %s: Internal error.  Unable to expand words"
		    " to characters\n",proc);
	    return;
	}
	/* traverse the network, setting the character times to equal */
	/* durations based on the original word */
	{  
	  double dur = tw->T_dur/subnet->arc_count, t1 = tw->T1, conf = tw->conf;
	  int od = tw->opt_del;
	  Network_traverse(subnet,NULL,0,set_word_duration,&dur,0);
	  Network_traverse(subnet,NULL,0,set_word_start,&t1,0);
	  Network_traverse(subnet,NULL,0,set_word_conf,&conf,0);
	  if (od){
	    Network_traverse(subnet,NULL,0,set_word_opt_del,&od,0);
	    Network_traverse(subnet,NULL,0,decode_opt_del,0,0);
	  }
	}
	Network_merge_network(arc->from_node,arc->to_node,subnet);
	Network_delete_arc(arc);
	alt++;
    }
    free_singarr(charsEsc, TEXT);
}


      
void lookup_word_weight(ARC *arc, void *ptr){
    WORD *tw = (WORD *)(arc->data);
    WWL *wwl = (WWL *)ptr;

    if (tw->value == (TEXT *)0)
      tw->weight = 0.0;
    else if (TEXT_strcmp(tw->value,(TEXT *)"@") == 0) 
      tw->weight = 0.0;
    else if (tw->opt_del) 
      tw->weight = 0.0;
    else
      tw->weight = Weight_wwl((tw->opt_del ? tw->intern_value : tw->value), wwl);

    return;
}


void decode_opt_del(ARC *arc, void *ptr){
    WORD *tw = (WORD *)(arc->data);
    int len;

    if (tw->value == (TEXT *)0) return;
    len = TEXT_strlen(tw->value);

    if ((*(tw->value + len - 1) == WORD_OPT_DEL_POST_CHAR) &&
	(*(tw->value) == WORD_OPT_DEL_PRE_CHAR)){
      tw->opt_del = TRUE;
      tw->intern_value = TEXT_strBdup(tw->value + 1,len - 2);
    }
    return;
}

void decode_fragment(ARC *arc, void *ptr){
    WORD *tw = (WORD *)(arc->data);
    int len;

    if (tw->value == (TEXT *)0) return;
    len = TEXT_strlen(tw->value);
    if (((*(tw->value + len - 1) == WORD_FRAG_CHAR) || 
	 (*(tw->value          ) == WORD_FRAG_CHAR))    ||
	(tw->opt_del && 
	 (*(tw->value + len - 2) == WORD_FRAG_CHAR)))
      tw->frag_corr = TRUE;

    return;
}


/**********************************************************************/
/************      USE GNU-DIFF to compute the alignments *************/
/**********************************************************************/

void create_word_lists(SCORES *sc, TEXT **reftran, TEXT **refid, int refcnt, char *hyp_file, char *out_ref, char *out_hyp, int *num_hyp, int case_sense);
void process_diff_line(TEXT *diff_out, PATH *path, TEXT **firsttoken);
void set_temp_files(char *refwords, char *hypwords);

void create_word_lists(SCORES *sc, TEXT **reftran, TEXT **refid, int refcnt, char *hyp_file, char *out_ref, char *out_hyp, int *num_hyp, int case_sense){
    FILE *fp_hyp, *fp_out_ref, *fp_out_hyp;
    TEXT *hyp_buff, *hyp_id, *ctext, *tbuf;
    int hyp_buff_len=LOAD_BUFF_LEN, hyp_id_len=LOAD_BUFF_LEN;
    int tbuf_len=100;
    int rind;
    TEXT **ind;

    alloc_singZ(tbuf,tbuf_len,TEXT,'\0');
    alloc_singZ(hyp_buff,hyp_buff_len,TEXT,'\0');
    alloc_singZ(hyp_id,hyp_id_len,TEXT,'\0');

    if ((fp_hyp = fopen(hyp_file,"r")) == NULL){
        fprintf(stderr,"Can't open input hypothesis file %s\n",hyp_file);
        exit(1);
    }

    if ((fp_out_hyp=fopen(out_hyp,"w")) == NULL){
	fprintf(stderr,"Error: Unable to open Hypothesis word file '%s'\n",
		out_hyp);
	exit(1);
    }
    if ((fp_out_ref=fopen(out_ref,"w")) == NULL){
	fprintf(stderr,"Error: Unable to open Reference word file '%s'\n",
		out_ref);
	exit(1);
    }

    while (!feof(fp_hyp)){
        *hyp_buff  = *(hyp_buff+1) = COMMENT_CHAR;
        *(hyp_buff+2) = NULL_TEXT;
 
        while (!feof(fp_hyp) && (parse_input_comment_line(sc,hyp_buff) ||
                                 TEXT_is_comment_info(hyp_buff) ||
                                 TEXT_is_empty(hyp_buff)))
            if (TEXT_ensure_fgets(&hyp_buff,&hyp_buff_len,fp_hyp) == NULL)
               *hyp_buff = '\0';

        if (feof(fp_hyp))
            break;

	(*num_hyp) ++;

	if (!case_sense)
	     TEXT_str_case_change_with_mem_expand(&hyp_buff, &hyp_buff_len, 1);

	remove_id(hyp_buff,&hyp_id,&hyp_id_len);
	if ((ind = (TEXT **)bsearch(hyp_id,refid,refcnt,sizeof(TEXT *),
				    bsearch_TEXT_strcmp)) == NULL){
	    fprintf(stderr,"Error: Unable to locate Ref transcript for '%s'\n",
		    hyp_id);
	    exit(1);
	}
	rind = ind - refid;

	/* Write each individual word to a file */
	if (TEXT_strlen(reftran[rind]) > tbuf_len+1){
	    free_singarr(tbuf,TEXT);
	    tbuf = TEXT_strdup(reftran[rind]);
	    tbuf_len = TEXT_strlen(tbuf);
	} else
	    TEXT_strcpy(tbuf,reftran[rind]);

	ctext = tokenize_TEXT_first_alt(tbuf,(TEXT *)" \t\n");
	while (ctext != NULL) {
	    fprintf(fp_out_ref,"%s\n",ctext);
	    ctext = tokenize_TEXT_first_alt(NULL,(TEXT *)" \t\n");
	}
	
	ctext = TEXT_strtok(hyp_buff,(TEXT *)" \t\n");
	while (ctext != NULL) {
	    fprintf(fp_out_hyp,"%s\n",ctext);
	    ctext = TEXT_strtok(NULL,(TEXT *)" \t\n");
	}
    }

    fclose(fp_out_ref);
    fclose(fp_out_hyp);
    fclose(fp_hyp);
    free_singarr(hyp_buff,TEXT);
    free_singarr(hyp_id,TEXT);
    free_singarr(tbuf,TEXT);
}

void process_diff_line(TEXT *diff_out, PATH *path, TEXT **firsttoken){
    static TEXT s1[100],s2[100],s3[100];
    int iret;
    WORD *rwd, *hwd;

    check_space_in_PATH(path);
    *s1 = *s2 = *s3 = '\0';
    iret = sscanf((char *)diff_out,
		  "%s %s %s",(char *)s1,(char *)s2,(char *)s3);
    *firsttoken = s1;

    if (iret == 2 && TEXT_strcmp(s1,s2) == 0)
	path->pset[path->num].eval = P_CORR;
    else if (iret == 2 && TEXT_strcmp(s1,(TEXT *)">") == 0)
	path->pset[path->num].eval = P_INS;
    else if (iret == 3)
	path->pset[path->num].eval = P_SUB;
    else
	path->pset[path->num].eval = P_DEL;

    path->pset[path->num].b_ptr = path->pset[path->num].a_ptr = (void *)0;

    switch (path->pset[path->num].eval){
      case P_CORR:
      case P_DEL:
      case P_SUB:
	rwd = new_WORD(s1, -1, 0.0, 0.0, 0.0, (TEXT *)0, (TEXT *)0, 0, 0, -1.0);
	path->pset[path->num].a_ptr = rwd;
      case P_INS:
	;
    }
    switch (path->pset[path->num].eval){
      case P_CORR:
      case P_INS:
      case P_SUB:
	hwd = (WORD *)0;
	switch (path->pset[path->num].eval){
	  case P_CORR:  
	  case P_INS:  
	    hwd = new_WORD(s2, -1, 0.0, 0.0, 0.0,(TEXT *)0, (TEXT *)0, 0, 0, -1.0); break;
	  case P_SUB:
	    hwd = new_WORD(s3, -1, 0.0, 0.0, 0.0,(TEXT *)0, (TEXT *)0, 0, 0, -1.0); break;
	  case P_DEL:
	    hwd = new_WORD((TEXT *)0, -1, 0.0, 0.0, 0.0,(TEXT *)0, (TEXT *)0, 0, 0, -1.0);
	    break;
	}
	if (hwd == (WORD *)0){
	  fprintf(stderr,"Error: Failed to alloc word in process_diff_line\n");
	  exit(1);
	}
	path->pset[path->num].b_ptr = hwd;		    
      case P_DEL:
	;
    }

    path->num++;
}

void set_temp_files(char *refwords, char *hypwords){
    int pid = getpid();
    int max=30, n=0, fail;
    struct stat fileinfo;

    do {
	sprintf(refwords,"/tmp/ref.%d",pid+n);
	sprintf(hypwords,"/tmp/hyp.%d",pid+n);
	n++;
	fail = (stat(refwords,&fileinfo) == 0) ||
	       (stat(hypwords,&fileinfo) == 0);
    } while (fail && (n < max));
    if (fail){
	fprintf(stderr,"Can't locate temp file names. Searched /tmp/ref.[%d-%d]\n",
		pid,pid+n);
	exit(1);
    }
}

SCORES *align_trans_mode_diff(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, enum id_types idt){
    FILE *fp_hyp, *fp_diff;
    TEXT *hyp_buff, *hyp_id, *spkr_id, *diff_out;
    int hyp_buff_len=LOAD_BUFF_LEN, hyp_id_len=LOAD_BUFF_LEN;
    int spkr_id_len=LOAD_BUFF_LEN, diff_out_len=LOAD_BUFF_LEN;
    PATH *path;
    SCORES *scor;
    int spk, rind;
    TEXT **reftran, **refid, **ind;
    TEXT *firsttoken;
    int refcnt, num_hyp=0;
    TEXT *ctext = hyp_buff;               /* current position is text string */
    char refwords[100], hypwords[100];

    alloc_singZ(diff_out,diff_out_len,TEXT,'\0');
    alloc_singZ(hyp_buff,hyp_buff_len,TEXT,'\0');
    alloc_singZ(hyp_id,hyp_id_len,TEXT,'\0');
    alloc_singZ(spkr_id,spkr_id_len,TEXT,'\0');

    set_temp_files(refwords, hypwords);

    /* initialize the score structure */
    scor = SCORES_init(set_title,20);
    scor->ref_fname = (char *)TEXT_strdup((TEXT *)ref_file);
    scor->hyp_fname = (char *)TEXT_strdup((TEXT *)hyp_file);
    scor->creation_date = get_date();

    /* load the required reference sentences */
    if (feedback >= 1)	printf("    Loading transcripts\n");
    load_refs(scor, hyp_file, ref_file, &reftran, &refid, &refcnt, case_sense); 

    /* create a file of words for both the ref and hyp */
    create_word_lists(scor, reftran, refid, refcnt, hyp_file,
		      refwords, hypwords, &num_hyp, case_sense);

    /* use readpipe to open a process without creating a temporary file */
    if (feedback >= 1)	printf("    Executing diff\n");
    if ((fp_diff = readpipe (DIFF_PROGRAM, "-y",refwords,
			     hypwords, NULL)) ==NULL){
	fprintf(stderr,"Error: Execution of 'diff' failed\n");
	exit(1);
    }
    
    if ((fp_hyp = fopen(hyp_file,"r")) == NULL){
        fprintf(stderr,"Can't open input hypothesis file %s\n",hyp_file);
        exit(1);
    }
    while (!feof(fp_hyp)){
        *hyp_buff  = *(hyp_buff+1) = COMMENT_CHAR;
        *(hyp_buff+2) = NULL_TEXT;
 
        while (!feof(fp_hyp) && (parse_input_comment_line(scor,hyp_buff) ||
                                 TEXT_is_comment_info(hyp_buff) ||
                                 TEXT_is_empty(hyp_buff)))
            if (TEXT_ensure_fgets(&hyp_buff,&hyp_buff_len,fp_hyp) == NULL)
               *hyp_buff = '\0';

        if (feof(fp_hyp))  break;
	if (!case_sense)    TEXT_str_case_change_with_mem_expand(&hyp_buff, &hyp_buff_len, 1);
	remove_id(hyp_buff,&hyp_id,&hyp_id_len);
	extract_speaker(hyp_id,spkr_id,idt);
	spk = SCORES_get_grp(scor,(char *)spkr_id);

	if ((ind = (TEXT **)bsearch(hyp_id,refid,refcnt,sizeof(TEXT *),
				    bsearch_TEXT_strcmp)) == NULL){
	    fprintf(stderr,"Error: Unable to locate Ref transcript for '%s'\n",
		    hyp_id);
	    exit(1);
	}
	rind = ind - refid;
	--num_hyp;

	if (feedback >= 3)
	    printf("REF: %s\nHYP: %s\n",reftran[rind],hyp_buff);

	path = PATH_alloc(100);

	ctext = tokenize_TEXT_first_alt(reftran[rind],(TEXT *)" \t\n");
	while (ctext != NULL) {
	    do {
		check_space_in_PATH(path);
		if (TEXT_ensure_fgets(&diff_out,&diff_out_len,fp_diff) == 
			NULL)
		    *diff_out = '\0';
		if (feof(fp_diff)) break;
		process_diff_line(diff_out,path,&firsttoken);
	    } while (!feof(fp_diff) && 
		     ((TEXT_strcmp(firsttoken,ctext) != 0) || num_hyp==0));
	    ctext = tokenize_TEXT_first_alt(NULL,(TEXT *)" \t\n");
	}

	PATH_add_utt_id(path,(char *)hyp_id);
	PATH_set_sequence(path);
	if (case_sense) BF_SET(path->attrib,PA_CASE_SENSE);
	
	if (feedback >= 2){
	    printf("\n");
	    PATH_print(path,stdout,199);
	    printf("\n");
	}

	add_PATH_score(scor,path,spk, keep_path);

	if (!keep_path)  PATH_free(path);
    }
    fclose(fp_diff);
    fclose(fp_hyp);

    if (num_hyp != 0){
	fprintf(stderr,"Error: Unexpected number of output utts.  Missing %d\n",num_hyp);
	exit(1);
    }

    free_singarr(diff_out,TEXT);
    free_singarr(hyp_buff,TEXT);
    free_singarr(hyp_id,TEXT);
    free_singarr(spkr_id,TEXT);
    free_refs(&reftran, &refid, refcnt);

    unlink(refwords);
    unlink(hypwords);
    return(scor);
}

SCORES *align_ctm_to_stm_diff(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, enum id_types idt){
    char *proc="align_ctm_to_stm";
    FILE *fp_ref, *fp_hyp;
    WTOKE_STR1 *hyp_segs;
    int ref_eof, ref_err, hyp_eof, hyp_err;
    int hyp_end_chan1, ref_end_chan1;
    int hyp_end_chan2, ref_end_chan2;
    int just_read, hyp_file_end, ref_file_end;
    int number_of_channels;
    int i;
    PATH *path;
    STM *stm;
    SCORES *scor;
    int spkr, r;
    char refwords[100], hypwords[100];
    TEXT *diff_out, *firsttoken, *ctext;
    static int diff_out_len=100;
    FILE *fp_diff;
    int pdb = 0;

    /* allocate memory */
    hyp_segs = WTOKE_STR1_init(hyp_file);
    scor = SCORES_init(set_title,20);
    scor->ref_fname = (char *)TEXT_strdup((TEXT *)ref_file);
    scor->hyp_fname = (char *)TEXT_strdup((TEXT *)hyp_file);
    scor->creation_date = get_date();

    alloc_singZ(diff_out,diff_out_len,TEXT,'\0');
    
    set_temp_files(refwords, hypwords);

    if ((fp_hyp = fopen(hyp_file,"r")) == NULL){
        fprintf(stderr,"Can't open input hypothesis file %s\n",hyp_file);
        exit(1);
    }
    if ((fp_ref = fopen(ref_file,"r")) == NULL){
        fprintf(stderr,"Can't open input reference file %s\n",ref_file);
        exit(1);
    }

    if ((stm = alloc_STM(200)) == (STM *)0) {
	fprintf(stderr,"%s: Failed to alloc STM struct\n",proc);
	exit(1);
    }
    hyp_eof = ref_eof = 0;
    do {
        do {
	    locate_WTOKE_boundary(hyp_segs, hyp_segs->s, 1, 0, &hyp_file_end);
            just_read = 0;
            if ((hyp_file_end == hyp_segs->n) && !hyp_eof){
                fill_mark_struct(fp_hyp,hyp_segs,hyp_file,&hyp_eof,&hyp_err, 
				 case_sense);
                if (hyp_err != 0){
                    fprintf(stdout,"; *Err: Error detected in hyp file '%s'\n",
                            hyp_file);
                    exit(1);
                }
                just_read = 1;
            }
        } while (just_read);
	just_read = 0;
        do {
	    locate_STM_boundary(stm, stm->s, 1, 0, &ref_file_end);
            if ((ref_file_end == stm->num) && !ref_eof){
		if (just_read == 1){
		    just_read = 0;
		    expand_STM(stm);
		}
		fill_STM(fp_ref, stm, ref_file, &ref_eof, case_sense,&ref_err);
                if (hyp_err != 0){
                    fprintf(stdout,"; *Err: Error detected in hyp file '%s'\n",
                            hyp_file);
                    exit(1);
                }
                just_read = 1;
            } else 
		just_read = 0;
        } while (just_read);

	/* find the end of the second channel for the hyp file */
	locate_WTOKE_boundary(hyp_segs, hyp_segs->s, 1, 1, &hyp_end_chan1);
	if (hyp_end_chan1+1 < hyp_segs->n &&
            strcmp(hyp_segs->word[hyp_segs->s].conv,
		   hyp_segs->word[hyp_end_chan1+1].conv)==0){
            locate_WTOKE_boundary(hyp_segs,hyp_end_chan1+1,1,1,&hyp_end_chan2);
            number_of_channels = 2;
        } else {
            /* There was no second channel */
            hyp_end_chan2 = hyp_end_chan1;
            number_of_channels = 1;
        }

	/* find the end of the second channel for the ref file */
	locate_STM_boundary(stm, stm->s, 1, 1, &ref_end_chan1);
	if (ref_end_chan1+1 < stm->num &&
            TEXT_strcmp(stm->seg[stm->s].file,
			stm->seg[ref_end_chan1+1].file)==0){
	    if (number_of_channels != 2){
		fprintf(stderr,"%s: Channel Mis-match, 1 in Hyp, 2 in Ref\n",
			proc);
		return((SCORES *)0);
	    }
	    locate_STM_boundary(stm, ref_end_chan1+1, 1, 1, &ref_end_chan2);
        } else {
            /* There was no second channel */
	    if (number_of_channels != 1){
		fprintf(stderr,"%s: Channel Mis-match, 2 in Hyp, 1 in Ref\n",
			proc);
		return((SCORES *)0);
	    }
            ref_end_chan2 = ref_end_chan1;
        }

	/* give some user feedback */
	if (feedback >= 1)
	    printf("    Performing alignments for file '%s'.\n",
		   stm->seg[stm->s].file);
	if (db+pdb >= 5){
	    printf("%s: hyp CTM File Range [%d,%d]",
		   proc,hyp_segs->s,hyp_file_end);
	    printf("   Chan1 Range[%d,%d] ",hyp_segs->s,hyp_end_chan1);
	    printf("  Chan2 Range[%d,%d]\n",hyp_end_chan1+1,hyp_end_chan2);
	    if (db+pdb >= 10){
		dump_word_tokens2(hyp_segs, hyp_segs->s, hyp_file_end);
		printf("\n");
	    }
	    printf("%s: ref STM File Range [%d,%d]",proc,stm->s,ref_file_end);
	    printf("   Chan1 Range[%d,%d] ",stm->s, ref_end_chan1);
	    printf("  Chan2 Range[%d,%d]\n",ref_end_chan1+1,ref_end_chan2);
	    if (db+pdb >= 10) dump_STM(stm, stm->s, ref_file_end);
	    printf("\n");
	}
	/* Align each channel using diff */
	for (i=0; i<number_of_channels; i++){
	    int h_st, h_end, r_st, r_end;

	    if (db+pdb >= 5) printf("\n%s: Starting Channel %d\n",proc,i);
	    if (i==0){
		h_st = hyp_segs->s;       h_end = hyp_end_chan1;
		r_st = stm->s;            r_end = ref_end_chan1;
	    } else {
		h_st = hyp_end_chan1+1;   h_end = hyp_end_chan2;
		r_st = ref_end_chan1+1;   r_end = ref_end_chan2;
	    }

	    if (strcmp(hyp_segs->word[h_st].conv,
		       (char *)stm->seg[r_st].file) != 0){
		fprintf(stderr,"%s: File identifiers do not match,\nhyp ",proc);
		fprintf(stderr,"file '%s' and ref file '%s' not synchronized\n",
			hyp_segs->word[h_st].conv,stm->seg[r_st].file);
		return((SCORES *)0);
	    }
	    if (strcmp(hyp_segs->word[h_st].turn,
		       (char *)stm->seg[r_st].chan) != 0){
		fprintf(stderr,"%s: channel identifiers do not match, ",proc);
		fprintf(stderr,"hyp file '%s' and ref file '%s' not",
			hyp_segs->word[h_st].conv,stm->seg[r_st].file);
		fprintf(stderr," synchronized\n");
		return((SCORES *)0);
	    }

	    /* write out the words in the STM file */
	    dump_STM_words(stm,r_st,r_end+((r_end==stm->num)?-1:0),refwords);
	    dump_WTOKE_words(hyp_segs, h_st, h_end, hypwords);

	    if ((fp_diff = readpipe (DIFF_PROGRAM, "-y",refwords, hypwords, NULL)) == NULL){
		fprintf(stderr,"Error: Execution of 'diff' failed\n");
		exit(1);
	    }

	    for (r=r_st; r<=r_end + ((r_end==stm->num)?-1:0); r++){
		path = PATH_alloc(100);
		path->num = 0;

		/* for each reference word, located it's matching diff output*/
		ctext = tokenize_TEXT_first_alt(stm->seg[r].text,
						(TEXT*)" \t\n");
		while (ctext != NULL){
 		    do {
			if (TEXT_ensure_fgets(&diff_out,&diff_out_len,fp_diff)
			    == NULL)
			    *diff_out = '\0';
			if (feof(fp_diff)) break;
			process_diff_line(diff_out,path,&firsttoken);
		    } while (!feof(fp_diff) && 
			     ((TEXT_strcmp(firsttoken,ctext) != 0) ||
			      r== r_end + (r_end==stm->num)?-1:0));
		    ctext = tokenize_TEXT_first_alt(NULL,(TEXT *)" \t\n");
		}
		spkr = SCORES_get_grp(scor,(char *)stm->seg[r].spkr);
		PATH_add_utt_id(path,rsprintf("(%s-%03d)",stm->seg[r].spkr,
					      scor->grp[spkr].nsent));
		PATH_set_sequence(path);
		PATH_add_label(path,(char *)stm->seg[r].labels);
		PATH_add_file(path,(char *)stm->seg[r].chan);
		PATH_add_channel(path,(char *)stm->seg[r].chan);
		if (case_sense) BF_SET(path->attrib,PA_CASE_SENSE);
		add_PATH_score(scor,path,spkr, keep_path);
		
		if (feedback >= 2){
		    printf("\n");
		    PATH_print(path,stdout,199);
		    printf("\n");
		}
		if (!keep_path)  PATH_free(path);
	    }
	    fclose(fp_diff);

    
	    stm->s = r_end+1;
	} /* for each channel */

	/* increment the start pointers */
	hyp_segs->s = hyp_end_chan2+1;
	stm->s = ref_end_chan2+1;
    } while (hyp_segs->s <= hyp_segs->n);

    fclose(fp_hyp);
    fclose(fp_ref);

    unlink(refwords);
    unlink(hypwords);
    free_singarr(diff_out,TEXT);

    free_mark_file(hyp_segs);
    free_STM(stm);

    return(scor);
}

void convert_text_to_word_list(char *file, char *words,int case_sense){
    FILE *fp, *fp_out;
    TEXT *buf, *ctext;
    int buf_len=100;

    if ((fp=fopen(file,"r")) == NULL){
	fprintf(stderr,"Error: Unable to open input STM file '%s'\n",file);
	exit(1);
    }

    if ((fp_out=fopen(words,"w")) == NULL){
	fprintf(stderr,"Error: Unable to open STM word file '%s'\n",words);
	exit(1);
    }

    alloc_singZ(buf,buf_len,TEXT,'\0');

    /* for each stm, load the aligned text and create a path as before */
    while (!feof(fp)){
	/* read the next ref line */
	*buf = '\0';
        while (!feof(fp) && (TEXT_is_comment(buf) ||
			     TEXT_is_comment_info(buf) ||
			     TEXT_is_empty(buf)))
	    if (TEXT_ensure_fgets(&buf,&buf_len,fp) == NULL)
		*buf = '\0';
	if (*buf == '\0' && feof(fp)) break;
	if (!case_sense)  TEXT_str_case_change_with_mem_expand(&buf, &buf_len, 1);

	/* for each reference word, located it's matching diff output */
	ctext = TEXT_strtok(buf,(TEXT *)" \t\n");
	while (ctext != NULL){
	    fprintf(fp_out,"%s\n",ctext);
	    ctext = TEXT_strtok(NULL,(TEXT *)" \t\n");
	}
    }
    fclose(fp);
    fclose(fp_out);
    free_singarr(buf,TEXT);
}

SCORES *align_text_to_stm(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, enum id_types idt){
    /* char *proc="align_text_to_stm"; */
    FILE *fp_ref;
    PATH *path;
    SCORES *scor;
    int spkr, num_ref;
    char refwords[100], hypwords[100];
    TEXT *diff_out, *ref_buf, *firsttoken, *ctext;
    int diff_out_len=100, ref_buf_len=100;
    FILE *fp_diff;
    STM_SEG ref_seg;

    /* allocate memory */
    scor = SCORES_init(set_title,20);
    scor->ref_fname = (char *)TEXT_strdup((TEXT *)ref_file);
    scor->hyp_fname = (char *)TEXT_strdup((TEXT *)hyp_file);
    scor->creation_date = get_date();

    alloc_singZ(diff_out,diff_out_len,TEXT,'\0');
    alloc_singZ(ref_buf,ref_buf_len,TEXT,'\0');

    set_temp_files(refwords, hypwords);

    /* stm to word texts */
    convert_stm_to_word_list(ref_file,refwords,case_sense,&num_ref);

    /* text to word separeted texts */
    convert_text_to_word_list(hyp_file,hypwords,case_sense);

    /* start up diff */
    if ((fp_diff = readpipe (DIFF_PROGRAM,"-y",refwords, hypwords, NULL)) ==
	NULL){
	fprintf(stderr,"Error: Execution of 'diff' failed\n");
	exit(1);
    }

    /* again, open the input reference file */
    if ((fp_ref = fopen(ref_file,"r")) == NULL){
        fprintf(stderr,"Can't open input reference file %s\n",ref_file);
        exit(1);
    }

    /* for each stm, load the aligned text and create a path as before */
    while (!feof(fp_ref)){
	/* read the next ref line */
	read_stm_line(&ref_buf,&ref_buf_len,fp_ref);
	if (feof(fp_ref)) break;

	/* parse the reference transcript */
	parse_stm_line(&ref_seg, &ref_buf, &ref_buf_len, case_sense, 0);
	num_ref --;
	
	/* allocate a path structure */
	path = PATH_alloc(100);	path->num = 0;

	/* for each reference word, located it's matching diff output */
	ctext = tokenize_TEXT_first_alt(ref_seg.text,(TEXT *)" \t\n");
	while (ctext != NULL){
	    do {
		if (TEXT_ensure_fgets(&diff_out,&diff_out_len,fp_diff) == NULL)
		    *diff_out = '\0';
		if (feof(fp_diff)) break;
		process_diff_line(diff_out,path,&firsttoken);
	    } while (!feof(fp_diff) && 
		     ((TEXT_strcmp(firsttoken,ctext) != 0) || num_ref==0));
	    ctext = tokenize_TEXT_first_alt(NULL,(TEXT *)" \t\n");
	}

	spkr = SCORES_get_grp(scor,(char *)ref_seg.spkr);
	PATH_add_utt_id(path,rsprintf("(%s-%03d)",ref_seg.spkr,
				      scor->grp[spkr].num_path));
	PATH_set_sequence(path);
	PATH_add_label(path,(char *)ref_seg.labels);
	PATH_add_file(path,(char *)ref_seg.file);
	PATH_add_channel(path,(char *)ref_seg.chan);
	if (case_sense) BF_SET(path->attrib,PA_CASE_SENSE);
	add_PATH_score(scor,path,spkr, keep_path);
	
	if (feedback >= 2){
	    printf("\n");
	    PATH_print(path,stdout,199);
	    printf("\n");
	}
	if (!keep_path)  PATH_free(path);
	free_STM_SEG(&ref_seg);
    }
    fclose(fp_diff);
    fclose(fp_ref);

    free_singarr(diff_out,TEXT);
    free_singarr(ref_buf,TEXT);

    unlink(refwords);
    unlink(hypwords);

    return(scor);
}
