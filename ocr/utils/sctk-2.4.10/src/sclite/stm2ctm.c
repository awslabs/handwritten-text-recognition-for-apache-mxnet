#include "sctk.h"

static int align_one_channel(SCORES *scor, int chan, WTOKE_STR1 *hyp_segs, STM *stm, int h_st, int h_end, int r_st, int r_end, int keep_path, int feedback, char *proc, int case_sense, int char_align, char *lexicon, int infer_word_seg, int inf_no_ascii, int fcorr, int opt_del, int reduce_ref, int reduce_hyp, int left_to_right, WWL *wwl, char *lm_file);
void segment_hyp_for_utt(WTOKE_STR1 *hyp_segs, STM *stm, int *curhyp, int *curhend, int h_st, int h_end, int rs, int r_st, int r_end);

static int align_one_channel(SCORES *scor, int chan, WTOKE_STR1 *hyp_segs, STM *stm, int h_st, int h_end, int r_st, int r_end, int keep_path, int feedback, char *proc, int case_sense, int char_align, char *lexicon, int infer_word_seg, int inf_no_ascii, int fcorr, int opt_del, int reduce_ref, int reduce_hyp, int left_to_right, WWL *wwl, char *lm_file){
    int curhyp, curhend, rs;
    int spkr;
    NETWORK *hnet;
    PATH *path;
    char *id;
    int skips = 0, skip_words=0;
    int ignored_refseg = 0;
    int ignored_hypword = 0;
    int ignore_segment = 0;

//    if (TEXT_strcmp((TEXT*)hyp_segs->word[h_st].conv,stm->seg[r_st].file)!= 0){
//	fprintf(stderr,"%s: File identifiers do not match,\nhyp ",proc);
//	fprintf(stderr,"file '%s' and ref file '%s' not synchronized\n",
//		hyp_segs->word[h_st].conv,stm->seg[r_st].file);
//	return(0);
//    }
//    if (TEXT_strcmp((TEXT *)hyp_segs->word[h_st].turn,stm->seg[r_st].chan)!=0){
//	fprintf(stderr,"%s: channel identifiers do not match, ",proc);
//	fprintf(stderr,"hyp file '%s' and ref file '%s' not",
//		hyp_segs->word[h_st].conv,stm->seg[r_st].file);
//	fprintf(stderr," synchronized\n");
//	return(0);
//    }

    /* set flags within the ref and hyp structure to enable reductions */
    reset_WTOKE_flag(hyp_segs, "ignore");
    for (rs=r_st; rs<stm->num && rs<=r_end; rs++)
	stm->seg[rs].flag1 = 1;

    if (reduce_ref){
	/* search the stm structure, marking the segments that end */
	/* before the first hyp words, or begin after the last hyp */
	/* word by setting the segments "flag1" variable */
	for (rs=r_st; rs<stm->num && rs<=r_end; rs++){
	    if (h_end >= h_st)
		if ((hyp_segs->word[h_st].t1 >= stm->seg[rs].t2) ||
		    (hyp_segs->word[h_end].t1+hyp_segs->word[h_end].dur <=
		     stm->seg[rs].t1))
		    stm->seg[rs].flag1 = 0;
	    if (feedback >= 10){
		if (!stm->seg[rs].flag1) printf("Segment %d Ignored\n",rs);
	    }
	}
    }
    
    if (reduce_hyp){
	/* search the wtoke structure, MARKING any words that          */
	/* have a time mark less than the beginning of the first ref-   */
	/* erence segment, or occur after the end of the last reference */
	/* segment.  */
	float bt=9999999, et=-999999; 
	int xx;

	/* measure the begin and end times for the reference file */
	for (rs=r_st; rs<stm->num && rs<=r_end; rs++)
	    if (stm->seg[rs].flag1){
		if (bt > stm->seg[rs].t1) bt = stm->seg[rs].t1;
		if (et < stm->seg[rs].t2) et = stm->seg[rs].t2;
	    }
	if (db >= 10) printf("      bt= %f   et= %f\n",bt, et);
	/* loop through the hyp words, MARKING the out-of-bounds */
	/* words that are NOT alternation markers */
	for (xx = h_st; xx <= h_end; xx++)
	    if (TEXT_strCcasecmp((TEXT *)"<ALT",
				 hyp_segs->word[xx].sp,4) != 0){
		if ((hyp_segs->word[xx].t1 > et) ||
		    (hyp_segs->word[xx].t1 + hyp_segs->word[xx].dur < bt)){
		    hyp_segs->word[xx].ignore = TRUE;		    
		    if (feedback >= 10)
			printf("Ignore word %d %s\n",xx,
			       hyp_segs->word[xx].sp);
		}
	    }
	/* GO back to make sure if any part of an alternate is legal, */
	/* then the entire alternate is legal !!!! */
	for (xx = h_st; xx <= h_end; xx++){
	    int yy, usable;
	    if (TEXT_strcasecmp((TEXT *)"<ALT_BEGIN>",
				hyp_segs->word[xx].sp) == 0) {
		/* first search for and <ALT_END>, checking word status */
		for (yy = xx, usable = 0;
		     (yy <= h_end &&
		      TEXT_strcasecmp((TEXT *)"<ALT_END>",
				      hyp_segs->word[yy].sp) != 0); yy++)
		    if (TEXT_strCcasecmp((TEXT *)"<ALT",
					 hyp_segs->word[yy].sp,4) != 0)
			if (!hyp_segs->word[yy].ignore) usable++;
		/* set the ignore flag based on the outcome of the */
		/* usable varaible */
		for (yy = xx;
		     (yy <= h_end &&
		      TEXT_strcasecmp((TEXT *)"<ALT_END>",
				      hyp_segs->word[yy].sp) != 0); yy++)
		    hyp_segs->word[yy].ignore= (usable > 0) ? FALSE : TRUE;
	    }
	}
	for (xx = h_st; xx <= h_end; xx++)
	    if (hyp_segs->word[xx].ignore &&
		(TEXT_strCcasecmp((TEXT *)"<ALT",
				  hyp_segs->word[xx].sp,4) != 0))
		skip_words ++;
    }

    reset_WTOKE_flag(hyp_segs, "comment");
	    
    curhyp = h_st;
    curhend = h_st-1;
    for (rs=r_st; rs<stm->num && rs<=r_end; rs++){
        ignore_segment = 0;

	if (reduce_ref || reduce_hyp){
	    if (!stm->seg[rs].flag1)
		skips++;
	    if (feedback >= 1){
		printf("\r        %d of %d Segments (",
		       rs+1 - stm->s,
		       r_end - stm->s + ((r_end==stm->num)?0:1));
		if (reduce_ref) { printf("%d skipped segs",skips); }
		if (reduce_ref && reduce_hyp) printf(", ");
		if (reduce_hyp) { printf("%d skipped words",skip_words); }
		printf(") For Channel %d. ", chan+1);
	    }
	    if (!stm->seg[rs].flag1){
	        if (ignored_refseg > 0){
		    printf(" (%d Ignored Segments, %d Ignored Hyp Words.)    ",
			   ignored_refseg,ignored_hypword);
		}
		continue;
	    }
	} else {
	    if (feedback >= 1)
		printf("\r        %d of %d Segments For Channel %s. ",
		       rs+1 - stm->s,r_end - stm->s + ((r_end==stm->num)?0:1),
		       stm->seg[rs].chan);
	}
	fflush(stdout);
	if (db >= 5) {
	    printf("\n%s: Aligning segment %d\n",proc,rs);
	    if (db >= 10) {
	        printf("%d: %s chan: %s  ",
		       rs,stm->seg[rs].file,stm->seg[rs].chan);
		printf("Spkr: %s  T1: %f  T2:%f  Text: '%s'\n",
		       stm->seg[rs].spkr,stm->seg[rs].t1,
		       stm->seg[rs].t2,stm->seg[rs].text);
	    }
	}

	segment_hyp_for_utt(hyp_segs, stm, &curhyp, &curhend, h_st, h_end,
			    rs, r_st, r_end);

	if ((TEXT_strstr(stm->seg[rs].text,
			(TEXT *)"IGNORE_TIME_SEGMENT_IN_SCORING") == (TEXT*)0) &&
	    (TEXT_strstr(stm->seg[rs].text,
			(TEXT *)"IGNORETIMESEGMENTINSCORING") == (TEXT*)0)){
	  ;
	} else {
	    ignored_refseg++;
	    ignored_hypword += curhend - curhyp + 1;
	    ignore_segment = 1;
	}

	if (feedback >= 1)
	    if (ignored_refseg > 0)
	        printf(" (%d Ignored Segments, %d Ignored Hyp Words.)    ",
		       ignored_refseg,ignored_hypword);
	    else
	        printf("    ");

	if (ignore_segment){
	    curhyp = curhend + 1;
	    continue;
	}

	if (db >= 10){
	    printf("Hyp set [%d,%d]\n",curhyp,curhend);
	    dump_word_tokens2(hyp_segs, curhyp, curhend);
	}
	
	/* create the utterance ID */
	spkr = SCORES_get_grp(scor,(char *)stm->seg[rs].spkr);
	
	/* Actually do the alingment */
	if ((hnet=Network_create_from_WTOKE(hyp_segs,curhyp,curhend,
					    "Hypothesis net",
					    print_WORD_wt,
					    equal_WORD2,
					    release_WORD, null_alt_WORD,
					    opt_del_WORD,
					    copy_WORD, make_empty_WORD,
					    use_count_WORD, left_to_right))
	    == NULL_NETWORK){
	    fprintf(stderr,"%s: Network_create_from_WTOKE failed\n",
		    proc);
	    return(0);
	}
	
	id = (char *)TEXT_strdup((TEXT *)rsprintf("(%s-%03d)",
						  stm->seg[rs].spkr,
						  scor->grp[spkr].nsent));
	if (infer_word_seg == 0)
	    path = network_dp_align_texts(stm->seg[rs].text, (NETWORK *)0,
					  (TEXT *)0, hnet, char_align, case_sense,
					  id, fcorr, opt_del, FALSE, wwl, lm_file);
	else if (infer_word_seg == INF_SEG_ALGO1)
	    path = infer_word_seg_algo1(stm->seg[rs].text, (TEXT *)0,
					hnet, case_sense,id,lexicon,
					fcorr, opt_del, inf_no_ascii);
	else if (infer_word_seg == INF_SEG_ALGO2)
	    path = infer_word_seg_algo2(stm->seg[rs].text, (TEXT *)0,
					hnet, case_sense,id,lexicon,
					fcorr, opt_del, inf_no_ascii);
	else{
	    fprintf(scfp,"Error: unknown alignment procedure\n");
	    exit(1);
	}
	
	free_singarr(id,char);
	
	add_PATH_score(scor,path,spkr, keep_path);
	PATH_add_label(path,(char *)stm->seg[rs].labels);
	PATH_add_file(path,(char *)stm->seg[rs].file);
	PATH_add_channel(path,(char *)stm->seg[rs].chan);
	if (case_sense) BF_SET(path->attrib,PA_CASE_SENSE);
	if (char_align) BF_SET(path->attrib,PA_CHAR_ALIGN);
	BF_SET(path->attrib,PA_HYP_WTIMES);
	BF_SET(path->attrib,PA_REF_TIMES);
	path->ref_t1 = stm->seg[rs].t1;
	path->ref_t2 = stm->seg[rs].t2;
	if (hyp_segs->has_conf) BF_SET(path->attrib,PA_HYP_CONF);
	
	if (feedback >= 2){
	    printf("\n");
	    if (feedback >= 4)
		printf("REFTIMES:  T1: %f  T2: %f\n",
		       stm->seg[rs].t1,stm->seg[rs].t2);
	    PATH_print(path,stdout,199);
	    printf("\n");
	}
	
	if (!keep_path) PATH_free(path);
	
	curhyp = curhend + 1;
    }
    stm->s = r_end+1;
    if (feedback >= 1)
	printf("\n");
    return(1);
}


/*******  Segment the hypothesis words into an utterance *****/

void segment_hyp_for_utt(WTOKE_STR1 *hyp_segs, STM *stm, int *curhyp, int *curhend, int h_st, int h_end, int rs, int r_st, int r_end){

    if (rs == r_end || rs == (stm->num-1)) {
	/* The last utterance, through everything in */
	*curhend = h_end;
/* #define NEW1 */
#ifdef NEW1
	/* Diagnostic code */
	{   int x=0;
	    for (x=0; ((h_end+x >= *curhyp) &&
		       ((hyp_segs->word[h_end+x].t1 +
			 hyp_segs->word[h_end+x].dur/2.0) >
			stm->seg[rs].t2));
		 x--){
		printf("\nWarning: orphaned tail word!'%s'  (%.2f,%.2f) REF(%.2f)\n",
		       hyp_segs->word[h_end+x].sp,
		       hyp_segs->word[h_end+x].t1,
		       hyp_segs->word[h_end+x].t1  + 
		       hyp_segs->word[h_end+x].dur, 
		       stm->seg[rs].t2);
	    }
	}
#endif
    } else {
	/* find the end, Always include anything before the start time */
	int cont = 1, ch;
	double mid, nmid;
	
#ifdef NEW1
	/* Separate out any words which do not belong to ANY */
	/* segment.  If any exist, load them into the (special) */
	/* speaker variable */
	for (w=0; ((*curhend+1+w <= h_end) &&
		   ((hyp_segs->word[*curhend+1+w].t1 +
		     hyp_segs->word[*curhend+1+w].dur/2.0) <
		    stm->seg[rs].t1));
	     w++)
	    ;
	if (w > 0){  /* then Words need to be handled */
	    for (ww=0; ww < w; ww++){
		printf("\nWarning: orphaned word!'%s'  (%.2f,%.2f) REF(%.2f)\n",
		       hyp_segs->word[*curhend+1+ww].sp,
		       hyp_segs->word[*curhend+1+ww].t1,
		       hyp_segs->word[*curhend+1+ww].dur, 
		       stm->seg[rs].t1);
	    }
	}
#endif
	while (*curhend+1 <= h_end && cont){
	    if (! hyp_segs->word[*curhend+1].alternate) 
		mid = (hyp_segs->word[*curhend+1].t1 +
		       hyp_segs->word[*curhend+1].dur/2.0);
	    else {
		/* measure the bigest mid in the alternate */
		for (ch = *curhend+1, mid=0;
		     ch <= h_end &&
		     TEXT_strcasecmp((TEXT *)"<ALT_END>",
				     hyp_segs->word[ch].sp) != 0;
		     ch++)
		    if (TEXT_strCcasecmp((TEXT *)"<ALT",
					 hyp_segs->word[ch].sp,4)
			!= 0){
			nmid = (hyp_segs->word[ch].t1 +
				hyp_segs->word[ch].dur/2.0);
			if (nmid > mid) mid = nmid;
		    }
	    }
	    if (mid >= stm->seg[rs].t2)
		cont = 0;
	    else {
		if (! hyp_segs->word[*curhend+1].alternate) 
		    (*curhend)++;
		else {
		    while (*curhend <= h_end &&
			   TEXT_strcasecmp((TEXT *)"<ALT_END>",
					   hyp_segs->word[*curhend+1].sp) != 0)
			(*curhend)++;
		    if (*curhend <= h_end) (*curhend)++;
		}
	    }
	}
    }
    
}

SCORES *align_ctm_to_stm_dp(char *ref_file, char *hyp_file, char *set_title, int keep_path, int case_sense, int feedback, int char_align, enum id_types idt, int infer_word_seg, char *lexicon, int fcorr, int opt_del, int inf_no_ascii, int reduce_ref, int reduce_hyp, int left_to_right, WWL *wwl, char *lm_file){
    char *proc="align_ctm_to_stm";
    FILE *fp_ref, *fp_hyp;
    WTOKE_STR1 *hyp_segs;
    int hend_chan1=0, rend_chan1=0;
    int hend_chan2=0, rend_chan2=0;
    int hyp_file_end=0, ref_file_end=0;
    int number_of_channels=0;
    STM *stm;
    SCORES *scor;
    TEXT *ref_chars=(TEXT *)0;
    int ref_chars_len=100;
    int i=0;

    /* allocate memory */
    hyp_segs = WTOKE_STR1_init(hyp_file);

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

    if (char_align){
	alloc_singZ(ref_chars,ref_chars_len,TEXT,'\0');
    }

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

    do {
        int hend, rend;
        // Fills the structures ensuring the fill next file was loaded
	fill_STM_structure(stm, fp_ref, ref_file, &ref_file_end, case_sense);
	fill_WTOKE_structure(hyp_segs, fp_hyp, hyp_file, &hyp_file_end,
			     case_sense);
        		     
        // We assume one channel will be processed so we check to see if it's scoreabe
        locate_WTOKE_boundary(hyp_segs, hyp_segs->s, 1, 1, &hend);
        locate_STM_boundary(stm, stm->s, 1, 1, &rend);
        
        if (feedback >= 1)
    	    printf("    Performing alignments for file '%s'.\n",
	       stm->seg[stm->s].file);
        if (db >= 5){
           printf("%s: hyp CTM File Range [%d,%d,%s,%s] ", proc,hyp_segs->s,hend,hyp_segs->word[hyp_segs->s].conv,hyp_segs->word[hyp_segs->s].turn);
           printf(" ref STM File Range [%d,%d,%s,%s]\n",stm->s,rend,stm->seg[stm->s].file,stm->seg[stm->s].chan);
        }
        
        if (hyp_segs->s > hyp_segs->n){
           fprintf(stderr,"%s: Hyp files ends before ref but continuing for ",proc);
	   fprintf(stderr,"ref file/channel '%s' '%s'.\n",stm->seg[stm->s].file,stm->seg[stm->s].chan);           
        } else if (hyp_segs->s <= hyp_segs->n && stm->s > stm->num){
           fprintf(stderr,"%s: Error: Hyp file has more data than ref file, beginning at ",proc);
	   fprintf(stderr,"hyp file/channel '%s' '%s'.\n",hyp_segs->word[hyp_segs->s].conv,hyp_segs->word[hyp_segs->s].turn);           
           return((SCORES *)0);
        } else if ( TEXT_strcmp((TEXT*)hyp_segs->word[hyp_segs->s].conv,stm->seg[stm->s].file) == 0 &&
             TEXT_strcmp((TEXT*)hyp_segs->word[hyp_segs->s].turn,stm->seg[stm->s].chan) == 0){
           // Align this channel as is
        } else {
   	   fprintf(stderr,"%s: File identifiers do not match but continuing. ",proc);
	   fprintf(stderr,"ref file/channel '%s' '%s', ",stm->seg[stm->s].file,stm->seg[stm->s].chan);
	   fprintf(stderr,"next hyp '%s' '%s'.\n",hyp_segs->word[hyp_segs->s].conv,hyp_segs->word[hyp_segs->s].turn);
           // Align the ref to nothing      
           hend = hyp_segs->s-1;
        }
    
        if (align_one_channel(scor, i, hyp_segs, stm, hyp_segs->s, hend, stm->s,
         		  rend, keep_path, feedback, proc, case_sense,
            		  char_align, lexicon, infer_word_seg,
            		  inf_no_ascii, fcorr, opt_del,
            		  reduce_ref, reduce_hyp, left_to_right, wwl, lm_file)
            != 1)
            return((SCORES *)0);
            
        // Increment and move on
        hyp_segs->s = hend+1;
        stm->s = rend+1;                
    } while ((hyp_segs->s <= hyp_segs->n) || (stm->s < stm->num));

    fclose(fp_hyp);
    fclose(fp_ref);
    free_mark_file(hyp_segs);
    free_STM(stm);

    /* load any labels in either hyp or ref file */
    load_comment_labels_from_file(scor, ref_file);
    load_comment_labels_from_file(scor, hyp_file);

    if (ref_chars != (TEXT *)0) free_singarr(ref_chars,TEXT);
    fill_STM_structure((STM *)0, (FILE *)NULL, (char *)0, (int *)NULL, (int)0);
    return(scor);
}
