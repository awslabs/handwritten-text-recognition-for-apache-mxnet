#include "sctk.h"

int chop_WTOKE_2(WTOKE_STR1 *ref, WTOKE_STR1 *hyp, int Rstart, int Hstart, int Rendconv, int Hendconv, int max_words, int *Rret_end, int *Hret_end);


SCORES *align_ctm_to_ctm(char *hyp_file, char *ref_file, char *set_title, int feedback, int fcorr, int opt_del, int case_sense, int time_align, int left_to_right, WWL *wwl, char *lm_file){

    char *proc = "align_ctm_to_ctm";
    WTOKE_STR1 *hyp;
    WTOKE_STR1 *ref;
    FILE *fp_hyp, *fp_ref;
    int hyp_conv_end, hyp_eof;
    int ref_conv_end, ref_eof;
    int hyp_chop_end, ref_chop_end;
    int i;
    PATH *path;
    int ref_end_chan1, hyp_end_chan1, ref_end_chan2, hyp_end_chan2;
    int ref_end, hyp_end, ref_begin, hyp_begin;
    int ref_opp_end, hyp_opp_end, ref_opp_begin, hyp_opp_begin;
    int dbg = TRUE;
    int chunk;
    char uttid[100];
    int spkr;
    int number_of_channels;
    SCORES *scor;
    NETWORK *hnet, *rnet;
    int keep_path = 1;
   
    hyp = WTOKE_STR1_init(hyp_file);
    ref = WTOKE_STR1_init(ref_file);

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

    dbg=0;

    if ((fp_hyp = fopen(hyp_file,"r")) == NULL){
        fprintf(stderr,"Can't open input hypothesis file %s\n",hyp_file);
        exit(1);
    }
    if ((fp_ref = fopen(ref_file,"r")) == NULL){
        fprintf(stderr,"Can't open input reference file %s\n",ref_file);
	exit(1);
    }

    hyp_eof = ref_eof = FALSE;
    do {
        if (dbg  > 5) printf("Starting do-while loop on conversations"
			     "--------------------------\n");
	fill_WTOKE_structure(hyp, fp_hyp, hyp_file, &hyp_eof, case_sense);
	fill_WTOKE_structure(ref, fp_ref, ref_file, &ref_eof, case_sense);
	
	if ((hyp->n < hyp->s) && (ref->n > ref->s)) {
	  fprintf(stderr,"Error: premature EOF in hyp file '%s' at conv"
		  "ersation '%s' of ref file '%s'\n",hyp_file,
		  ref->word[ref->s].conv,ref_file);
	  exit(1);
	}
	if ((hyp->n > hyp->s) && (ref->n < ref->s)) {
	  fprintf(stderr,"Error: premature EOF in ref file '%s' at conv"
		  "ersation '%s' of hyp file '%s'\n",ref_file,
		  hyp->word[hyp->s].conv,hyp_file);
	  exit(1);
	}
	
	locate_boundary(ref, ref->s, TRUE, FALSE, &ref_conv_end);
	locate_boundary(hyp, hyp->s, TRUE, FALSE, &hyp_conv_end);

	/* let's do some checking !!!!!! */
	/* Are the conversations in sync */
	if (strcmp(ref->word[ref->s].conv,hyp->word[hyp->s].conv)!=0){
	  fprintf(stderr,"Error: in ref file '%s' and hyp file '%s'\n",
		  ref_file, hyp_file);
	  fprintf(stderr,"       file strings out of "
		  "synchronization around ref-file='%s'  hyp-file='%s'\n",
		  ref->word[ref->s].conv,hyp->word[hyp->s].conv);
	  exit(1);
	}

	locate_boundary(ref, ref->s, TRUE, TRUE, &ref_end_chan1);
	locate_boundary(hyp, hyp->s, TRUE, TRUE, &hyp_end_chan1);

	number_of_channels = 1;
	hyp_end_chan2 = ref_end_chan2 = -1;
	if (ref_end_chan1+1 < ref->n &&
	    strcmp(ref->word[ref->s].conv,ref->word[ref_end_chan1+1].conv)==0){
	    locate_boundary(ref, ref_end_chan1+1, TRUE, TRUE, &ref_end_chan2);
	    number_of_channels = 2;
	}
	if (hyp_end_chan1+1 < hyp->n &&
	    strcmp(hyp->word[hyp->s].conv,hyp->word[hyp_end_chan1+1].conv)==0){
	    locate_boundary(hyp, hyp_end_chan1+1, TRUE, TRUE, &hyp_end_chan2);
	    number_of_channels = 2;
	}

	if (number_of_channels == 2){
	  /* either the ref, hyp or both had two channels.  make sure there */
	  /* are two channels in both */
	  if (hyp_end_chan2 == -1){
	    fprintf(stderr,"Error: in ref file '%s' and hyp file '%s'\n   ",
		    ref_file, hyp_file);
	    fprintf(stderr,"    Hyp is missing second channel of file '%s'\n",
		    ref->word[ref->s].conv);
	    exit(1);
	  }
	  if (ref_end_chan2 == -1){
	    fprintf(stderr,"Error: in ref file '%s' and hyp file '%s'\n   ",
		    ref_file, hyp_file);
	    fprintf(stderr,"    ref is missing second channel of file '%s'\n",
		    hyp->word[hyp->s].conv);
	    exit(1);
	  }
	  
	  
	} else {
	    /* There was no second channel */
	    ref_end_chan2 = ref_end_chan1;
	    hyp_end_chan2 = hyp_end_chan1;
	}

	if (feedback >= 1)
	    printf("    Performing %s alignments for file '%s'.\n",
		   (time_align?"Time-mediated":"Word"),
		   ref->word[ref->s].conv);

	if (dbg > 5){
	    printf("Located REF boundary: for %s %s -> %d,%d\n",
		   ref->word[ref->s].conv,
		   ref->word[ref->s].turn,ref->s,ref_conv_end);
	    printf("Located HYP boundary: for %s %s -> %d,%d\n",
		   hyp->word[hyp->s].conv,
		   hyp->word[hyp->s].turn,hyp->s,hyp_conv_end);

	    printf("Located REF Channel 1 boundary: for %d - %d\n",
		   ref->s,ref_end_chan1);
	    printf("Located HYP Channel 1 boundary: for %d - %d\n",
		   hyp->s,hyp_end_chan1);	
	    printf("Located REF Channel 2 boundary: for %d - %d\n",
		   ref_end_chan1+1,ref_end_chan2);
	    printf("Located HYP Channel 2 boundary: for %d - %d\n",
		   hyp_end_chan1+1,hyp_end_chan2);
	}


	/* Start the alignment */
	/* Side A */
	for (i=0; i<number_of_channels; i++){
	    int base_len;
	    chunk = 0;	    
	    if (i==0){
		ref_end = ref_end_chan1;
		hyp_end = hyp_end_chan1;
		ref_begin = ref->s;
		hyp_begin = hyp->s;

		ref_opp_end = ref_end_chan2;
		hyp_opp_end = hyp_end_chan2;
		ref_opp_begin = ref_end_chan1 + 1;
		hyp_opp_begin = hyp_end_chan1 + 1;
	    } else {
		ref_end = ref_end_chan2;
		hyp_end = hyp_end_chan2;
		ref_begin = ref_end_chan1 + 1;
		hyp_begin = hyp_end_chan1 + 1;

		ref_opp_end = ref_end_chan1;
		hyp_opp_end = hyp_end_chan1;
		ref_opp_begin = ref->s;
		hyp_opp_begin = hyp->s;
	    }
	    if (dbg > 5){
		printf("Starting channel %d\n",i);
		printf("Ref Range: %d - %d\n",ref_begin,ref_end);
		printf("Hyp Range: %d - %d\n",hyp_begin,hyp_end);
	    }
	    while (ref_begin<=ref_end || hyp_begin<=hyp_end){
		int removals;

		base_len = 50;
		while (! chop_WTOKE_2(ref, hyp, ref_begin, hyp_begin,
				      ref_end, hyp_end, base_len,
				      &ref_chop_end,&hyp_chop_end)){
		    /* printf("Expanding len from %d to %d\n",base_len,
		       base_len*2); */
		    base_len *= 2;
		}

		removals = 0;

		/* create the utterance ID */
		sprintf(uttid,"(%s-%s-%04d)",ref->word[ref_begin].conv,
			ref->word[ref_begin].turn,chunk);

		spkr = SCORES_get_grp(scor,
				      rsprintf("%s-%s",
					       ref->word[ref_begin].conv,
					       ref->word[ref_begin].turn));

		if (dbg > 5){
		    printf("Chunk %d: Starting for channel %d  id=%s",
			   chunk,i,uttid);
		    printf("    Ref Range: %d - %d",ref_begin,ref_chop_end);
		    printf("    Hyp Range: %d - %d\n",hyp_begin,hyp_chop_end);
		}
		

		/* create the networks to be aligned */
		if ((hnet=Network_create_from_WTOKE(hyp,hyp_begin,
					 hyp_chop_end,"Hypothesis net",
					 print_WORD_wt,
					 equal_WORD2,
				         release_WORD, null_alt_WORD,
						    opt_del_WORD,
						    copy_WORD,make_empty_WORD,
						    use_count_WORD,
						    left_to_right))
		    == NULL_NETWORK){ 
		    fprintf(stderr,"%s: Network_create_from_WTOKE failed\n",
			    proc);
		    return(0);
		}
		if ((rnet=Network_create_from_WTOKE(ref,ref_begin,
					 ref_chop_end,"Reference net",
					 print_WORD_wt,
					 equal_WORD2,
				         release_WORD, null_alt_WORD,
						    opt_del_WORD,
						    copy_WORD,make_empty_WORD,
						    use_count_WORD,
						    left_to_right))
		    == NULL_NETWORK){ 
		    fprintf(stderr,"%s: Network_create_from_WTOKE failed\n",
			    proc);
		    return(0);
		}

		path = network_dp_align_texts((TEXT *)0, rnet,
					      (TEXT *)0, hnet,
					      FALSE, case_sense, 
					      (char *)uttid, fcorr,
					      opt_del, time_align, wwl, lm_file);
		BF_SET(path->attrib,PA_HYP_WTIMES);
		BF_SET(path->attrib,PA_REF_WTIMES);
		if (ref->has_conf) BF_SET(path->attrib,PA_REF_CONF);
		if (hyp->has_conf) BF_SET(path->attrib,PA_HYP_CONF);

		sort_PATH_time_marks(path);

		if (dbg > 10){
		    printf("After Sorting\n");
		    PATH_print_wt(path, stdout);
		}

		add_PATH_score(scor,path,spkr, keep_path);
		PATH_add_file(path,ref->word[ref_begin].conv);
		PATH_add_channel(path,ref->word[ref_begin].turn);
		/* PATH_add_label(path,(char *)stm->seg[rs].labels); */
	
		ref_begin = ref_chop_end+1;
		hyp_begin = hyp_chop_end+1;

		chunk++;
	    }
	}


	hyp->s = hyp_conv_end+1;
	ref->s = ref_conv_end+1;

    } while (hyp->s <= hyp->n || ref->s <= ref->n);

    free_mark_file(ref);
    free_mark_file(hyp);

    fclose(fp_hyp);
    fclose(fp_ref);

    return (scor);
}

/*******************************************************************/
/*  Compute the overlap of two segments in time and return that    */
/*  time.  Negative times indicate no overlap.  There are 6 cases  */
/*  Case 1:    S1   t1  t2                                         */
/*             S2            t1  t2                                */
/*                                                                 */
/*  Case 2:    S1            t1  t2                                */
/*             S2   t1  t2                                         */
/*                                                                 */
/*  Case 3a:   S1   t1    t2                                       */
/*             S2      t1    t2                                    */
/*                                                                 */
/*  Case 3b:   S1   t1           t2                                */
/*             S2      t1    t2                                    */
/*                                                                 */
/*  Case 4a:   S1         t1    t2                                 */
/*             S2      t1           t2                             */
/*                                                                 */
/*  Case 4b:   S1         t1        t2                             */
/*             S2      t1       t2                                 */
/*                                                                 */
double overlap(double s1_t1, double s1_t2, double s2_t1, double s2_t2)
{ 
    double rval;
    char *rule;
    int dbg=0;
    /* Case 1: */
    if (s1_t2 < s2_t1)
	rule="Case 1", rval=(s1_t2 - s2_t1);
    /* Case 2: */
    if (s1_t1 > s2_t2)
	rule="Case 2", rval=(s2_t2 - s1_t1);
    /* Case 3: */
    if (s1_t1 < s2_t1){
	/* Case 3a: */
	if (s1_t2 < s2_t2)
	    rule="Case 3a", rval=(s1_t2 - s2_t1);
	else /* Case 3b: */
	    rule="Case 3b", rval=(s2_t2 - s2_t1);
    } else {  /* Case 4: */
	/* Case 4a: */
	if (s1_t2 < s2_t2)
	    rule="Case 4a", rval=(s1_t2 - s1_t1);
	else /* Case 4b: */
	    rule="Case 4b", rval=(s2_t2 - s1_t1);
    }
    if (dbg){
	printf("Overlap: s1_t1=%2.2f s1_t2=%2.2f",s1_t1,s1_t2);
	printf(" s2_t1=%2.2f s2_t2=%2.2f   Rule:%s  Rval=%2.2f\n",
	       s2_t1,s2_t2,rule,rval);
    }
    return(rval);
}


int chop_WTOKE_2(WTOKE_STR1 *ref, WTOKE_STR1 *hyp, int Rstart, int Hstart, int Rendconv, int Hendconv, int max_words, int *Rret_end, int *Hret_end)
{
    int Rend, Hend, Rend_sync, Hend_sync;
    char *proc="chop_WTOKE_2";
    int dbg=0, skipped_alt;
    double Rgap_t1, Rgap_t2, Hgap_t1, Hgap_t2;
    
    if (dbg)
	printf("%s:  Start Ref=%d  Start Hyp=%d  Max_words=%d\n",
	       proc,Rstart,Hstart,max_words);

    /* Find the end of this segment */
    /*     locate_boundary(ref, Rstart, TRUE, TRUE, &Rendconv);
	   locate_boundary(hyp, Hstart, TRUE, TRUE, &Hendconv); */

    if (dbg) printf("%s:  Conversation Rendconv=%d  Hendconv=%d\n",
		    proc,Rendconv,Hendconv);

    if ((Rstart > Rendconv) && (Hstart > Hendconv))
	return(0);

    /* if we've hit the end of the conversation, send everything */
    if (((Rendconv - Rstart) <max_words) && ((Hendconv - Hstart) < max_words)){
	*Rret_end = Rendconv; 
	*Hret_end = Hendconv; 
	if (dbg) printf("%s: R1 Success Chop Ref (%d-%d)  Hyp (%d-%d)\n",
			proc,Rstart,*Rret_end,Hstart,*Hret_end);
	return(1);
    }

    /* set the indexes to maximum length */
    Rend = MIN(Rstart + max_words, Rendconv);
    Hend = MIN(Hstart + max_words, Hendconv);
	
    if (dbg) printf("%s:  Limited Rend=%d  Hend=%d\n",proc,Rend,Hend);

    /* synchronize the ends to permit searching */
    while (Rend >= Rstart && ref->word[Rend].alternate)
	Rend--;
    skipped_alt = TRUE;
    while (skipped_alt){
	skipped_alt = FALSE;
	if (ref->word[Rend].t1 > hyp->word[Hend].t1)
	    while (!skipped_alt &&
		   Rend >= Rstart && 
		   ref->word[Rend].t1 > hyp->word[Hend].t1 && 
		   overlap(ref->word[Rend].t1,
			   ref->word[Rend].t1+ref->word[Rend].dur,
			   hyp->word[Hend].t1,
			   hyp->word[Hend].t1+hyp->word[Hend].dur) < 0.0)
		for (Rend--; Rend >= Rstart && ref->word[Rend].alternate; ) {
		    Rend--;
		    skipped_alt = TRUE;
		}
	else if (ref->word[Rend].t1 < hyp->word[Hend].t1)
	    while (Hend >= Hstart &&
		   ref->word[Rend].t1 < hyp->word[Hend].t1 && 
		   overlap(ref->word[Rend].t1,
			   ref->word[Rend].t1+ref->word[Rend].dur,
			   hyp->word[Hend].t1,
			   hyp->word[Hend].t1+hyp->word[Hend].dur) < 0.0)
		Hend --;
    }
    if (dbg) {
	printf("%s:  Synchronized Rend=%d  Hend=%d  ",proc,Rend,Hend);
	printf("Times: %2.2f,%2.2f   %2.2f,%2.2f\n",
	       ref->word[Rend].t1, ref->word[Rend].t1+ref->word[Rend].dur,
	       hyp->word[Hend].t1, hyp->word[Hend].t1+hyp->word[Hend].dur);
    }
    Rend_sync = Rend;
    Hend_sync = Hend;

    /* begin the backward search */
    while (Rend > Rstart && Hend > Hstart){
	/* first compute t1 and t2 for the gap in each segment */
	if (Rend >= Rstart){
	    Rgap_t1 = ref->word[Rend].t1 + ref->word[Rend].dur;
	    Rgap_t2 = (Rend == Rendconv) ? 999999.99 : ref->word[Rend+1].t1;
	} else {
	    Rgap_t1 = 0.0;
	    Rgap_t2 = ref->word[Rstart].t1;
	}
	if (Hend >= Hstart) {
	    Hgap_t1 = hyp->word[Hend].t1 + hyp->word[Hend].dur;
	    Hgap_t2 = (Hend == Hendconv) ? 999999.99 : hyp->word[Hend+1].t1;
	} else {
	    Hgap_t1 = 0.0;
	    Hgap_t2 = hyp->word[Hstart].t1;
	}

	if (dbg) {
	    printf("\n%s:  Gaps: ref %2.2f,%2.2f  hyp %2.2f,%2.2f\n",
		   proc,Rgap_t1, Rgap_t2, Hgap_t1, Hgap_t2);
	    printf("%s:  **** middle word is Rend or Hend\n",proc);
	    dump_word_tokens2(ref,Rend-1,Rend+1);
	    dump_word_tokens2(hyp,Hend-1,Hend+1);
	}

	/* check to see if there is any overlap */
	/* yes if :    S1   |   |         #          |  |    */
	/*             S2          |   |  #  |   |           */
	   
	if (overlap(Rgap_t1,Rgap_t2,Hgap_t1,Hgap_t2) < 0.0){
	    /* no overlap decide which to move */
	
	    if (dbg) printf("%s: Shifting down\n",proc);

	    skipped_alt = TRUE;
	    while (skipped_alt){
		skipped_alt = FALSE;
		if (ref->word[Rend].t1 > hyp->word[Hend].t1){
		    if (Rend > Rstart) {
			Rend --;
			while (Rend >= Rstart && ref->word[Rend].alternate) {
			    Rend--;
			    skipped_alt = TRUE;
			}
		    }
		    if (dbg) printf("%s: Moving Ref\n",proc);
		    while (!skipped_alt &&
			   Hend >= Hstart && 
			   ref->word[Rend].t1 < hyp->word[Hend].t1 &&
			   overlap(ref->word[Rend].t1,
				   ref->word[Rend].t1+ref->word[Rend].dur,
				   hyp->word[Hend].t1,
				 hyp->word[Hend].t1+hyp->word[Hend].dur)< 0.0){
			Hend --;
			if (dbg) printf("%s: Adjusting Hyp\n",proc);
		    }
		} else if (ref->word[Rend].t1 <= hyp->word[Hend].t1) {
		    if (Hend > Hstart) 
			Hend --;
		    if (dbg) printf("%s: Moving Hyp\n",proc);
		    while (!skipped_alt &&
			   Rend >= Rstart &&
			   ref->word[Rend].t1 > hyp->word[Hend].t1 && 
			   overlap(ref->word[Rend].t1,
				   ref->word[Rend].t1+ref->word[Rend].dur,
				   hyp->word[Hend].t1,
				  hyp->word[Hend].t1+ref->word[Hend].dur)<0.0){
			for (Rend--;
			     Rend >= Rstart && ref->word[Rend].alternate; ) {
			    Rend--;
			    skipped_alt = TRUE;
			}
			if (dbg) printf("%s: Adjusting Ref\n",proc);
		    }
		}
	    }
	} else { /* Overlap exists !!!!!   Return the values */
	    *Rret_end = Rend; 
	    *Hret_end = Hend; 
	    if (dbg) printf("%s: R2 Success Chop Ref (%d-%d)  Hyp (%d-%d)\n",
			    proc,Rstart,*Rret_end,Hstart,*Hret_end);
	    return(1);
	}
    }

    if (Rend > Rstart || Hend > Hstart){
	*Rret_end = Rend; 
	*Hret_end = Hend; 
	if (dbg) printf("%s: R3 Success Chop Ref (%d-%d)  Hyp (%d-%d)\n",
			proc,Rstart,*Rret_end,Hstart,*Hret_end);
	return(1);
    }
    /* fprintf(stdout,"%s: Failure to find chop'd segment, returning 0\n",proc); */
    return(0);

}
