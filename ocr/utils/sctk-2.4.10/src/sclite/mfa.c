#include <sctk.h>

void locate_next_file_channel(WTOKE_STR1 **ctms, int nctms, FILE **files, char **hypname, int *eofs, int *conv_end, int case_sense, int feedback);
void print_linear(NODE *node, void *p);
int gap_at_times(WTOKE_STR1 *ctm, int end, double *mgap_beg, double *mgap_end, int *mgap_s);
NETWORK *perform_mfalign(WTOKE_STR1 **ctms, int nctm, int *sil_end, int time_align);
void populate_tag1(NODE *node, void *p);

int mfdb = 0;
int glob_ties = 0;

void mfalign_ctm_files(char **hypname, int nhyps, int time_align, int case_sense, int feedback, void (*callback)(NETWORK *, char *, char *), double silence_dur){
    char *proc = "mfalign_ctm_files";
    WTOKE_STR1 **ctms;
    NETWORK *mfnet;
    FILE **files;
    int *eofs, *conv_end, *sil_end;
    int in;
    int done;
    NETWORK  *cor;
    char *file, *chan;


    {if (getenv("MFALIGN_DBG") != NULL) {
	mfdb=(int)atof(getenv("MFALIGN_DBG")); }}

    if (feedback > 1) printf("Beginning: %s\n",proc);
    /* ALLOCATE MEMORY */
    alloc_singZ(ctms,nhyps,WTOKE_STR1 *,(WTOKE_STR1 *)0);
    alloc_singZ(files,nhyps,FILE *,(FILE *)0);
    alloc_singZ(eofs,nhyps,int,0);
    alloc_singZ(conv_end,nhyps,int,0);
    alloc_singZ(sil_end,nhyps,int,0);
  
    /* OPEN FILES, INIT WTOKE'S, FILE WTOKE'S */
    for (in=0; in < nhyps; in++){
	if ((files[in] = fopen(hypname[in],"r")) == NULL){
	    fprintf(stderr,"Can't open input hypothesis file %s\n",hypname[in]);
	    exit(1);
	}
	ctms[in] = WTOKE_STR1_init(hypname[in]);
    }
    done = 0;
    while (!done){
        locate_next_file_channel(ctms, nhyps, files, hypname, eofs, conv_end,
				 case_sense, feedback);
	
	file = ctms[0]->word[ctms[0]->s].conv;
	chan = ctms[0]->word[ctms[0]->s].turn;

	/* loop though the file/channel section, looking for time breaks */
	while(find_common_silence(ctms, nhyps, conv_end,
				  sil_end, silence_dur) == 1){
	    if (mfdb >= 5){
		printf("--------- Aligning this chunk -----------\n");

		for (in=0; in<nhyps; in++){
		    double beg, end;
		    beg = (ctms[in]->s > conv_end[in]) ? -1.0 : 
			ctms[in]->word[sil_end[in]].t1 +
			ctms[in]->word[sil_end[in]].dur;
		    end = (sil_end[in] < conv_end[in]) ? 
		    ctms[in]->word[sil_end[in]+1].t1 : 9999999.9;
		    printf("   CTM %d: start: %d end_word: %d start: %.2f"
			   " end: %.2f\n",in,ctms[in]->s,sil_end[in],beg,end);
		}
	    }

	    mfnet = perform_mfalign(ctms,nhyps,sil_end,time_align);
	    Network_traverse(mfnet,populate_tag1,(void *)hypname,0,0,0);
	    cor = Network_WORD_to_CORES(mfnet);

	    callback(cor, file, chan);

	    Network_destroy(cor);
	    Network_destroy(mfnet);

	    /* skip over this chunk because we're done with it. */
	    done = 1;
	    for (in=0; in < nhyps; in++){
		ctms[in]->s = sil_end[in] + 
		    ((sil_end[in] <= conv_end[in]) ? 1 : 0);
		if (!eofs[in] || (ctms[in]->s < ctms[in]->n)) done = 0;
	    }
	}
    }
    if (glob_ties > 0)
	fprintf(stderr,"Warning: %d ties were arbitrarily broken\n",glob_ties);
  
    /* FREE THE MEMORY */
    free_singarr(eofs, int);
    free_singarr(conv_end, int);
    free_singarr(sil_end, int);
    for (in=0; in<nhyps; in++) {
	free_mark_file(ctms[in]);
	fclose(files[in]);
    }
    free_singarr(files, FILE *);
    free_singarr(ctms,WTOKE_STR1 *);
    cleanup_NET_ALIGN();
}

NETWORK *perform_mfalign(WTOKE_STR1 **ctms, int nctm, int *end, int time_align){
    char *proc = "perform_mfalign";
    NETWORK **nets, *out_net;
    WORD *null_alt;
    int in;

    alloc_singZ(nets,nctm,NETWORK *,(NETWORK *)0);
    null_alt = new_WORD((TEXT *)"@",-1,0.0,0.0,0.0,(TEXT *)0,(TEXT *)0,0,0,-1.0);
     
    /* create the networks */
    for (in=0; in < nctm; in++){
	if ((nets[in]=
	     Network_create_from_WTOKE(ctms[in],ctms[in]->s,end[in],
				       rsprintf("mfalign net %d",in),
				       print_WORD_wt,
				       equal_WORD2,
				       release_WORD, null_alt_WORD,
				       opt_del_WORD,
				       copy_WORD, make_empty_WORD,
				       use_count_WORD,
				       1))
	    == NULL_NETWORK){ 
	    fprintf(stderr,"%s: Network_create_from_WTOKE failed\n",proc);
	    exit(1);
	}
	/*	Network_traverse(nets[in],0,0,set_tag1,(void *)ctms[in]->id,0); */
    }
    
    /* align the networks */
    Network_dpalign_n_networks(nets,nctm,(!time_align)?wwd_WORD_rover:wwd_time_WORD,
			       &out_net,(void *)null_alt);
    
    /* delete the networks */
    for (in=0; in < nctm; in++)
	Network_destroy(nets[in]);

    free_singarr(nets,NETWORK *);
    release_WORD(null_alt);

    return(out_net);
}

    
void locate_next_file_channel(WTOKE_STR1 **ctms, int nctm, FILE **files, char **hypname, int *eofs, int *conv_end, int case_sense, int feedback){
    int in;

    /* locate the file boundaries */
    for (in=0; in < nctm; in++){ 
	fill_WTOKE_structure(ctms[in], files[in], hypname[in],
			     &(eofs[in]), case_sense);
	locate_boundary(ctms[in], ctms[in]->s, TRUE,TRUE,&(conv_end[in]));
	if (in >= 1 && ctms[in]->s <= ctms[in]->n)
	    if (strcmp(ctms[0]->word[ctms[0]->s].conv,
		       ctms[in]->word[ctms[in]->s].conv) != 0){
		fprintf(scfp,"Error: Conversation mismatches in %s and %s, "
			"%s and %s respectively",hypname[0],hypname[in],
			ctms[0]->word[ctms[0]->s].conv,
			ctms[in]->word[ctms[in]->s].conv);
		exit(1);
	    }
    }
    if (feedback >= 1) {
	printf("Working on conversation: %s channel: %s\n",	
	       ctms[0]->word[ctms[0]->s].conv,
	       ctms[0]->word[ctms[0]->s].turn);
	for (in=0; in < nctm; in++)
	    printf("   File: %s  range %d-%d\n",hypname[in],ctms[in]->s,
		   conv_end[in]);
    }
}

/* This function looks for common periods of silence to all the  WTOKE
 * structures 
 */
int find_common_silence(WTOKE_STR1 **ctms, int nctm, int *ctm_end, int *sil_end, double silence_dur){
    double mgap_beg, mgap_end;
    int nc, failed, more_data;
    int dbg = 0;

    {if (getenv("MFALIGN_COMMON_SIL_DBG") != NULL) {
      dbg=(int)atof(getenv("MFALIGN_COMMON_SIL_DBG")); }}

    /* locate a gap in the master, ctms[0] */
    sil_end[0] = ctms[0]->s;
    while (sil_end[0] <= ctm_end[0]){
	/* set the gap betwen this word and the next */
	mgap_beg = ctms[0]->word[sil_end[0]].t1 +ctms[0]->word[sil_end[0]].dur;
	/* if there is no next word, make it the end of time */
	mgap_end = (sil_end[0] < ctm_end[0]) ? ctms[0]->word[sil_end[0]+ 1].t1:
	    9999999.9 ;

	if (mgap_end > mgap_beg + silence_dur){
	    if (dbg) printf("Found a gap %d (%.2f-%.2f) = %.2f\n",
			    sil_end[0],mgap_beg,mgap_end,mgap_end-mgap_beg);
	    for (nc=1, failed=0; nc<nctm && !failed; nc++){
		if (gap_at_times(ctms[nc],ctm_end[nc],
			 &mgap_beg,&mgap_end,&sil_end[nc]) == 1){
		    if (dbg) printf("  Gap in %d\n",nc);
		} else {
		    failed = 1;
		}
	    }
	    if (failed != 1){
		if (dbg) printf("Segment defined at:\n");
		return(1);
	    }
	}
	sil_end[0] ++;
    }
    /* ctm[0] is at the end of the road, move all the other ctms to the 
     * end as well.  If there is data left in them, return 1 
     */
    if (dbg) printf("*** ctm[0] has ran out of data\n");
    more_data = 0;
    for (nc=0; nc<nctm; nc++){
	sil_end[nc] = ctm_end[nc];
	if (ctms[nc]->s <= ctm_end[nc]) more_data=1;
    }
    return (more_data);
}

void populate_tag1(NODE *node, void *p){
  char **tags = (char **)p;
  int Narc, f, *use, arc;
  ARC_LIST_ATOM *parc;

  if (node != NULL && strcmp(node->name,"STOP") != 0){
    /* count the arcs */
    for (Narc=0, parc=node->out_arcs; parc != (ARC_LIST_ATOM *)0;
	 parc = parc->next)
      Narc++;
    alloc_singZ(use,Narc,int,-1);
    /* map the usage */
    for (arc = 0, parc=node->out_arcs; parc != (ARC_LIST_ATOM *)0;
	 parc = parc->next, arc++){
      /* Eprintf("   %s\n",((WORD*)(parc->arc->data))->value); */
      if (TEXT_strcmp(((WORD*)(parc->arc->data))->value,(TEXT *)"@") != 0){
	/* reset that systems use pointer */
	for (f=0; f<Narc; f++){
	  if (TEXT_strcmp(((WORD*)(parc->arc->data))->tag1,(TEXT*)tags[f]) == 0)
	    use[arc] = f;
	}
      }
    }
    /* dump_singarr(use,Narc," %d",stdout); */
    for (arc = 0, parc=node->out_arcs; parc != (ARC_LIST_ATOM *)0;
	 parc = parc->next, arc++)
      if (use[arc] < 0){
	int hit = -1, y;
	/* printf("   Arc %d not tagged\n",arc); */
	for (f=0; f<Narc && hit < 0; f++){
	  hit = f;
	  /* printf("     is %d used? \n",f); */ 
	  /* does any  of the use elements == f */
	  for (y=0; y<Narc; y++)
	    if (use[y] == f)
	      hit = -1;
	  /* printf("        %d\n",hit); */
	}
	if (hit >= 0){
	  ((WORD*)(parc->arc->data))->tag1 = TEXT_strdup((TEXT*)tags[hit]);
	  use[arc] = hit;
	}
      }
    /* printf("   "); dump_singarr(use,Narc," %d",stdout); */
    
    free_singarr(use,int);
  }
  
}

int gap_at_times(WTOKE_STR1 *ctm, int end, double *mgap_beg, double *mgap_end, int *mgap_s){
    int i;
    double gap_end, gap_beg;
    
    /* if there is no more data in this ctm, that answer is YES */
    if (ctm->s > end) {
	*mgap_s = ctm->s;
	return(1);
    }

    for (i=ctm->s; i<=end; i++){
	gap_beg = ctm->word[i].t1 + ctm->word[i].dur;
	gap_end = (i < end) ? ctm->word[i+1].t1 : 9999999.9;
	if (overlap(*mgap_beg,*mgap_end,gap_beg,gap_end) > 0.0){
	    *mgap_s = i;
	    /* printf("   Gap@ %d times %.2f to %.2f\n",
	       i,gap_beg,gap_end); */
	    *mgap_beg = MAX(*mgap_beg,gap_beg);
	    *mgap_end = MIN(*mgap_end,gap_end);
	    /* printf("   Reset gap times to %.2f-%.2f\n",
	     *mgap_beg,*mgap_end); */
	    return (1);
	}
    }
    return(0);
}
