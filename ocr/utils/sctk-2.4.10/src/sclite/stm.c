#include "sctk.h"

STM *alloc_STM(int n){
    STM *ts;
    alloc_singarr(ts,1,STM);
    ts->max = n;
    ts->num = 0;
    ts->s = 0;
    alloc_singarr(ts->seg,ts->max,STM_SEG);
    return(ts);
}

void expand_STM(STM *stm){
    expand_singarr(stm->seg,stm->num,stm->max,2,STM_SEG);
}    

void free_STM_SEG(STM_SEG *seg){
    if (seg == (STM_SEG *)0)
	return;
    TEXT_free(seg->file);
    TEXT_free(seg->spkr);
    TEXT_free(seg->chan);
    TEXT_free(seg->text);
    if (seg->labels != (TEXT *)0) TEXT_free(seg->labels);
}

void free_STM(STM *stm){
    int i;
    for (i=0; i<stm->num; i++)
	free_STM_SEG(&(stm->seg[i]));
    free_singarr(stm->seg,STM_SEG);
    free_singarr(stm,STM);
}

void read_stm_line(TEXT **buf, int *len, FILE *fp){
    **buf  = *(*buf+1) = COMMENT_CHAR;
    while (!feof(fp) && (TEXT_is_comment(*buf) ||
			 TEXT_is_comment_info(*buf) ||
			 TEXT_is_empty(*buf))){
	if (TEXT_ensure_fgets(buf, len, fp) == NULL)
	    **buf = '\0';
    }
}


void parse_stm_line(STM_SEG *seg, TEXT **buf_ptr, int *buf_len, int case_sense, int dbg){
    TEXT *rp, *buf;
    int i, len;

    if (**buf_ptr == '\0')
	return;

    if (!case_sense)
        TEXT_str_case_change_with_mem_expand(buf_ptr, buf_len, 1);
	
    buf = *buf_ptr;

    len = TEXT_strlen(buf);

    /* parse stm string */
    rp = TEXT_strtok(buf,(TEXT *)" \t\n");     	seg->file = TEXT_strdup(rp);
    rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");  seg->chan = TEXT_strdup(rp);
    rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");  seg->spkr = TEXT_strdup(rp);
    rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");  seg->t1   = TEXT_atof(rp);
    rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");  seg->t2   = TEXT_atof(rp);
    seg->flag1 = 0;
    seg->labels = (TEXT *)0;
    if (rp != (TEXT *)0){
	/* snag the next token, if it's a set identifier, load it, */
	/* if not, overwrite the ending null with a space provided */
	/* there is data beyond the end of this token */
	if (((rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n")) != (TEXT *)0) &&
	    (*rp == '<')) {
	    seg->labels = TEXT_strdup(rp);
	    rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");
	} else 
	    seg->labels = (TEXT *)0;
	if (rp != (TEXT *)0){
	    /* reclaim the first token from the text */
	    if (rp + TEXT_strlen(rp) < buf + len)
		*(rp + TEXT_strlen(rp)) = ' ';
	    seg->text = TEXT_strdup(rp);
	} else
	    seg->text = TEXT_strdup((TEXT *)"");
    } else 
	seg->text = TEXT_strdup((TEXT *)"");

    /* was    seg->text = TEXT_strdup(rp + TEXT_strlen(rp) + 1); */

    if (dbg){
	printf("Parsed: file: %s\n",seg->file);
	printf("        chan: %s\n",seg->chan);
	printf("        spkr: %s\n",seg->spkr);
	printf("        t1  : %f\n",seg->t1);
	printf("        t2  : %f\n",seg->t2);
	printf("        text: %s\n",seg->text);
    }
    i = TEXT_strlen(seg->text) - 1;
    if (i > 0 && seg->text[i] == '\n') 
	seg->text[i] = (TEXT)0; 
}

void fill_STM(FILE *fp, STM *stm, char *fname, boolean *end_of_file, int case_sense, int *perr){
    static int len=100;
    int i;
    TEXT *buf;
    int dbg=0;

    alloc_singZ(buf,len,TEXT,(TEXT)0);
    *perr = 0;

    /* if the data doesn't begin at 1, copy it down and go from there */
    if (stm->s > 1){
	/* first free the already alloc'd data */
	for (i=0; i<stm->s; i++){
	    TEXT_free(stm->seg[i].file);
	    TEXT_free(stm->seg[i].spkr);
	    TEXT_free(stm->seg[i].chan);
	    TEXT_free(stm->seg[i].text);
	    if (stm->seg[i].labels != (TEXT *)0)
		TEXT_free(stm->seg[i].labels);
	}
       
	/* then copy down the residual data */
	for (i=stm->s; i<=stm->num; i++)
	    stm->seg[i - stm->s] = stm->seg[i];

	stm->num = stm->num - stm->s;
	stm->s=0;
    }

    /* now read in the data */
    while (!feof(fp) && stm->num+1 < stm->max){
	read_stm_line(&buf,&len,fp);
	if (dbg) printf("STM Read %s\n",buf);

	if (*buf != NULL_TEXT){
	    parse_stm_line(&(stm->seg[stm->num]), &buf, &len, case_sense, dbg);
	    stm->num++;	
	}
    }
    *end_of_file = feof(fp);
    free_singarr(buf,TEXT);
}


void locate_STM_boundary(STM *stm, int start, int by_file, int by_chan, int *end){
    int w;
    int limit=-1;
    int tchg, cchg;
 
    if (start == stm->num){
        *end = start;
        return;
    }
    for (w=start; w<stm->num && limit == -1; w++){
        tchg = (!by_file) ? 1 : 
            (TEXT_strcmp(stm->seg[start].file,stm->seg[w].file) == 0);
        cchg = (!by_chan) ? 1 : 
            (TEXT_strcmp(stm->seg[start].chan,stm->seg[w].chan) == 0);
        if (!(tchg && cchg))
            limit = w-1;
    }
    if (limit == -1)
        limit = stm->num;
    *end = limit;
}

void dump_STM_words(STM *stm,int s, int e, char *file){
    int i;
    FILE *fp = fopen(file,"w");
    TEXT *ctext, *tbuf;
    int tbuf_len=100;

    if (fp == NULL){
	fprintf(stderr,"Error: Can't open STM words file '%s'\n",file);
	exit(1);
    }
    alloc_singZ(tbuf,tbuf_len,TEXT,'\0');

    for (i=s; i<=e; i++){
	/* Write each individual word to a file */
	if (TEXT_strlen(stm->seg[i].text) > tbuf_len+1){
	    free_singarr(tbuf,TEXT);
	    tbuf = TEXT_strdup(stm->seg[i].text);
	    tbuf_len = TEXT_strlen(tbuf);
	} else
	    TEXT_strcpy(tbuf,stm->seg[i].text);

	ctext = tokenize_TEXT_first_alt(tbuf,(TEXT *)" \t\n");
	while (ctext != NULL) {
	    fprintf(fp,"%s\n",ctext);
	    ctext = tokenize_TEXT_first_alt(NULL,(TEXT *)" \t\n");
	}
    }
    fclose(fp);
    free_singarr(tbuf,TEXT);
}

void dump_STM(STM *stm, int s, int e){
    char *proc="dump_STM";
    int i;

    printf("%s: Range [%d,%d]  of  [%d,%d]\n",proc,s,e,stm->s,stm->num);
    for (i=s; i<e; i++)
	printf("%d: %s chan: %s  Spkr: %s  T1: %f  T2:%f  Text: '%s'\n",i,
	       stm->seg[i].file,stm->seg[i].chan,
	       stm->seg[i].spkr,stm->seg[i].t1,
	       stm->seg[i].t2,stm->seg[i].text);
    printf("\n");
}

void convert_stm_to_word_list(char *file, char *words, int case_sense, int *num_ref){
    STM_SEG seg;
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
    (*num_ref) = 0;
    alloc_singZ(buf,buf_len,TEXT,'\0');

    /* for each stm, load the aligned text and create a path as before */
    while (!feof(fp)){
	/* read the next ref line */
	read_stm_line(&buf,&buf_len,fp);
	if (feof(fp)) break;

	/* parse the reference transcript */
	parse_stm_line(&seg,&buf,&buf_len,case_sense,0);
	
	/* for each reference word, located it's matching diff output */
	ctext = tokenize_TEXT_first_alt(seg.text,(TEXT *)" \t\n");

	while (ctext != NULL){
	    fprintf(fp_out,"%s\n",ctext);
	    ctext = tokenize_TEXT_first_alt(NULL,(TEXT *)" \t\n");
	}
	(*num_ref) ++;	
	free_STM_SEG(&seg);
    }
    fclose(fp);
    fclose(fp_out);
    free_singarr(buf,TEXT);
}


/****************************************************************/
/*  File an stm structure, making sure all records have been    */
/*  loaded for each file in the designator.  If not, additional */
/*  records are read in.                                        */
/****************************************************************/
void fill_STM_structure(STM *stm, FILE *fp_stm, char *stm_file, int *stm_file_end, int case_sense){ 
    static int stm_eof=0, stm_err=0;
    int just_read;
    
    if (stm == (STM *)0){
	/* Reset the static variables */
        stm_eof = stm_err = 0;
	return;
    }
    
    just_read = 0;
    do {
	locate_STM_boundary(stm, stm->s, 1, 0, stm_file_end);
	if ((*stm_file_end == stm->num) && !stm_eof){
	    if (just_read == 1){
		just_read = 0;
		expand_STM(stm);
	    }
	    fill_STM(fp_stm, stm, stm_file, &stm_eof, case_sense,&stm_err);
	    if (stm_err != 0){
		fprintf(stdout,"; *Err: Error detected in STM file '%s'\n",
			stm_file);
		exit(1);
	    }
	    just_read = 1;
	} else 
	    just_read = 0;
    } while (just_read);
}
