/* file fillmrks.c                                              */
/* NOTE: THIS IS ESSENTIALLY JON FISCUS' CODE!                  */

#include "sctk.h"

/****************************************************************/
/*  File a wtoke structures, making sure all records have been  */
/*  loaded for each file in the designator in the first field.  */
/*  If not, additional records are read in.                     */
/****************************************************************/
void fill_WTOKE_structure(WTOKE_STR1 *ctm_segs, FILE *fp_ctm, char *ctm_file, int *ctm_eof, int case_sense){
    int ctm_file_end=0, ctm_err=0;
    int just_read;
    
    *ctm_eof = feof(fp_ctm);

    do {
      locate_WTOKE_boundary(ctm_segs, ctm_segs->s, 1, 0, &ctm_file_end);
	just_read = 0;
	if ((ctm_file_end == ctm_segs->n) && !*ctm_eof){
	    fill_mark_struct(fp_ctm,ctm_segs,ctm_file,ctm_eof,&ctm_err, 
			     case_sense);
	    if (ctm_err != 0){
		fprintf(stdout,"; *Err: Error detected in ctm file '%s'\n",
			ctm_file);
		exit(1);
	    }
	    just_read = 1;
	}
    } while (just_read);
}


 /**********************************************************************/
 /*                                                                    */
 /*    void fill_mark_struct(fp,word_tokens,fname,end_of_file,perr);   */
 /*      FILE *fp, WTOKE_STR1 *word_tokens, char *fname,               */
 /*      boolean *end_of_file, int *perr                               */
 /*                                                                    */
 /*    Reads a list of time-marked word tokens from opened file fp,    */
 /*    called fname, and load them into the structure *word_tokens.    */
 /*    The function loads words until the word tokens structure is     */
 /*    filled.  If there is data already in the word tokens structure  */
 /*    it is first copied to the beginning of the structure before     */
 /*    data is read.                                                   */
 /*                                                                    */
 /*   *perr = 0 means a.o.k.                                           */
 /*   error codes:                                                     */
 /*    1: invalid mispronunciation mark at start of file.              */
 /*   11: error in opening file.                                       */
 /*   13: word_tokens array overflow.                                  */
 /*                                                                    */
 /*   **** Added by Jon Fiscus                                         */
 /*    Modification: 9/14/95 JGF Changed so that if the file at the    */
 /*    beginning of the word array is the same as the last element int */
 /*    in the word array, expand the array and continue reading until  */
 /*    eof or the the word is not the same.                            */
 /**********************************************************************/
 void fill_mark_struct(FILE *fp, WTOKE_STR1 *word_tokens, char *fname, boolean *end_of_file, int *perr, int case_sense)
  {
/* data: */
   char *proc = "fill_mark_struct";

   int i, j, n=0;
   TEXT *in_buf, *rp, *gets_rtn;
   static int in_buf_len = 2000;
       /* sxx[LINE_LENGTH], *sx    = &sxx[0];*/
   char *xconv, *xconf, *xsp, *side, *s2, *s3, *xcorr;
   double xt1, xdur;
   char comment_char = ';';
   boolean in_overlap, in_comment, in_unsure, in_mispron, in_crosstalk;
   boolean in_alternate;
/* code: */
db_enter_msg(proc,0); /* debug only */

   alloc_singarr(xconv,LINE_LENGTH,char);
   alloc_singarr(xconf,LINE_LENGTH,char);
   alloc_singarr(xcorr,LINE_LENGTH,char);
   alloc_singarr(xsp,LINE_LENGTH,char);
   alloc_singarr(side,LINE_LENGTH,char);
   alloc_singarr(s2,LINE_LENGTH,char);
   alloc_singarr(s3,LINE_LENGTH,char);
   
   /* allocate memory for the input buffer */
   alloc_singZ(in_buf,in_buf_len,TEXT,(TEXT)0);

   /* if the data doesn't begin at 1, copy it down and go from there */
   if (word_tokens->s > 1){
       /* first free the already alloc'd data */
       for (i=1; i<word_tokens->s; i++){
	   free(word_tokens->word[i].turn);
	   free(word_tokens->word[i].conv);
	   free(word_tokens->word[i].sp);       
       }
       
       /* then copy down the residual data */
       /*  dump_word_tokens2(word_tokens,word_tokens->s,word_tokens->n); */
       for (i=word_tokens->s; i<=word_tokens->n; i++){
	   word_tokens->word[i-word_tokens->s + 1] = word_tokens->word[i];
       }
       word_tokens->n = word_tokens->n - word_tokens->s + 1;
       word_tokens->s=1;
   }

   n = word_tokens->n;
   *perr = 0;

 /* and loop on contents */
   in_overlap   = F;
   in_comment   = F;
   in_unsure    = F;
   in_mispron   = F;
   in_crosstalk = F;
   in_alternate = F;
   gets_rtn = NULL;
   while (n < word_tokens->max-1 && 
	  (gets_rtn = TEXT_ensure_fgets(&in_buf, &in_buf_len,fp)) != NULL){
       if (*in_buf != (unsigned char)comment_char){
	   if (!case_sense)
	       TEXT_str_case_change_with_mem_expand(&in_buf, &in_buf_len, 1);
	   n += 1;
	   *s2 = *s3 = *xconv = *xsp = *xconf = *xcorr = '\0';

	   /* printf("WTOKE: %s  %d  %d\n",
	      in_buf,TEXT_strlen(in_buf),strlen(in_buf)); */
	   if (in_buf[(j = TEXT_strlen(in_buf)) - 1] == '\n')
	       in_buf[j-1] = '\0';

	   if (*in_buf != '\0'){
	       /* parse stm string */
	     rp = TEXT_strtok((TEXT *)in_buf + TEXT_strspn(in_buf,
							   (TEXT*)" \t"),
			      (TEXT *)" \t\n");
	       TEXT_strBcpy((TEXT *)xconv,rp,LINE_LENGTH);
	       rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");
	       TEXT_strBcpy((TEXT *)side,rp,LINE_LENGTH);
	       rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");
	       TEXT_strBcpy((TEXT *)s2,rp,LINE_LENGTH);
	       rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n"); 
	       TEXT_strBcpy((TEXT *)s3,rp,LINE_LENGTH);
	       rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n");
	       TEXT_strBcpy((TEXT *)xsp,rp,LINE_LENGTH);
	       if ((rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n")) != NULL){
		 TEXT_strBcpy((TEXT *)xconf,rp,LINE_LENGTH);
		 if ((rp = TEXT_strtok((TEXT *)0,(TEXT *)" \t\n")) != NULL){
		   TEXT_strBcpy((TEXT *)xcorr,rp,LINE_LENGTH);
		 }
	       }
	   }	    
	   /* printf("       conv %s\n",xconv);
	      printf("       side %s\n",side);
	      printf("       s2   %s\n",s2);
	      printf("       s3   %s\n",s3);
	      printf("       text %s\n",xsp);
	      printf("       conf %s\n",xconf);  
	      printf("       corr %s\n",xcorr);  */
	   if (strcmp(xconv,"") == 0) {
	       fprintf(stdout,"; *ERR: Conversation is empty '%s'.\n",
		       in_buf);
	       *perr = 14; goto RETURN;
	   }
	   if (strcmp(side,"") == 0) {
	       fprintf(stdout,"; *ERR: Conversation side is empty '%s'.\n",
		       in_buf);
	       *perr = 15; goto RETURN;
	   }
	   if (strcmp(s2,"") == 0) {
	       fprintf(stdout,"; *ERR: Start time is empty '%s'.\n",
		       in_buf);
	       *perr = 16; goto RETURN;
	   }
	   if (strcmp(s3,"") == 0) {
	       fprintf(stdout,"; *ERR: Duration time is empty '%s'.\n",
		       in_buf);
	       *perr = 17; goto RETURN;
	   }
	   if (strcmp(xsp,"") == 0) {
	       fprintf(stdout,"; *ERR: Word string is empty '%s'.\n",
		       in_buf);
	       *perr = 18; goto RETURN;
	   }
	   
	   if ((s2[0] == '&') && (s2[1] == '&')) /* bad time marks */
	       {xt1 = atof(s2+2);
		word_tokens->word[n].bad_marking = T;
	    }
	   else
	       {xt1 = atof(s2);
		word_tokens->word[n].bad_marking = F;
	    }
	   
	   xdur = atof(s3);
	   word_tokens->word[n].turn  = (char *)TEXT_strdup((TEXT *)side);
	   if (strcmp(xconf,"") == 0 || strcasecmp(xconf,"NA") == 0)
	       word_tokens->word[n].confidence    = 0.0;
	   else {
	       word_tokens->word[n].confidence    = atof(xconf);
	       word_tokens->has_conf = 1;
	   }
	   if (strcmp(xcorr,"") == 0)
	     word_tokens->word[n].correct   = -1;
	   else {
	     word_tokens->word[n].correct   = atof(xcorr);
	   }
	   word_tokens->word[n].t1   = xt1;
	   word_tokens->word[n].dur  = xdur;
	   word_tokens->word[n].sp   = TEXT_strdup((TEXT *)xsp);
	   word_tokens->word[n].conv = (char *)TEXT_strdup((TEXT *)xconv);
	   
	   word_tokens->word[n].overlapped    = F;
	   word_tokens->word[n].comment       = F;
	   word_tokens->word[n].unsure        = F;
	   word_tokens->word[n].mispronounced = F;
	   word_tokens->word[n].crosstalk     = F;
	   word_tokens->word[n].ignore        = F;
	   in_alternate = F;
	   if (n > 1){
	       if (word_tokens->word[n-1].alternate == T &&
		   TEXT_strcasecmp(word_tokens->word[n-1].sp,
				   (TEXT *)"<ALT_END>")!=0)
		   in_alternate = T;
	   }
	   if (TEXT_strCcasecmp(word_tokens->word[n].sp,(TEXT*)"<ALT",4) == 0)
	       in_alternate = T;
	   word_tokens->word[n].alternate = in_alternate;
       }
       if (n >= word_tokens->max-1 && 
	   strcmp(word_tokens->word[n].conv,word_tokens->word[1].conv) == 0){
	   /* expanding the words array */
	   expand_singarr(word_tokens->word,n+1,word_tokens->max,1.3,WTOKE1);
       }
   }
   if (gets_rtn == NULL)
       *end_of_file = T;
   
   /* mark overlap indicated by following turn being starred */
   i = 2;
   while ((i <= n)&&(*(word_tokens->word[i].turn) != '*')) i++;
   if (i <= n)
       {xconv = strcpy(xconv,word_tokens->word[i-1].turn);
	for (j=i-1; ((j > 0)&&streq(word_tokens->word[j].turn,xconv)); j--);
        {word_tokens->word[j].overlapped = T;
     }  }
 RETURN:
   free_singarr(in_buf,TEXT);
   free_singarr(xconv,char);
   free_singarr(xconf,char);
   free_singarr(xcorr,char);
   free_singarr(xsp,char);
   free_singarr(side,char);
   free_singarr(s2,char);
   free_singarr(s3,char);

   word_tokens-> s = 1;
   word_tokens-> n = n;
   if (word_tokens->id == (char *)0)
       word_tokens->id = (char *)TEXT_strdup((TEXT *)"");
   db_leave_msg(proc,0); /* debug only */
   return;
} /* end of function "fill_mark_struct" */

void locate_WTOKE_boundary(WTOKE_STR1 *seg, int start, int by_conv, int by_turn, int *end){
    int w;
    int limit=0;
    int tchg, cchg;
 
    if (start == seg->n){
        *end = start;
        return;
    }
    for (w=start; w<=seg->n && limit == 0; w++){
        tchg = (!by_conv) ? 1 : 
            (strcmp(seg->word[start].conv,seg->word[w].conv) == 0);
        cchg = (!by_turn) ? 1 : 
            (strcmp(seg->word[start].turn,seg->word[w].turn) == 0);
        if (!(tchg && cchg))
            limit = w-1;
    }
    if (limit == 0)
        limit = seg->n;
    *end = limit;
}
 
 
void reset_WTOKE_flag(WTOKE_STR1 *seg,char *flag_name)
{
    int w;
 
    if (strcmp(flag_name,"overlapped") == 0){
        for (w=1; w<seg->n; w++)
            seg->word[w].overlapped = F;
    } else if (strcmp(flag_name,"comment") == 0){
        for (w=1; w<seg->n; w++)
            seg->word[w].comment = F;
    }
}


 /**********************************************************************/
 /*                                                                    */
 /*    void free_mark_file(word_tokens);                               */
 /*    WTOKE_STR1 *word_tokens;                                         */
 /*                                                                    */
 /*    Frees the dynamic memory allocated to hold *word_tokens.        */
 /*    Modification: 9/14/95 JGF Changed to free the the word array    */
 /*                                                                    */
 /**********************************************************************/
 void free_mark_file(WTOKE_STR1 *word_tokens)
  {
/* data: */
   char *proc = "free_mark_file";
   int i;
/* code: */
db_enter_msg(proc,0); /* debug only */
   for (i=1; i <= word_tokens->n; i++)

     {/* free((void *)word_tokens->word[i].turn); */ /* K&R */
      free(word_tokens->word[i].turn);
      /* free((void *)word_tokens->word[i].conv); */ /* K&R */
      free(word_tokens->word[i].conv);
      /* free((void *)word_tokens->word[i].sp);   */ /* K&R */
      free(word_tokens->word[i].sp);
      word_tokens->word[i].overlapped = F;
      word_tokens->word[i].mispronounced = F;
      word_tokens->word[i].unsure = F;
      word_tokens->word[i].comment = F;
      word_tokens->word[i].bad_marking = F;
      word_tokens->word[i].crosstalk = F;
      word_tokens->word[i].alternate = F;
      word_tokens->word[i].ignore = F;
     }
   /* free((void *)word_tokens->id);   */ /* K&R */
   free(word_tokens->id);
   free(word_tokens->word);
   free(word_tokens);
 db_leave_msg(proc,0); /* debug only */
   return;
  } /* end of function "free_mark_file" */


/*
 * This function looks forward in a WTOKE struct for the end of either
 * the converstion, the end of the turn, which is actually the channel
 * to every one but the program, or both
 */
void locate_boundary(WTOKE_STR1 *seg, int start, int by_conv, int by_turn, int *end){ 
   int w;
    int limit=0;
    int tchg, cchg;

    if (start == seg->n){
	*end = start;
	return;
    }
    for (w=start; w<=seg->n && limit == 0; w++){
	tchg = (!by_conv) ? TRUE : 
	    (strcmp(seg->word[start].conv,seg->word[w].conv) == 0);
 	cchg = (!by_turn) ? TRUE : 
	    (strcmp(seg->word[start].turn,seg->word[w].turn) == 0);
	if (!(tchg && cchg))
	    limit = w-1;
    }
    if (limit == 0)
	limit = seg->n;
    *end = limit;
}
