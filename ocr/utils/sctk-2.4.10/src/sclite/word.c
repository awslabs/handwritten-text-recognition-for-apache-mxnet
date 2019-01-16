#include "sctk.h"

int null_alt_WORD(void *p){
    return (TEXT_strcmp(((WORD *)p)->value, (TEXT *)"@") == 0);
}

int opt_del_WORD(void *p){
    return (((WORD *)p)->opt_del);
}

/* Description of the cost structure:
   0      ->  Correct
   0.001  ->  Insertion, deletion of '@' (a NULL)
   1      ->  Substitution of '@' for '@'
   3      ->  Deletion or insertion
   4      ->  Substitution 
   */
float wwd_WORD(void *p1, void *p2, int (*cmp)(void *p1, void *p2)){
    WORD *w1 = (WORD *)p1, *w2 = (WORD *)p2;

    /* 4-3-96 JGF Changed cost of insert/deletion of a @'s to 0.001 */

    if (w1 == NULL_WORD && w2 == NULL_WORD){
	fprintf(stderr,"Error: wwd_WORD computed for two NULL_WORDS\n");
	exit(1);
    }
    if (w2 == NULL_WORD && TEXT_strcmp(w1->value, (TEXT *)"@") == 0)
	return 0.001;
    if (w1 == NULL_WORD && TEXT_strcmp(w2->value, (TEXT *)"@") == 0)
	return 0.001;

    /* look for optionally deletable words, set the cost to CORR < OD < INS/DEL*/
    if ((w1 == NULL_WORD && w2->opt_del) || 
	(w2 == NULL_WORD && w1->opt_del))
      return(2.0);

    /* Standard insertion deletion checks */
    if (w1 == NULL_WORD || w2 == NULL_WORD)
	return 3.0;

    /* Added by JGF, 4-3-96 substitutions of @'s should be slightly */
    /* higher than 0 */
    if ((TEXT_strcmp(w1->value, (TEXT *)"@") == 0) && 
	(TEXT_strcmp(w2->value, (TEXT *)"@") == 0)){
	return(1.0);
    }
    if (cmp(p1,p2) == 0)
	return 0.0;
    return 4.0;  /* a substitution */
}

float wwd_WORD_rover(void *p1, void *p2, int (*cmp)(void *p1, void *p2)){
    WORD *w1 = (WORD *)p1, *w2 = (WORD *)p2;

    /* 4-3-96 JGF Changed cost of insert/deletion of a @'s to 0.001 */
    if (w1 == NULL_WORD && w2 == NULL_WORD){
	fprintf(stderr,"Error: wwd_WORD computed for two NULL_WORDS\n");
	exit(1);
    }
    if (w2 == NULL_WORD && TEXT_strcmp(w1->value, (TEXT *)"@") == 0)
	return 0.001;
    if (w1 == NULL_WORD && TEXT_strcmp(w2->value, (TEXT *)"@") == 0)
	return 0.001;
    
    if (w1 != NULL_WORD && w2 != NULL_WORD){
      if (((TEXT_strstr(w1->value,(TEXT*)"%sync%time") != (TEXT *)0) || (TEXT_strstr(w2->value,(TEXT*)"%sync%time") != (TEXT *)0)) &&
	  (TEXT_strcmp(w1->value,w2->value) == 0))
	return -3.0;
      /*      if (((TEXT_strcmp(w2->value,(TEXT*)"@") == 0) && (TEXT_strstr(w1->value,(TEXT*)"%sync%time") != (TEXT *)0)) ||
	      ((TEXT_strcmp(w1->value,(TEXT*)"@") == 0) && (TEXT_strstr(w2->value,(TEXT*)"%sync%time") != (TEXT *)0))) */
      if ((TEXT_strstr(w1->value,(TEXT*)"%sync%time") != (TEXT *)0) ||
	  (TEXT_strstr(w2->value,(TEXT*)"%sync%time") != (TEXT *)0))	
	return(20.0);
    }
    if ((w2 == NULL_WORD && (TEXT_strstr(w1->value,(TEXT*)"%sync%time") != (TEXT *)0)) ||
	(w1 == NULL_WORD && (TEXT_strstr(w2->value,(TEXT*)"%sync%time") != (TEXT *)0)))
      return 0.00001;


    /* look for optionally deletable words, set the cost to CORR < OD < INS/DEL*/
    if ((w1 == NULL_WORD && w2->opt_del) || 
	(w2 == NULL_WORD && w1->opt_del))
      return(2.0);

    /* Standard insertion deletion checks */
    if (w1 == NULL_WORD || w2 == NULL_WORD)
	return 3.0;

    /* Added by JGF, 4-3-96 substitutions of @'s should be slightly */
    /* higher than 0 */
    if ((TEXT_strcmp(w1->value, (TEXT *)"@") == 0) && 
	(TEXT_strcmp(w2->value, (TEXT *)"@") == 0)){
	return(1.0);
    }
    if (cmp(p1,p2) == 0)
      return 0.0;

    if ((TEXT_strcmp(w1->value, (TEXT *)"@") == 0) ||
	(TEXT_strcmp(w2->value, (TEXT *)"@") == 0))
      return 1.0;

    /* special case for sync tags */
    if ((TEXT_strstr(w1->value,(TEXT*)"%sync%time") != (TEXT *)0) ||
	(TEXT_strstr(w2->value,(TEXT*)"%sync%time") != (TEXT *)0)){
      return 15.0;
    }

    return 4.0;  /* a substitution */
}

float wwd_time_WORD(void *p1, void *p2, int (*cmp)(void *p1, void *p2)){
    WORD *w1 = (WORD *)p1, *w2 = (WORD *)p2;

    if (w1 == NULL_WORD && w2 == NULL_WORD){
	fprintf(stderr,"Error: wwd_WORD computed for two NULL_WORDS\n");
	exit(1);
    }

    if (w2 == NULL_WORD && TEXT_strcmp(w1->value, (TEXT *)"@") == 0)
	return 0.001;
    if (w1 == NULL_WORD && TEXT_strcmp(w2->value, (TEXT *)"@") == 0)
	return 0.001;
    if (w1 == NULL_WORD)	return w2->T_dur;
    if (w2 == NULL_WORD) 	return w1->T_dur;
    /* no difference between correct and substituted */
    return (fabs(w1->T1 - w2->T1) + fabs(w1->T2 - w2->T2) +
	    ((cmp(p1,p2) == 0) ? 0.000 : 0.001));
}

float wwd_weight_WORD(void *p1, void *p2, int (*cmp)(void *p1, void *p2)){
    WORD *w1 = (WORD *)p1, *w2 = (WORD *)p2;

    if (w1 == NULL_WORD && w2 == NULL_WORD){
	fprintf(stderr,"Error: wwd_WORD computed for two NULL_WORDS\n");
	exit(1);
    }

    if (w2 == NULL_WORD && TEXT_strcmp(w1->value, (TEXT *)"@") == 0)
	return 0.001;
    if (w1 == NULL_WORD && TEXT_strcmp(w2->value, (TEXT *)"@") == 0)
	return 0.001;
    if (w1 == NULL_WORD)	return w2->weight;
    if (w2 == NULL_WORD) 	return w1->weight;

    /* Add the extra cost for a substitution */
    /*     return (fabs(w1->weight - w2->weight) 
	    + ((cmp(p1,p2) == 0) ? 0.000 : 0.001)); */
    if (cmp(p1,p2) == 0) return(0.0);
    return (w1->weight + w2->weight);
}

void *append_WORD(void *p1, void *p2){
    WORD *tw, *w1 = (WORD *)p1, *w2 = (WORD *)p2;

    tw = get_WORD();
    
    tw->use = 1;
    tw->T1 = w1->T1;
    tw->T2 = w2->T2;
    tw->T_dur = tw->T2 - tw->T1; 
    tw->weight = w1->weight + w2->weight; 
    tw->conf = (w1->conf + w2->conf) / 2.0;
    if (w1->opt_del && w2->opt_del) {
      tw->intern_value =
	TEXT_strdup((TEXT*)rsprintf("%s%s",
			      (! w1->opt_del) ? w1->value : w1->intern_value,
			      (! w2->opt_del) ? w2->value : w2->intern_value));
      tw->value = 
      	TEXT_strdup((TEXT*)rsprintf(WORD_OPT_DEL_PRE_STR "%s"
				    WORD_OPT_DEL_POST_STR, tw->intern_value));

      tw->opt_del = TRUE;
    } else {
      tw->value = TEXT_add((! w1->opt_del) ? w1->value : w1->intern_value,
			   (! w2->opt_del) ? w2->value : w2->intern_value);
    }

    /* tw->opt_del = (w1->opt_del && w2->opt_del); */
    tw->frag_corr = (w1->frag_corr && w2->frag_corr);

    return(tw);
}

void *append_WORD_no_NULL(void *p1, void *p2){
    WORD *tw, *w1 = (WORD *)p1, *w2 = (WORD *)p2;

    tw = get_WORD(); 
    tw->use = 1;
    tw->T1 = w1->T1;
    tw->T2 = w2->T2;
    tw->T_dur = tw->T2 - tw->T1; 
    tw->weight = w1->weight + w2->weight; 
    tw->conf = (w1->conf + w2->conf) / 2.0;
    tw->frag_corr = (w1->frag_corr && w2->frag_corr);

    if (w1->opt_del && w2->opt_del){
      tw->intern_value =
	TEXT_strdup((TEXT*)rsprintf("%s%s",
			      (! w1->opt_del) ? w1->value : w1->intern_value,
			      (! w2->opt_del) ? w2->value : w2->intern_value));
      tw->value = 
      	TEXT_strdup((TEXT*)rsprintf(WORD_OPT_DEL_PRE_STR "%s"
				    WORD_OPT_DEL_POST_STR, tw->intern_value));

      tw->opt_del = TRUE;
    } else {
      if (TEXT_strcmp(w1->value, (TEXT *)"@") == 0)
	tw->value = TEXT_strdup((! w2->opt_del) ? w2->value : w2->intern_value);
      else if (TEXT_strcmp(w2->value,(TEXT *)"@") == 0)
	tw->value = TEXT_strdup((! w1->opt_del) ? w1->value : w1->intern_value);
      else
	tw->value = TEXT_add((! w1->opt_del) ? w1->value : w1->intern_value,
			     (! w2->opt_del) ? w2->value : w2->intern_value);
    }

    return(tw);
}

WORD *get_WORD(void){
    WORD *tw;

    alloc_singarr(tw,1,WORD);
    
    tw->use = 0;
    tw->conf = -1.0;
    tw->weight = -1.0;
    tw->opt_del = FALSE;
    tw->frag_corr = FALSE;
    tw->value = NULL_TEXT;
    tw->intern_value = NULL_TEXT;
    tw->tag1 = NULL_TEXT;
    tw->tag2 = NULL_TEXT;
    tw->value_id = -1;
    tw->T1 = tw->T2 = tw-> T_dur = 0.0;
    return(tw);
}

TEXT *nextColon(TEXT *t){
  int extentPos = 0;
  int endFound = 0;
  TEXT *nextColon;
//  printf("  NextCol: %s\n",t);
  do {
    nextColon = TEXT_strstr(t+extentPos, (TEXT *)WORD_SGML_SUB_WORD_SEP_STR);
    if (nextColon == (TEXT *)NULL){
       endFound = 1;
       extentPos = TEXT_strlen(t);
    } else if (*(nextColon - 1) == WORD_SGML_ESCAPE) {
       extentPos = nextColon - t + 1;
    } else {
       endFound = 1;
       extentPos = nextColon - t;
    }
//        printf ("  Reloop %s\n",t+extentPos);
  } while (! endFound);
//  printf("    final: %s\n",t+extentPos);
  return (t + extentPos);
}

/* This procudure parses the text for tag1 and tag2 */
WORD *new_WORD_parseText(TEXT *t, int id, double t1, double t2, double conf, int fcorr, int odel, double weight){
  TEXT *endOfElement;
  TEXT *text = (TEXT *)NULL, *tag1 = (TEXT *)NULL, *tag2 = (TEXT *)NULL;
  TEXT *textPtr = t;
  WORD *word;

//  printf("Element is %s\n",t);
  // Is this an alternation
  if (*textPtr == '{'){
    word = new_WORD(t, id, t1, t2, conf, tag1, tag2, fcorr, odel, weight);
    return word;
  } else {
    endOfElement = nextColon(textPtr);
    text = TEXT_strBdup_noEscape(textPtr, endOfElement - textPtr);
    if (*endOfElement != NULL_TEXT){
      textPtr = endOfElement + 1;
      endOfElement = nextColon(textPtr);
      tag1 = TEXT_strBdup_noEscape(textPtr, endOfElement - textPtr);
      if (*endOfElement != NULL_TEXT){
        textPtr = endOfElement + 1;
        endOfElement = nextColon(textPtr);
        tag2 = TEXT_strBdup_noEscape(textPtr, endOfElement - textPtr);
      }
    }

//    printf("   TExt is %s\n",text);
//    printf("   Tag1 is %s\n",(tag1 != (TEXT *)0) ? tag1 : (TEXT *)"null");
//    printf("   Tag2 is %s\n",(tag2 != (TEXT *)0) ? tag2 : (TEXT *)"null");
    word = new_WORD(text, id, t1, t2, conf, tag1, tag2, fcorr, odel, weight);
    if (text != (TEXT *)0) free_1dimarr(text, TEXT);
    if (tag1 != (TEXT *)0) free_1dimarr(tag1, TEXT);
    if (tag2 != (TEXT *)0) free_1dimarr(tag2, TEXT);
    return word;
 }
}

WORD *new_WORD(TEXT *t, int id, double t1, double t2, double conf, TEXT *tag1, TEXT *tag2, int fcorr, int odel, double weight){
    WORD *tw = get_WORD();
    int len;
    
    tw->use = 1;
    if (t != (TEXT *)0){
      len = TEXT_strlen(t);
      if (len > 1 && *(t + len - 1) == '*')
	tw->value = TEXT_strBdup(t,len-1);
      else
	tw->value = TEXT_strdup(t);
    }

    tw->tag1  = (tag1 == (TEXT *)0) ? tag1 : TEXT_strdup(tag1);
    tw->tag2  = (tag2 == (TEXT *)0) ? tag2 : TEXT_strdup(tag2);
    tw->value_id = id;
    tw->T1 = t1;
    tw->T2 = t2;
    tw->T_dur = t2 - t1;
    tw->weight = weight;
    tw->conf = conf;
    tw->frag_corr = fcorr;
    tw->opt_del = odel;
    return(tw);
}

void release_WORD(void *p){
    WORD *tw = (WORD *)p;    
    if (tw == NULL_WORD)
	return;
    tw->use --;
    if (tw->use != 0)
	return;
    if (tw->value != NULL_TEXT)	TEXT_free(tw->value);
    if (tw->intern_value != NULL_TEXT) TEXT_free(tw->intern_value);
    if (tw->tag1 != NULL_TEXT)	TEXT_free(tw->tag1);
    if (tw->tag2 != NULL_TEXT)	TEXT_free(tw->tag2);
    free_singarr(tw,WORD);

}

#ifdef old
int equal_WORD(void *p1, void *p2){
    if (p1 == (void *)0) return(-1);
    if (p2 == (void *)0) return(1);
    return(TEXT_strcmp(((WORD *)p1)->value,((WORD *)p2)->value));
}

int equal_WORD_wfrag(void *p1, void *p2){
    int l1, l2; 
    WORD *w1 = ((WORD *)p1), *w2 = ((WORD *)p2);
    
    if (p1 == (void *)0) return(-1);
    if (p2 == (void *)0) return(1);
    l1 = TEXT_strlen(w1->value);
    l2 = TEXT_strlen(w2->value);
    if (*(w1->value) == '-' && l1 > 1) 
	if (l2 < l1-1)
	    return(1);
	else
	    return(TEXT_strBcmp(w1->value + 1,w2->value + (l2 - (l1-1)),l1-1));
    else if (*(w1->value + l1 - 1) == '-'  && l1 > 1)
	return(TEXT_strBcmp(w1->value,w2->value,l1-1));

    if (*(w2->value) == '-' && l2 > 1) 
	if (l1 < l2-1)
	    return(-1);
	else
	    return(TEXT_strBcmp(w1->value + (l1 - (l2-1)),w2->value + 1,l2-1));
    else if (*(w2->value + l2 - 1) == '-' && l2 > 1)
	return(TEXT_strBcmp(w1->value,w2->value,l2-1));

    return(TEXT_strcmp(w1->value,w2->value));
}
#endif

int equal_WORD2(void *p1, void *p2){
    int l1, l2; 
    WORD *w1 = ((WORD *)p1), *w2 = ((WORD *)p2);
    /* it stands for internal text string,  This will be modified to 
       remove the opt_del identifier when needed */
    TEXT *it1, *it2;

    if (p1 == (void *)0) return(-1);
    if (p2 == (void *)0) return(1);

    it1 = (! w1->opt_del) ? w1->value : w1->intern_value; 
    it2 = (! w2->opt_del) ? w2->value : w2->intern_value;

    l1 = TEXT_strlen(it1);
    l2 = TEXT_strlen(it2);
    
    if (w1->frag_corr || w2->frag_corr){
      if (*(it1) == '-' && l1 > 1) 
	if (l2 < l1-1)
	  return(1);
	else
	  return(TEXT_strBcmp(it1 + 1,it2 + (l2 - (l1-1)),l1-1));
      else if (*(it1 + l1 - 1) == '-'  && l1 > 1)
	return(TEXT_strBcmp(it1,it2,l1-1));
      
      if (*(it2) == '-' && l2 > 1) 
	if (l1 < l2-1)
	  return(-1);
	else
	  return(TEXT_strBcmp(it1 + (l1 - (l2-1)),it2 + 1,l2-1));
      else if (*(it2 + l2 - 1) == '-' && l2 > 1)
	return(TEXT_strBcmp(it1,it2,l2-1));
    }
    return(TEXT_strcmp(it1,it2));
}

void print_WORD(void *p){
    WORD *tw = (WORD *)p;
    printf("WORD  id=%d  str=%s  conf=%f  tag1=%s  tag2=%s  frag_corr=%d  opt_del=%d  weight=%f\n",
	   (tw == NULL_WORD) ? 0: tw->value_id,
	   (tw == NULL_WORD) ? "NULL_WORD" : (char *)tw->value,
	   (tw == NULL_WORD) ? -1.0 : tw->conf,
	   (tw == NULL_WORD || tw->tag1 == (TEXT *)0) ? (TEXT *)"" : tw->tag1,
	   (tw == NULL_WORD || tw->tag2 == (TEXT *)0) ? (TEXT *)"" : tw->tag2,
	   (tw == NULL_WORD) ? 0 : tw->frag_corr,
	   (tw == NULL_WORD) ? 0 : tw->opt_del,
	   (tw == NULL_WORD) ? 0 : tw->weight);
}

void *copy_WORD(void *p){
    WORD *tw = (WORD *)p, *nw;
    if (tw == (WORD *)0) return((void *)0);
    nw = (void *)new_WORD(tw->value,
			  tw->value_id,tw->T1,tw->T2,tw->conf,
			  tw->tag1,tw->tag2,tw->frag_corr,tw->opt_del,tw->weight);
    return(nw);
}

void *copy_WORD_via_use_count(void *p){
    WORD *tw = (WORD *)p, *nw;
    if (tw == (WORD *)0) return((void *)0);
    tw->use ++;
    return(tw);
}

void *make_empty_WORD(void *p){
    WORD *nw;

    nw = (void *)new_WORD((TEXT*)"",-1,0.0,0.0,0.0,(TEXT *)0,(TEXT *)0, 0, 0, 0.0);
    return(nw);
}

int use_count_WORD(void *p, int i){
  ((WORD*)p)->use += i;
  return(((WORD*)p)->use);
}

void print_WORD_wt(void *p){
    WORD *tw = (WORD *)p;
    printf("WORD %s  ",(tw == NULL_WORD) ? "NULL_WORD" : (char *)tw->value);
    if (tw != NULL_WORD)
	printf("T1:%3.2f  Dur:%3.2f  T2:%3.2f  Conf:%5.4f  Tag1:%s  Tag2:%s  frag_corr=%d  opt_del=%d  weight=%f\n",
	       tw->T1,tw->T_dur,tw->T2,tw->conf,
	       (tw->tag1 == (TEXT *)0) ? (TEXT *)"" : tw->tag1,
	       (tw->tag2 == (TEXT *)0) ? (TEXT *)"" : tw->tag2,
	       tw->frag_corr,tw->opt_del,tw->weight);
    else
	printf("\n");
}

void print_2_WORD_wt(void *p, void *p2, FILE *fp){
    WORD *tw = (WORD *)p;
    WORD *tw2 = (WORD *)p2;
    if (tw == NULL_WORD)
	fprintf(fp,"A: %20s %8s %8s -> ","","","");
    else
	fprintf(fp,"A: %20s %8.2f %8.2f -> ",(char *)tw->value,
		tw->T1,tw->T2);

    if (tw2 == NULL_WORD)
	fprintf(fp,"B: %20s %8s %8s\n","","","");
    else
	fprintf(fp,"B: %20s %8.2f %8.2f\n",(char *)tw2->value,
		tw2->T1,tw2->T2);
}

void set_WORD_tag1(WORD *w, TEXT *t){
    if (w->tag1 != (TEXT *)0) TEXT_free(w->tag1);
    w->tag1 = TEXT_strdup(t);
}

void sgml_dump_WORD(WORD *w, FILE *fp){
  fprintf(fp,"<WORD value=\"%s\" id=\"%d\" tag1=\"%s\" tag2=\"%s\" conf=\"%f\" T1=\"%f\""
	  " T2=\"%f\" fc=\"%d\" od=\"%d\" weight=\"%f\">\n", w->value, w->value_id, 
	  (w->tag1 == (TEXT *)0) ? (TEXT *)"" : w->tag1,
	  (w->tag2 == (TEXT *)0) ? (TEXT *)"" : w->tag2, w->conf,
	  w->T1, w->T2, w->frag_corr,w->opt_del,w->weight);
}

