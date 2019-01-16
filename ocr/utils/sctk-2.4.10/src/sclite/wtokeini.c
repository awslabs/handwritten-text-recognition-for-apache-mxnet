
#include "sctk.h"

 /*    Modification: 9/14/95 JGF Changed to free the the word array    */

WTOKE_STR1 *WTOKE_STR1_init(char *filename){
    WTOKE_STR1 *tsegs;
    alloc_singarr(tsegs,1,WTOKE_STR1);
    tsegs->s=1;
    tsegs->n=0; 
    tsegs->has_conf=0;
    tsegs->id=(char *)TEXT_strdup((TEXT *)filename);
    tsegs->max = 5000;
    alloc_singarr(tsegs->word,tsegs->max,WTOKE1);
    return(tsegs);
}


