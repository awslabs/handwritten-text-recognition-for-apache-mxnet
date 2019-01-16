#include "sctk.h"
 
void dump_word_tokens2(WTOKE_STR1 *word_tokens, int start, int lim)
{int i;
   printf("\n");
   printf(" DUMP OF WORD TOKENS (OCCURRENCES):\n");
   printf(" (Starting at word %d going to %d)",start,lim);
   printf("\n");
   printf(" No. of word tokens = %d     Id = %s\n",
          word_tokens->n,word_tokens->id);
   printf("                             CON-  OVER  MIS  UN- CROS COM- \n");
   printf(" CONV  TURN   T1      DUR    FID   LAPD PRON SURE TALK MENT BAD ALT IGN WORD\n");
   for (i=start; (i<=word_tokens->n)&&(i<=lim); i++)
       {printf("%4s  %1s  %9.2f  %6.2f  %6.2f  %s    %s    %s    %s    %s    %s   %s   %s   %s\n",
       word_tokens->word[i].conv,
       word_tokens->word[i].turn,
       word_tokens->word[i].t1,
       word_tokens->word[i].dur,
       word_tokens->word[i].confidence,
       bool_print(word_tokens->word[i].overlapped),
       bool_print(word_tokens->word[i].mispronounced),
       bool_print(word_tokens->word[i].unsure),
       bool_print(word_tokens->word[i].crosstalk),
       bool_print(word_tokens->word[i].comment),
       bool_print(word_tokens->word[i].bad_marking),
       bool_print(word_tokens->word[i].alternate),
       bool_print(word_tokens->word[i].ignore),
       word_tokens->word[i].sp);
    }
   return;
} /* end of function "dump_word_tokens2" */
 
void dump_WTOKE_words(WTOKE_STR1 *word_tokens, int start, int lim, char *file)
{
    int i;
    FILE *fp = fopen(file,"w");
    
    if (fp == NULL){
	fprintf(stderr,"Error: Can't open STM words file '%s'\n",file);
	exit(1);
    }

    for (i=start; (i<=word_tokens->n)&&(i<=lim); i++)
	fprintf(fp,"%s\n",word_tokens->word[i].sp);

    fclose(fp);
} /* end of function "dump_WTOKE_words" */
 



