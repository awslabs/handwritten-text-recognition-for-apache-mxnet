#include "sctk.h"

static int index_search(AUTO_LEX *alex, TEXT *goal, int *success);

void AUTO_LEX_init(AUTO_LEX *alex, int size){
    alex->max = size;
    alex->num = 0;
    alloc_singZ(alex->str,alex->max,TEXT *,(TEXT *)0);
    alloc_singZ(alex->field_a,alex->max,int,0);
    alloc_singZ(alex->field_b,alex->max,int,0);
    alloc_singZ(alex->field_c,alex->max,double,0);
    alloc_singZ(alex->sort,alex->max,int,0);
}

void AUTO_LEX_free(AUTO_LEX *alex){
    free_2dimarr(alex->str,alex->num,TEXT);
    free_singarr(alex->field_a,int);
    free_singarr(alex->field_b,int);
    free_singarr(alex->field_c,double);
    free_singarr(alex->sort,int);
    alex->max = 0;
    alex->num = 0;
}

static int index_search(AUTO_LEX *alex, TEXT *goal, int *success)
{
    int low, high, mid, eval;
 
    low = 0, high = alex->num-1;
 
    if (low > high){
	*success = 0;
	return(low);
    }

    *success = 1;
    do { 
        mid = (low + high)/2;
        eval = TEXT_strcmp(goal,alex->str[alex->sort[mid]]);
/*        printf("%s:  %s (%d) [mid %s (%d)] %s (%d) = res %d\n",
                   goal,list[low],low,list[mid],mid,
                   list[high],high,eval);*/
        if (eval == 0)
            return(mid);
        if (eval < 0)
            high = mid-1;
        else
            low = mid+1;
    } while (low <= high);
 
    *success = 0;
    return(low);
}

int AUTO_LEX_find(AUTO_LEX *alex, TEXT *str){
  int success = 0;
  int ind = index_search(alex,str,&success);
  if (success)
    return(ind);
  return(-1);
}

int AUTO_LEX_insert(AUTO_LEX *alex, TEXT *new){
    char *proc = "AUTO_LEX_insert";
    TEXT **newstr;
    int *newsort, ind, success, *new_field_a, *new_field_b;
    double *new_field_c;

    if (db >= 1) printf("Entering %s: new='%s'\n",proc, new);

    if (alex->num + 1 >= alex->max) {
	if (db >= 5)
	    printf("%s: Expanding Data Array\n",proc);
	alloc_singarr(newstr,alex->max * 2,TEXT *);
	memcpy(newstr,alex->str,sizeof(TEXT *) * alex->max);

	alloc_singarr(newsort,alex->max * 2,int);	
	memcpy(newsort,alex->sort,sizeof(int) * alex->max);

	alloc_singZ(new_field_a,alex->max * 2,int,0);	
	memcpy(new_field_a,alex->field_a,sizeof(int) * alex->max);

	alloc_singZ(new_field_b,alex->max * 2,int,0);	
	memcpy(new_field_b,alex->field_b,sizeof(int) * alex->max);

	alloc_singZ(new_field_c,alex->max * 2,double,0);	
	memcpy(new_field_c,alex->field_c,sizeof(double) * alex->max);
 
	alex->max *= 2;

	free_singarr(alex->str,TEXT *);
	free_singarr(alex->sort,int);
	free_singarr(alex->field_a,int);
	free_singarr(alex->field_b,int);
	free_singarr(alex->field_c,double);

	alex->str = newstr;
	alex->sort = newsort;
	alex->field_a = new_field_a;
	alex->field_b = new_field_b;	
	alex->field_c = new_field_c;	
    }

    ind = index_search(alex,new,&success);
    if (success)
	return(alex->sort[ind]);

    /* insert ind into the list */
    if (db >= 5)
	printf("Adding to Lexicon, ind = %d\n",ind);
    
    /* shift data from ind down to num */
    if (ind < alex->num){
	register int x, *ptr;
	if (db >= 5)
	    printf("Shifting down data array\n");
	for (x=alex->num, ptr=alex->sort + x; x>ind; x--){
	    *ptr = *(ptr-1);
	    ptr--;
	}
    } 
    alex->sort[ind] = alex->num;
    alex->str[alex->num] = TEXT_strdup(new);

    alex->num ++;

    return (alex->num-1);
}

double AUTO_LEX_get_c(AUTO_LEX *alex, int ind){
    char *proc = "AUTO_LEX_get_c";

    if (db >= 1) printf("Entering %s:\n",proc);
    if (ind >= 0 && ind < alex->num)
	return(alex->field_c[ind]);
    else
	return(0.0);
}

TEXT *AUTO_LEX_get(AUTO_LEX *alex, int ind){
    char *proc = "AUTO_LEX_get";

    if (db >= 1) printf("Entering %s:\n",proc);
    if (ind >= 0 && ind < alex->num)
	return(alex->str[ind]);
    else
	return(NULL_TEXT);
}
       
void AUTO_LEX_dump(AUTO_LEX *alex, FILE *fp){
    char *proc = "AUTO_LEX_dump";
    int ind;

    if (db >= 1) printf("Entering %s:\n",proc);

    fprintf(fp,"Dump of AUTO LEX:  %d items\n",alex->num);
    for (ind = 0; ind < alex->num; ind ++){
	fprintf(fp,"ind %4d:  Str: %30s  lsort: %3d  field_a: %3d  field_b: %3d  field_c: %f\n",ind,
		alex->str[ind],
		alex->sort[ind],
		alex->field_a[ind],
		alex->field_b[ind],
		alex->field_c[ind]);		
    }
    fprintf(fp,"\n");
}

void AUTO_LEX_printout(AUTO_LEX *alex, FILE *fp, char *title, int threshhold){
    char *proc = "AUTO_LEX_printout";
    int ind, ord, sum = 0, sumu = 0;

    if (db >= 1) printf("Entering %s:\n",proc);

    /* sort field_a of alex, using field_b as the pointers */
    sort_int_arr(alex->field_a, alex->num, alex->field_b, DECREASING);

    /* do a secondary sort so that the words with equal occurances are sorted */
    {   int beg_pos, pos=0, w, ss;
	while (pos < alex->num){
	    for (beg_pos = pos; pos < alex->num && 
		                alex->field_a[alex->field_b[beg_pos]] ==
		                    alex->field_a[alex->field_b[pos]]; pos++)
		;
	    /* printf("  SEt found from %d to %d\n",beg_pos, pos); */
	    for (ss=beg_pos; ss<pos; ss++){
		for (w=0; w<alex->num && alex->sort[w] != alex->field_b[ss]; w++)
		    ;
		if (w == alex->num) {
		    fprintf(scfp,"Error: internal error in %s, Call NIST\n",proc);
		    exit(1);
		}
		alex->field_b[ss] = w;
	    }	    
	    qsort((void *)(alex->field_b + beg_pos), pos - beg_pos,
		   sizeof(int), qsort_int_compare);
	    /* dereference the field_b pointers back through alex->sort[] */
	    for (ss=beg_pos; ss<pos; ss++)
		alex->field_b[ss] = alex->sort[alex->field_b[ss]];
	    beg_pos = pos;
	}
    }
    /* end of the sorting */

    for (ind = 0, ord = 0; ind < alex->num; ind ++)
	if (alex->field_a[alex->sort[ind]] >= threshhold)
	    sumu++;
    
    fprintf(fp,"%-30s   Total                 (%d)\n",title,alex->num);
    fprintf(fp,"%-30s   With >=%3d occurances (%d)\n","",threshhold,sumu);
    fprintf(fp,"\n");
    for (ind = 0, ord = 0; ind < alex->num; ind ++){
	if (alex->field_a[alex->sort[alex->field_b[ind]]] >= threshhold){
	    fprintf(fp,"%4d: %4d  ->  %s\n",++ord,alex->field_a[alex->field_b[ind]],
		    alex->str[alex->field_b[ind]]);
	    sum += alex->field_a[alex->field_b[ind]];
	}
    }
    fprintf(fp,"     -------\n");
    fprintf(fp,"     %5d\n",sum);
    fprintf(fp,"\n");
}







