#include "sctk.h"

void init_SGML(SGML *sg)
{
    sg->num_tags=0;
}

void dump_SGML_tag(SGML *sg, int n, FILE *fp)
{
    int a;

    fprintf(fp,"TAG: %d  %s  Atrribs:",n,sg->tags[n].name);
    for (a=0; a<sg->tags[n].attrib_num; a++)
	fprintf(fp,"%s=\"%s\" ",sg->tags[n].attrib_name[a],
		sg->tags[n].attrib_value[a]);
    fprintf(fp,"\n");
}

TEXT *get_SGML_attrib(SGML *sg, int tn, TEXT *sname)
{
    int a;

    for (a=0; a<sg->tags[tn].attrib_num; a++){
	if (TEXT_strcmp(sg->tags[tn].attrib_name[a],sname) == 0)
	    return(sg->tags[tn].attrib_value[a]);
    }
    return((TEXT *)"");
}

TEXT *delete_SGML_tag(SGML *sg, TEXT *str)
{
    TEXT *b;
    int a;

    if (sg->num_tags == 0) return(NULL);
    if (*(str+1) != '/') return(NULL);
    if ((b = TEXT_strchr(str+2,'>')) == NULL) return(NULL);
    if (TEXT_strBcmp(sg->tags[sg->num_tags-1].name,str+2,b - (str+2)) != 0)
	return(NULL);
    /* delete the last SGML tag */
    sg->num_tags --;
    for (a=0; a<sg->tags[sg->num_tags].attrib_num; a++){
	free_singarr(sg->tags[sg->num_tags].attrib_value[a],TEXT);
	free_singarr(sg->tags[sg->num_tags].attrib_name[a],TEXT);
    }
    sg->tags[sg->num_tags].attrib_num = 0;
    
    return(sg->tags[sg->num_tags].name);
}

int add_SGML_tag(SGML *sg, TEXT *str)
{
    TEXT *p, *b=str+1;
    int *ac, l;

    if (sg->num_tags >= MAX_TAGS) {
	fprintf(scfp,"Error: Too many SGML tags < %d\n",MAX_TAGS);
	exit(1);
    }

    sg->tags[sg->num_tags].attrib_num = 0;

    /* load the name */
    if ((p = TEXT_strchr(b,' ')) == NULL) return(0);
    sg->tags[sg->num_tags].name = TEXT_strBdup(b,p - b);
    p ++;
    
    b = p;
    /* begin with the attributes */
    ac = &(sg->tags[sg->num_tags].attrib_num);
    while (*b != '\0'){
	b += TEXT_strspn(b,(TEXT *)" \t\n");
	if ((l = TEXT_strcspn(b,(TEXT *)"=\n")) == 0) return (0);
	sg->tags[sg->num_tags].attrib_name[*ac] = TEXT_strBdup(b,l);
	b += l+1;
	if (*(b++) != '"') return(0);
	if ((l = TEXT_strcspn(b,(TEXT *)"\"")) == 0 && (*b != '"')) return (0);
	sg->tags[sg->num_tags].attrib_value[*ac] = TEXT_strBdup(b,l);
	b += l+1;
	b += TEXT_strspn(b,(TEXT *)" \t\n>");
	/*	printf("  Parsed attrib id='%s' value='%s'  remaind='%s'\n",
	       sg->tags[sg->num_tags].attrib_name[*ac],
	       sg->tags[sg->num_tags].attrib_value[*ac],b); */
	(*ac)++;
    }
    sg->num_tags ++;
    return(1);
}
