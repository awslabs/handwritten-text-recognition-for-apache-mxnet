#include "sctk.h"

#ifdef __STDC__
# include <stdarg.h>
#else
# include <varargs.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_VALUE_LEN   2000

#define LOCKED		'l'
#define UNLOCKED	'u'

static int PAGE_WIDTH=80;
static int CENTER_PAGE=0;
static int static_rpt_dbg=0;
static int static_parse_dbg=0;

typedef struct column_definition_struct{
    char format_str[100];
    int min_just_width;
    int num_col;
    int num_col_chars;
    int *col_just;            /* size by max_col */
    int *col_lock;            /* size by max_col */
    int num_locked;
    char *col_inter_space;    /* size by max_col */
} COLUMN_DEFS;

typedef struct report_definition_struct{
    int width;

    int max_row;       /* The maximum number of rows in the structure */
    int max_col;       /* The maximum number of columns in the structure */
    int max_line_seps; /* The maximum # of line separators between rows */

    int tot_num_col;                    /* the most # columns to expect */
    int num_col_defs;
    COLUMN_DEFS *col_defs;              /* size by max_col */
    int tot_num_row;
    char **before_row_separations;      /* size by max_row X max_line_seps */
    char **after_row_separations;       /* size by max_row X max_line_seps */
    int *row_just;                      /* size by max_row */
    char ***cell_values;                /* size by max_col X max_row X 100 */
    /*   These variables are for column sizing information */
    int **max_col_sizes;   /* this is dependent on the number of columns */
                           /* defined in the structure */
                           /* size by max_col X max_col */
} REPORT_DEF_STRUCT;

#ifdef __cplusplus
}
#endif

void initialize_REPORT_DEF_STRUCT(void);
void expand_columns(void);
void expand_rows(void);
void expand_line_seps(void);
void expand_REPORT_DEF_STRUCT(int targ_col, int targ_row, int targ_linesep);

static REPORT_DEF_STRUCT *rpgstr = (REPORT_DEF_STRUCT *)0;
static int was_initialized = 0;

/* allocate the REPORT_DEF_STRUCT */
void initialize_REPORT_DEF_STRUCT(void){
    int c;

    was_initialized = 1;

    /* first alloc the main structure */
    alloc_singarr(rpgstr,1,REPORT_DEF_STRUCT);

    /* set the status variables */
    rpgstr->tot_num_col = 0;
    rpgstr->num_col_defs = 0;
    rpgstr->tot_num_row = 0;
    
    /* Set the parameterization variables */
    rpgstr->width = 80;
    rpgstr->max_line_seps = 5;
    rpgstr->max_row       = 50;
    rpgstr->max_col       = 50;

    /* allocate some memory */
    alloc_2dimZ(rpgstr->before_row_separations,rpgstr->max_row,
		rpgstr->max_line_seps,char,'\0');
    alloc_2dimZ(rpgstr->after_row_separations,rpgstr->max_row,
		rpgstr->max_line_seps,char,'\0');

    alloc_singZ(rpgstr->row_just,rpgstr->max_row,int,0);

    alloc_3dimZ(rpgstr->cell_values,rpgstr->max_col,rpgstr->max_row,
		MAX_VALUE_LEN, char,'\0');

    alloc_2dimZ(rpgstr->max_col_sizes,rpgstr->max_col,
		  rpgstr->max_col,int,0);

    /* Now alloc the column definitions */
    alloc_singarr(rpgstr->col_defs,rpgstr->max_col,COLUMN_DEFS);
    /* and init all the structures within */
    for (c=0; c<rpgstr->max_col; c++){
	/* initialize the status variables */
	rpgstr->col_defs[c].min_just_width = 
	    rpgstr->col_defs[c].num_col = 
		rpgstr->col_defs[c].num_col_chars = 
		    	rpgstr->col_defs[c].num_locked = 0;
	/* alloc the memory */
	alloc_singZ(rpgstr->col_defs[c].col_just,rpgstr->max_col,int,0); 
	alloc_singZ(rpgstr->col_defs[c].col_lock,rpgstr->max_col,int,0);
	alloc_singZ(rpgstr->col_defs[c].col_inter_space,rpgstr->max_col,
		    char,'\0');
    }
}

void expand_REPORT_DEF_STRUCT(int targ_col, int targ_row, int targ_linesep){
    int c, tr, tc, tlsp, fr, fc, flsp;
    int len = MAX_VALUE_LEN;
    float exp_fact = 1.5, exp_lsp, exp_r, exp_c, exp_len = 1.0;

    /* the initial sizes */
    tlsp = rpgstr->max_line_seps;
    tr   = rpgstr->max_row;
    tc   = rpgstr->max_col;

    /* expansion sizes */
    exp_lsp = (targ_linesep) ? exp_fact : 1.0;
    exp_r   = (targ_row)     ? exp_fact : 1.0;
    exp_c   = (targ_col)     ? exp_fact : 1.0;

    /* the final sizes */
    flsp = (int)(tlsp * exp_lsp);
    fr   = (int)(tr   * exp_r);
    fc   = (int)(tc   * exp_c);

    if (tlsp != flsp || tr != fr){
	/* do the expansion */
	expand_2dimZ(rpgstr->before_row_separations,
		     rpgstr->tot_num_row,tr,exp_r,
		     rpgstr->max_line_seps,tlsp,exp_lsp,
		     char,'\0', FALSE);
	expand_2dimZ(rpgstr->after_row_separations,
		     rpgstr->tot_num_row,tr,exp_r,
		     rpgstr->max_line_seps,tlsp,exp_lsp,
		     char,'\0', FALSE);
    }

    if (tr != fr)
	expand_1dimZ(rpgstr->row_just,tr,tr,exp_r,int,0,FALSE);

    if (tr != fr || tc != fc)
	expand_3dimZ((rpgstr->cell_values),tc,tc,exp_c,tr,tr,exp_r,
		     len, len, exp_len, char,'\0',FALSE);

    if (tc != fc){
	expand_2dimZ(rpgstr->max_col_sizes,tc,tc,exp_c,tc,tc,exp_c,
		     int,0,FALSE);

	/* Now expand the column definitions */
	expand_1dim(rpgstr->col_defs,tc,tc,exp_c,COLUMN_DEFS,FALSE);

	/* for the new structures, initialize them (((SAME CODE AS ABOVE))) */
	/* this make a homogenous set of structures */
	for (c=tc; c<fc; c++){
	    /* initialize the status variables */
	    rpgstr->col_defs[c].min_just_width = 
		rpgstr->col_defs[c].num_col = 
		    rpgstr->col_defs[c].num_col_chars = 
		    	rpgstr->col_defs[c].num_locked = 0;
	    /* alloc the memory */
	    alloc_singZ(rpgstr->col_defs[c].col_just,rpgstr->max_col,int,0); 
	    alloc_singZ(rpgstr->col_defs[c].col_lock,rpgstr->max_col,int,0);
	    alloc_singZ(rpgstr->col_defs[c].col_inter_space,rpgstr->max_col,
			char,'\0');
	}

	/* NOW expand the arrays in ALL the structurs, including the */
	/* newly allocated ones. */
	for (c=0; c<fc; c++){
	    /* alloc the memory */
	    expand_1dimZ(rpgstr->col_defs[c].col_just,tc,tc,exp_c,int,0,FALSE); 
	    expand_1dimZ(rpgstr->col_defs[c].col_lock,tc,tc,exp_c,int,0,FALSE); 
	    expand_1dimZ(rpgstr->col_defs[c].col_inter_space,tc,tc,exp_c,char,
			 '\0',FALSE); 
	}
    }

    /* Change the parameterization variables */
    rpgstr->max_line_seps  = flsp;
    rpgstr->max_row        = fr;
    rpgstr->max_col        = fc;
}

void expand_columns(void){
    expand_REPORT_DEF_STRUCT(1, 0, 0);
}

void expand_rows(void){
    expand_REPORT_DEF_STRUCT(0, 1, 0);
}

void expand_line_seps(void){
    expand_REPORT_DEF_STRUCT(0, 0, 1);
}

int char_in_set(char chr, char *set)
{
    while (*set != '\0'){
	printf("char in set %c -> %c\n",chr,*set);
	if (chr == *set)
	    return(1);
	set++;
    }
    return(0);
}

void Desc_set_parse_verbose(int dbg){
    if (! was_initialized) initialize_REPORT_DEF_STRUCT();
    static_parse_dbg = dbg;
}

void Desc_set_report_verbose(int dbg){
    if (! was_initialized) initialize_REPORT_DEF_STRUCT();
    static_rpt_dbg = dbg;
}

void Desc_erase(void)
{
    int c, r;

    if (! was_initialized) initialize_REPORT_DEF_STRUCT();
    rpgstr->width = 80;
    rpgstr->tot_num_col = 0;
    rpgstr->num_col_defs = 0;
    rpgstr->tot_num_row = 0;
    
    for (r=0 ; r<rpgstr->max_row; r++){
	rpgstr->before_row_separations[r][0] = '\0';	
	rpgstr->after_row_separations[r][0] = '\0';	
    }
    for (c=0; c<rpgstr->max_col; c++)
	rpgstr->col_defs[c].num_col = rpgstr->col_defs[c].num_col_chars= 0;
}

void Desc_set_page_center(int width)
{
    if (! was_initialized) initialize_REPORT_DEF_STRUCT();

    PAGE_WIDTH = width;
    CENTER_PAGE = 1;
}

int Desc_set_justification(char *just_str)
{ 
    char *p;
    int j, col=0;

    if (! was_initialized) initialize_REPORT_DEF_STRUCT();

    /* check the existing set of column definitions, if it's a dup, use it */
    for (j=0; j<rpgstr->num_col_defs; j++)
	if (strcmp(just_str,rpgstr->col_defs[j].format_str) == 0)
	    return(j);

    if (j == rpgstr->max_col)
	expand_columns();

    /* parse the NEW column defintion and set up the structure */
    strcpy(rpgstr->col_defs[j].format_str,just_str);
    for (p=just_str; *p!='\0' && *p!=':'; ){
	if (col >= rpgstr->max_col)
	    expand_columns();
	if ((*p == 'c') || (*p == 'C') || (*p == 'l') || (*p == 'L') ||
	    (*p == 'r') || (*p == 'R') || (*p == 'n') || (*p == 'a')){
	    rpgstr->col_defs[j].col_just[col] = *p;
	    rpgstr->col_defs[j].col_lock[col] = UNLOCKED;
	    rpgstr->col_defs[j].num_col++;
	    p++;
	}
	if (*p == '|'){	
	    rpgstr->col_defs[j].col_inter_space[col] = '|';
	    p++;
	} else if (*p == '='){	
	    rpgstr->col_defs[j].col_inter_space[col] = '=';
	    rpgstr->col_defs[j].num_col_chars++;
	    p++;
	} else 
	    rpgstr->col_defs[j].col_inter_space[col] = ' ';
	col++;
    }
    rpgstr->num_col_defs++;
    if (col > rpgstr->tot_num_col)  /* JGF if this is  true, then expand */
	rpgstr->tot_num_col = col;
    /* now check for the locking column information */
    rpgstr->col_defs[j].num_locked=0;
    if (*p == ':')
	p++;
    for (col=0; *p!='\0' ; col++, p++){
	if (col >= rpgstr->max_col)
	    expand_columns();

	if ((*p == 'l')){
	    rpgstr->col_defs[j].col_lock[col] = LOCKED;
	    rpgstr->col_defs[j].num_locked++;
	} else 
	    rpgstr->col_defs[j].col_lock[col] = UNLOCKED;
    }
    return(j);       
}

char *get_next_string_value(char **str, int width)
{
    static char buf[300];
    char *p;
    int x=0;

    p = *str;

    for (x=0; (x < width) && (*p != '\0') && !(*p == '/' && *(p+1) == '/'); x++, p++)
	buf[x] = *p;
    if (*p == '/' && *(p+1) == '/')
	p+=2;
    else if (*p != '\0' && ((*(p+1) != ' ') || (*(p+1) != '/'))){
	/* backup to a space */
	while ((p != *str) && (*p != ' ')){
	    p--;
	    x--;
	}
	if (*p == ' ')
	    p++;
    }
    buf[x] = '\0';
    *str = p;
    return(buf);
}

void print_spaces(int n, FILE *fp)
{
    int x;
    for (x=0; x<n; x++)
	fprintf(fp," ");
}

#define SPACES_LEN   300
void Desc_dump_report(int space_pad, FILE *fp)
{
    int c, r, x, text_width, c2, j;
    int *hit, column_width;
    char *p;
    char fmt_str1[1000]; 
    char *spaces="                                                                                                                                                                                                                                                                                                            ";
    char **desc_column_ptr;
    int *desc_column_size;
    int *desc_column_text_size;

    if (! was_initialized) initialize_REPORT_DEF_STRUCT();

    alloc_singZ(desc_column_ptr,rpgstr->max_col,char *,NULL);
    alloc_singZ(desc_column_size,rpgstr->max_col,int,0);
    alloc_singZ(desc_column_text_size,rpgstr->max_col,int ,0);
    alloc_singZ(hit,rpgstr->max_col,int,0);
    if (static_rpt_dbg){
	int i,c,r,j,startc;
	printf("Dump of description:   %d Total Columns\n", rpgstr->tot_num_col);
	printf("Columns definitions:   \n");
	for (j=0; j<rpgstr->num_col_defs; j++){
	    printf("    %2d: %2d Col:  ",j,rpgstr->col_defs[j].num_col);
	    for (i=0; i<rpgstr->col_defs[j].num_col; i++)
		printf("%c - '%c' ", rpgstr->col_defs[j].col_just[i],
		       rpgstr->col_defs[j].col_inter_space[i]);
	    printf("\n");
	    printf("          Lock:  ");
	    for (i=0; i<rpgstr->col_defs[j].num_col; i++)
		printf("%c       ", rpgstr->col_defs[j].col_lock[i]);
	    printf("\n");
	}
	printf("Report Width:          %d\n",rpgstr->width);
	printf("Before Row Separations:       ");
	for (r=0; r<rpgstr->tot_num_row; r++){
	    char *p;
	    for (p = rpgstr->before_row_separations[r]; (*p != '\0'); p++)
		printf("  %d-'%c'",r,*p);
	}
	printf("\n");
	printf("After Row Separations:       ");
	for (r=0; r<rpgstr->tot_num_row; r++){
	    char *p;
	    for (p = rpgstr->after_row_separations[r]; (*p != '\0'); p++)
		printf("  %d-'%c'",r,*p);
	}
	printf("\n");
	printf("Table Row Values:      %d Rows\n",rpgstr->tot_num_row);
	for (r=0; r<rpgstr->tot_num_row; r++){
	    printf("   %d:  (Just %d)  ",r,rpgstr->row_just[r]);
	    for (c=0; c<rpgstr->col_defs[rpgstr->row_just[r]].num_col; c++){
		printf("  C%d",c);
		startc = c;
		if ((c < rpgstr->tot_num_col-1) && (rpgstr->col_defs[rpgstr->row_just[r]].col_just[c+1] == 'a')){
		    printf("-");
		    c++;
		    while ((c < rpgstr->tot_num_col-1) && (rpgstr->col_defs[rpgstr->row_just[r]].col_just[c] == 'a'))
			c++;
		    if (c !=  rpgstr->tot_num_col-1)
			c--;
		    printf("%d  ",c);
		} else
		    printf("  ");
		printf("%s  ",rpgstr->cell_values[startc][r]);
	    }
	    printf("\n");
	}
    }
    /* initialize the sizes to the minumum */
    for (c=0 ; c<rpgstr->max_col; c++)
        for (c2=0 ; c2<rpgstr->max_col; c2++)
	    rpgstr->max_col_sizes[c][c2]=0;
    /* first comput the max sizes for rows without spanning columns */
    for (r=0; r<rpgstr->tot_num_row; r++){
	for (c=0 ; c<rpgstr->col_defs[rpgstr->row_just[r]].num_col; c++){
	    if (((c < rpgstr->col_defs[rpgstr->row_just[r]].num_col-1) && 
		 (rpgstr->col_defs[rpgstr->row_just[r]].col_just[c+1] != 'a')) ||
		(c == rpgstr->col_defs[rpgstr->row_just[r]].num_col-1)){
		/* compute the max size of this column */
		p = rpgstr->cell_values[c][r];
		while (*p != '\0'){
		    if (isupper(rpgstr->col_defs[rpgstr->row_just[r]].col_just[c]))
			for (x=0; (*p != '\0') && !(*p == ' ') && !(*p == '/' && *(p+1) == '/'); x++, p++)
			    ;
		    else
			for (x=0; (*p != '\0') && !(*p == '/' && *(p+1) == '/'); x++, p++)
			    ;
		    if (*p == '/' && *(p+1) == '/')
			p+=2;
		    if (*p == ' ')
			p++;
		    if (x >  rpgstr->max_col_sizes[rpgstr->col_defs[rpgstr->row_just[r]].num_col][c])
			 rpgstr->max_col_sizes[rpgstr->col_defs[rpgstr->row_just[r]].num_col][c] = x;
		}
	    }
	}
    }
    if (static_rpt_dbg) {
	for (j=0; j<rpgstr->max_col; j++)
	    hit[j]=0;
	printf("   Maxlen Columns (1st Pass):  \n");
	for (j=0 ; j<rpgstr->num_col_defs; j++){
	    if (!hit[rpgstr->col_defs[j].num_col]){
		printf("      %d Col: ",rpgstr->col_defs[j].num_col);	    	    
		for (c2=0; c2<rpgstr->col_defs[j].num_col; c2++)
		    printf("  %2d", rpgstr->max_col_sizes[rpgstr->col_defs[j].num_col][c2]);
		printf("\n");
 		hit[rpgstr->col_defs[j].num_col]=1;
	    }
	}
	printf("\n");
    }

    /* SECOND compute the max sizes for rows WITH spanning columns */
    for (r=0; r<rpgstr->tot_num_row; r++){
	for (c=0 ; c<rpgstr->col_defs[rpgstr->row_just[r]].num_col; c++){
	    /* if this isn't the last column and the next column is spanning */
	    if ((c < rpgstr->col_defs[rpgstr->row_just[r]].num_col-1) &&
		(rpgstr->col_defs[rpgstr->row_just[r]].col_just[c+1] == 'a')){
		int siz=0, span_siz=0, startc, c2, add;
		/* compute the max size of this column */
		startc=c;
		p = rpgstr->cell_values[c][r];
		while (*p != '\0'){
		    if (isupper(rpgstr->col_defs[rpgstr->row_just[r]].col_just[c]))
			for (x=0; (*p != '\0') && !(*p == ' ') && !(*p == '/' && *(p+1) == '/'); x++, p++)
			    ;
		    else
			for (x=0; (*p != '\0') && !(*p == '/' && *(p+1) == '/'); x++, p++)
			    ;
		    if (*p == '/' && *(p+1) == '/')
			p+=2;
		    if (*p == ' ')
			p++;
		    if (x > siz)
			 siz = x;
		}
		/* compute the size of the columns spanned over */
		span_siz=0;
		while ((c < rpgstr->col_defs[rpgstr->row_just[r]].num_col-1) && (rpgstr->col_defs[rpgstr->row_just[r]].col_just[c+1] == 'a')){
		    span_siz +=  rpgstr->max_col_sizes[rpgstr->col_defs[rpgstr->row_just[r]].num_col][c] + 
			         ((c < rpgstr->col_defs[rpgstr->row_just[r]].num_col-1) ? 1 : 0) + space_pad*2;    
		    c++;
		}
		span_siz +=  rpgstr->max_col_sizes[rpgstr->col_defs[rpgstr->row_just[r]].num_col][c];
		/* if the siz > span_size THEN redistribute the characters over the N columns */
		if (siz > span_siz) {
		    int num_unlocked=0;
		    for (c2=startc; c2<=c; c2++)
			if (rpgstr->col_defs[rpgstr->row_just[r]].col_lock[c2] == UNLOCKED)
			    num_unlocked++;
		    if (static_rpt_dbg) printf("   Redistribute for Row %d, columns %d-%d  Unlocked Col %d ",r,startc,c,num_unlocked);
		    if (static_rpt_dbg) printf(" Span_Size %d  adjusting column size %d   Adding  ",span_siz,siz);
		    if (num_unlocked == 0){
			if (static_rpt_dbg) printf("   NONE\n");
		    } else {
			for (c2=startc; c2<=c; c2++){
			    if (rpgstr->col_defs[rpgstr->row_just[r]].col_lock[c2] == UNLOCKED){
				if (c2 != c)
  				    add = (int)(F_ROUND((double)(siz - span_siz) / (double)(num_unlocked), 0));
				else
				    add = siz - span_siz;
				if (static_rpt_dbg) printf(" %2d",add);
				rpgstr->max_col_sizes[rpgstr->col_defs[rpgstr->row_just[r]].num_col][c2] += add;
				span_siz += add;
			    }
			}
			if (static_rpt_dbg) printf("\n");
		    }
		}
	    }
	}
    }
    if (static_rpt_dbg) {
	for (j=0; j<rpgstr->max_col; j++)
	    hit[j]=0;
	printf("   Maxlen Columns (2nd Pass):  \n");
	for (j=0 ; j<rpgstr->num_col_defs; j++){
	    if (!hit[rpgstr->col_defs[j].num_col]){
		printf("      %d Col: ",rpgstr->col_defs[j].num_col);	    	    
		for (c2=0; c2<rpgstr->col_defs[j].num_col; c2++)
		    printf("  %2d", rpgstr->max_col_sizes[rpgstr->col_defs[j].num_col][c2]);
		printf("\n");
 		hit[rpgstr->col_defs[j].num_col]=1;
	    }
	}
	printf("\n");
    }
    rpgstr->width = 0;
    for (j=0; j<rpgstr->max_col; j++)
	hit[j]=0;
    if (static_rpt_dbg) printf("   Computing Report width per justification:  \n");
    for (j=0 ; j<rpgstr->num_col_defs; j++){
	/* init to account for the `|` on either side of the report */
	hit[rpgstr->col_defs[j].num_col] = 2;
	for (c2=0; c2<rpgstr->col_defs[j].num_col; c2++)
	    hit[rpgstr->col_defs[j].num_col] +=  rpgstr->max_col_sizes[rpgstr->col_defs[j].num_col][c2] + space_pad*2;
	/* add in the extra characters for the column separators */
	hit[rpgstr->col_defs[j].num_col] +=  rpgstr->col_defs[j].num_col_chars + 
	                                    rpgstr->col_defs[j].num_col - 1;

	rpgstr->col_defs[j].min_just_width =  hit[rpgstr->col_defs[j].num_col] ;
	if (rpgstr->width <  hit[rpgstr->col_defs[j].num_col])
	    rpgstr->width = hit[rpgstr->col_defs[j].num_col];
	if (static_rpt_dbg) printf("      Just %d: Col:%d   Width %d\n",j,rpgstr->col_defs[j].num_col,hit[rpgstr->col_defs[j].num_col]);
    }

    if (CENTER_PAGE && (rpgstr->width < PAGE_WIDTH))  print_spaces((PAGE_WIDTH - rpgstr->width)/2, fp);
    print_start_line(rpgstr->width,fp);
    /* produce the report */
    for (r=0; r<rpgstr->tot_num_row; r++){
	char *p;
	int row_not_done, c2, c;	
	int current_just, current_num_col, current_underage, current_add;
	int size_adjustment, current_num_unlocked;
	row_not_done = 1;

	current_just = rpgstr->row_just[r];
	current_num_col = rpgstr->col_defs[current_just].num_col;
	current_num_unlocked = rpgstr->col_defs[current_just].num_col - 
	    rpgstr->col_defs[current_just].num_locked; 
	current_underage = rpgstr->width - 
	    rpgstr->col_defs[current_just].min_just_width;

	/* first set pointers the the values and compute the column sizes and column text sizes */
	if (static_rpt_dbg) printf("   Row %d:  Just:%d  #C:%d  Adj:%d  Widths  ",r,current_just,current_num_col, current_underage);
	current_add = 0;
	for (c=0 ; c<current_num_col; c++){
	    if (rpgstr->col_defs[current_just].col_just[c] != 'a'){
		if (rpgstr->col_defs[current_just].col_lock[c] == LOCKED){
		    desc_column_text_size[c]=rpgstr->max_col_sizes[current_num_col][c];
		    desc_column_size[c]=(space_pad*2) + desc_column_text_size[c];
		} else {
		    desc_column_ptr[c] = rpgstr->cell_values[c][r];
		    size_adjustment = 0;
		    //		    printf("CA=%d ",current_add);
		    if (c != current_num_col-1){
			size_adjustment = (int)(F_ROUND((double)current_underage / (double)current_num_unlocked * (double)(c + num_span_col(current_just,c) + 1),0)) - current_add;
			//printf ("1 set sa=%d %d %d %d %d %d",size_adjustment, current_underage,current_num_unlocked, c, num_span_col(current_just,c), current_add);
			current_add += size_adjustment;
			//printf(" size_adjustment=%d curent_add=%d x9 ", size_adjustment, current_add);
		    } else {
			size_adjustment += current_underage - current_add;
			//printf ("2 set size_adjustment=%d curent_add=%d ",size_adjustment, current_add);
		    }
		    desc_column_text_size[c]=rpgstr->max_col_sizes[current_num_col][c];
		    desc_column_size[c]=(space_pad*2) + desc_column_text_size[c] + size_adjustment;
		    /* compute the size for this value */
		    for (c2=c+1; (c2 < current_num_col) && (rpgstr->col_defs[current_just].col_just[c2] == 'a'); c2++){
			if (c != current_num_col-1){
			    size_adjustment = (int)(F_ROUND((double)current_underage / (double)current_num_col * (double)(c + num_span_col(current_just,c) + 1), 0)) - current_add;
			    current_add += size_adjustment;
			    //printf ("3 set size_adjustment=%d curent_add=%d ",size_adjustment, current_add);
			} else {
			    size_adjustment += current_underage - current_add;
			    //printf ("4 set size_adjustment=%d curent_add=%d ",size_adjustment, current_add);
			}
			desc_column_size[c] += rpgstr->max_col_sizes[current_num_col][c2] + space_pad*2 + 1 + size_adjustment;
			desc_column_text_size[c] +=  rpgstr->max_col_sizes[current_num_col][c2] + space_pad*2 + 1;
		    }
		}
	    } else {
		desc_column_ptr[c] = "";
		desc_column_text_size[c] = desc_column_size[c] = 0;
	    }
	    if (static_rpt_dbg) printf("   C%d:%d-%d(%d) ",c,desc_column_text_size[c], desc_column_size[c],current_add);
	}
	if (static_rpt_dbg) printf("\n");

	for (p=rpgstr->before_row_separations[r]; *p != '\0'; p++) { /* add a row separation */
	    if (CENTER_PAGE && (rpgstr->width < PAGE_WIDTH)) print_spaces((PAGE_WIDTH - rpgstr->width)/2, fp);
	    fprintf(fp,"|");
	    for (c=0 ; c<current_num_col; c++){
		if (rpgstr->col_defs[current_just].col_just[c] != 'a'){
		    for (x=0; x< desc_column_size[c]; x++)
			fprintf(fp,"%c",*p);
		    if (! is_last_just_col(current_just,c)){
			int sp = num_span_col(current_just,c);
			char cchr = rpgstr->col_defs[current_just].col_inter_space[sp + c];
			char lcchr = (r > 0) ? rpgstr->col_defs[rpgstr->row_just[r-1]].col_inter_space[sp + c] : '\0';
			if ( cchr != ' '){
			    if (*p == ' '){
				if (cchr == '=') 
				    fprintf(fp,"||");
				else
				    fprintf(fp,"%c",cchr);
			    } else if (((cchr == '|') || (cchr == '=')) &&
				     (*p == '-') &&
				     ((r > 0) && ((lcchr == '|') || (lcchr == '=')))){
				fprintf(fp,"+");
				if (cchr == '=') fprintf(fp,"+");
			    } else {
				fprintf(fp,"%c", *p);
				if (cchr == '=') fprintf(fp,"%c", *p);
			    }
			} else	{
			    if (cchr == '=') fprintf(fp,"%c", *p); 
			    fprintf(fp,"%c", *p);
			}
		    }
		}
	    }
	    fprintf(fp,"|\n");	    
	}

	while (row_not_done){
	    row_not_done = 0;
	    if (CENTER_PAGE && (rpgstr->width < PAGE_WIDTH)) print_spaces((PAGE_WIDTH - rpgstr->width)/2, fp);
	    fprintf(fp,"|");
	    for (c=0 ; c< current_num_col; c++){
		column_width=desc_column_size[c];
		text_width=desc_column_text_size[c];
		if (rpgstr->col_defs[current_just].col_just[c] != 'a'){
		    switch (rpgstr->col_defs[current_just].col_just[c]){
		      case 'c':
		      case 'C':
			*fmt_str1='\0';
			sprintf(fmt_str1,"%s%%%ds%s", spaces+SPACES_LEN-((column_width-text_width)/2),text_width,
				spaces+SPACES_LEN-(column_width-text_width - ((column_width-text_width)/2)));
			fprintf(fp,fmt_str1,
				center(get_next_string_value(&(desc_column_ptr[c]),text_width),text_width));
			if (*desc_column_ptr[c] != '\0')
			    row_not_done = 1;
			break;
		      case 'r':
		      case 'R':
			*fmt_str1='\0';
			sprintf(fmt_str1,"%s%%%ds%s",spaces+SPACES_LEN-((column_width-text_width)/2),text_width,
				spaces+SPACES_LEN-(column_width-text_width - ((column_width-text_width)/2)));
			fprintf(fp,fmt_str1,get_next_string_value(&(desc_column_ptr[c]),text_width));
			if (*desc_column_ptr[c] != '\0')
			    row_not_done = 1;
			break;
		      case 'l':
		      case 'L':
			*fmt_str1='\0';
			sprintf(fmt_str1,"%s%%-%ds%s",spaces+SPACES_LEN-((column_width-text_width)/2),text_width,
				spaces+SPACES_LEN-(column_width-text_width - ((column_width-text_width)/2)));
			fprintf(fp,fmt_str1,get_next_string_value(&(desc_column_ptr[c]),text_width));
			if (*desc_column_ptr[c] != '\0')
			    row_not_done = 1;
			break;
		      default:
			fprintf(fp,"undefined inter column space\n");
		    }
		    if (! is_last_just_col(current_just,c)){
			int sp;
			sp =  num_span_col(current_just,c);
			if ( rpgstr->col_defs[current_just].col_inter_space[sp + c] != ' '){
			    if (rpgstr->col_defs[current_just].col_inter_space[sp + c] == '=')
				fprintf(fp,"||");
			    else
				fprintf(fp,"%c", rpgstr->col_defs[current_just].col_inter_space[sp + c]);
			} else
			    fprintf(fp," ");
		    }
		}
	    }
	    fprintf(fp,"|\n");
	}
	for (p=rpgstr->after_row_separations[r]; *p != '\0'; p++) { /* add a row separation */	
	    if (CENTER_PAGE && (rpgstr->width < PAGE_WIDTH)) print_spaces((PAGE_WIDTH - rpgstr->width)/2, fp);
	    fprintf(fp,"|");
	    for (c=0 ; c<current_num_col; c++){
		if (rpgstr->col_defs[current_just].col_just[c] != 'a'){
		    for (x=0; x< desc_column_size[c]; x++)
			fprintf(fp,"%c",*p);
		    if (! is_last_just_col(current_just,c)){
			int sp;
			sp =  num_span_col(current_just,c);
			if ( rpgstr->col_defs[current_just].col_inter_space[sp + c] != ' '){
			    if (*p == ' ')
				fprintf(fp,"%c", rpgstr->col_defs[current_just].col_inter_space[sp + c]);
			    else if ((rpgstr->col_defs[current_just].col_inter_space[c] == '|') && (*p == '-') &&
				     ((r > 0) && (rpgstr->col_defs[rpgstr->row_just[r-1]].col_inter_space[sp + c] == '|')))
				fprintf(fp,"+");
			    else 
				fprintf(fp,"%c", *p);
			} else	
			    fprintf(fp,"%c", *p);
		    }
		}
	    }
	    fprintf(fp,"|\n");	    
	}
    }
    if (CENTER_PAGE && (rpgstr->width < PAGE_WIDTH)) print_spaces((PAGE_WIDTH - rpgstr->width)/2, fp);
    print_final_line(rpgstr->width,fp);

    free_singarr(hit,int);
    free_singarr(desc_column_ptr,char *);
    free_singarr(desc_column_size,int);
    free_singarr(desc_column_text_size,int);
}

int is_last_just_col(int just, int col)
{
    int c;
    if (col == rpgstr->col_defs[just].num_col-1)
	return(1);
    c = col+1;
    while ((c < rpgstr->col_defs[just].num_col) && ( rpgstr->col_defs[just].col_just[c] == 'a'))
	c++; 
    if ((c == rpgstr->col_defs[just].num_col))
	return(1);
    return(0);	   
}

int is_last_span_col(int just, int col)
{
    if ((col+1 < rpgstr->col_defs[just].num_col) && ( rpgstr->col_defs[just].col_just[col] == 'a'))
	return(0); 
    return(1);
}

void print_line(int wid, FILE *fp)
{
    int i;
    fprintf(fp,"|");
    for (i=0; i<wid-2; i++)
	fprintf(fp,"-");
    fprintf(fp,"|\n");
}

void print_blank_line(int wid, FILE *fp)
{
    int i;
    fprintf(fp,"|");
    for (i=0; i<wid-2; i++)
	fprintf(fp," ");
    fprintf(fp,"|\n");
}

void print_double_line(int wid, FILE *fp)
{
    int i;
    fprintf(fp,"|");
    for (i=0; i<wid-2; i++)
	fprintf(fp,"=");
    fprintf(fp,"|\n");
}

void print_start_line(int wid, FILE *fp)
{
    int i;
    fprintf(fp,",");
    for (i=0; i<wid-2; i++)
	fprintf(fp,"-");
    fprintf(fp,".\n");
}

void print_final_line(int wid, FILE *fp)
{
    int i;
    fprintf(fp,"`");
    for (i=0; i<wid-2; i++)
	fprintf(fp,"-");
    fprintf(fp,"'\n");
}

void Desc_add_row_separation(char chr, int row_attach)
{
    char *p=NULL;

    if (! was_initialized) initialize_REPORT_DEF_STRUCT();

    if (rpgstr->tot_num_row >= rpgstr->max_row)
	expand_rows();

    if (row_attach == BEFORE_ROW){
	for (p = rpgstr->before_row_separations[rpgstr->tot_num_row];
	     (*p != '\0') && 
	     (p < rpgstr->before_row_separations[rpgstr->tot_num_row] + 
	      rpgstr->max_line_seps-1);
	     p++)
	    ;
	if (p == (rpgstr->before_row_separations[rpgstr->tot_num_row] +
		  rpgstr->max_line_seps-1)){
	    expand_line_seps();
	    /* redo the above step */
	    for (p = rpgstr->before_row_separations[rpgstr->tot_num_row];
		 (*p != '\0') && 
		 (p < rpgstr->before_row_separations[rpgstr->tot_num_row] + 
		  rpgstr->max_line_seps-1);
		 p++)
		;
	}
	*p = chr; *(p+1) = '\0';
    } else {
	for (p = rpgstr->after_row_separations[rpgstr->tot_num_row];
	     (*p != '\0') &&
	     (p < rpgstr->after_row_separations[rpgstr->tot_num_row] + 
	      rpgstr->max_line_seps-1);
	     p++)
	    ;
	if (p == (rpgstr->after_row_separations[rpgstr->tot_num_row] +
		  rpgstr->max_line_seps-1)){
	    expand_line_seps();
	    /* REDO the above step, row_seps is a new array */
	    for (p = rpgstr->after_row_separations[rpgstr->tot_num_row];
		 (*p != '\0') &&
		 (p < rpgstr->after_row_separations[rpgstr->tot_num_row] + 
		  rpgstr->max_line_seps-1);
		 p++)
		;
	}
	*p = chr; *(p+1) = '\0';
    }
}

int num_span_col(int just, int col)
{
    int c, span=0;
    if (col == rpgstr->col_defs[just].num_col-1)
	return(0);
    c = col+1;
    while ((c < rpgstr->col_defs[just].num_col) && ( rpgstr->col_defs[just].col_just[c] == 'a')){
	c++; 
	span++;
    }
    return(span);
}

void setup_iterations(void);
static int iter_num_val=(-1), iter_max_val = (-1);
static char **ival;
static char iter_fmt[MAX_VALUE_LEN];

void setup_iterations(void){
    if (iter_num_val == (-1)){
	iter_num_val = 0;
	iter_max_val = 60;
	alloc_2dimarr(ival,iter_max_val,MAX_VALUE_LEN,char);
    }
    if (rpgstr->max_col > iter_max_val){
	fprintf(stderr,"Error: You just tried to add information into column %d using\n",
		iter_max_val);
	fprintf(stderr,"       the iterated report generation functions.  This\n");
	fprintf(stderr,"       suite of functions could not be made general in size\n");
	fprintf(stderr,"       because the 'Desc_flush_iterated_row()' function must\n");
	fprintf(stderr,"       explicitly list N arguments.  You will have to manually\n");
	fprintf(stderr,"       change, iter_max_val, and to function to handle more columns\n");
	exit(1);
/* 	expand_2dimZ(ival,iter_num_val,iter_max_val,(rpgstr->max_col / iter_max_val) + 0.5,
		     len,len,1.0,char,'\0',TRUE);*/
    }
}

void Desc_set_iterated_format(char *format)
{
    if (! was_initialized) initialize_REPORT_DEF_STRUCT();
    setup_iterations();

    strcpy(iter_fmt,format);
}

void Desc_set_iterated_value(char *str)
{
    if (! was_initialized) initialize_REPORT_DEF_STRUCT();
    setup_iterations();
    if (iter_num_val > 59) {
      fprintf(stderr,"Error: an interated value was added to a column outside the RPG's defined size\n");
      exit(1);
    }
   
    strcpy(ival[iter_num_val++],str);
}

void Desc_flush_iterated_row(void)
{
    int i;

    if (! was_initialized) initialize_REPORT_DEF_STRUCT();
    setup_iterations();

    for (i=iter_num_val; i<rpgstr->max_col; i++)
	ival[i][0] = '\0';
/*     printf("iter_num_val = %d  '%s'\n",iter_num_val,iter_fmt);*/
    Desc_add_row_values(iter_fmt,ival[0],ival[1],ival[2],ival[3],ival[4],
			ival[5],ival[6],ival[7],ival[8],ival[9],
			ival[10],ival[11],ival[12],ival[13],ival[14],
			ival[15],ival[16],ival[17],ival[18],ival[19],
			ival[20],ival[21],ival[22],ival[23],ival[24],
			ival[25],ival[26],ival[27],ival[28],ival[29],
			ival[30],ival[31],ival[32],ival[33],ival[34],
			ival[35],ival[36],ival[37],ival[38],ival[39],
			ival[40],ival[41],ival[42],ival[43],ival[44],
			ival[45],ival[46],ival[47],ival[48],ival[49],
			ival[50],ival[51],ival[52],ival[53],ival[54],
			ival[55],ival[56],ival[57],ival[58],ival[59]); 

    iter_num_val=0;
}



#ifdef __STDC__
void Desc_add_row_values(char *format , ...)
#else
void Desc_add_row_values(va_alist)
va_dcl
#endif
{
    va_list args;
    char *str;
    int col=0;
#ifndef __STDC__    
    char *format;
#endif

#ifdef __STDC__    
    va_start(args,format);
#else
    va_start(args);
    format = va_arg(args,char *);
#endif

    if (! was_initialized) initialize_REPORT_DEF_STRUCT();

    if (rpgstr->tot_num_row >= rpgstr->max_row)
	expand_rows();

    if (static_parse_dbg) printf("Desc_add_row_values\n");
    rpgstr->row_just[rpgstr->tot_num_row] = 
	Desc_set_justification(format);
    if (static_parse_dbg) 
	printf("    Format %s   id %d\n",format,
	       rpgstr->row_just[rpgstr->tot_num_row]);
    for (col=0; col<rpgstr->col_defs[rpgstr->row_just[rpgstr->tot_num_row]].num_col; col++){
	if (rpgstr->col_defs[rpgstr->row_just[rpgstr->tot_num_row]].col_just[col] != 'a'){
	    str = va_arg(args, char *);
	    strcpy(rpgstr->cell_values[col][rpgstr->tot_num_row],str);
	    if (static_parse_dbg) printf("    value %d %s\n",col,str);
	}
    }
    va_end(args);
    rpgstr->tot_num_row ++;
}


int Desc_dump_ascii_report(char *file)
{
    int c, r;
    char *p;
    FILE *fp;

    if (! was_initialized) initialize_REPORT_DEF_STRUCT();

    if (strcmp(file,"") == 0)
	return(1);
    if ((fp=fopen(file,"w")) == (FILE *)NULL){
	fprintf(stderr,"Error: unable to open ascii report file '%s'\n",file);
	return(1);
    }
    
    if (CENTER_PAGE)
	fprintf(fp,"center_page %d\n",rpgstr->width);
    for (r=0; r<rpgstr->tot_num_row; r++){
	for (p = rpgstr->before_row_separations[r]; (*p != '\0'); p++)
	    fprintf(fp,"separate \"%c\" BEFORE_ROW\n",*p);
	for (p = rpgstr->after_row_separations[r]; (*p != '\0'); p++)
	    fprintf(fp,"separate \"%c\" AFTER_ROW\n",*p);
	fprintf(fp,"row \"%s\"",rpgstr->col_defs[rpgstr->row_just[r]].format_str);
	for (c=0; c<rpgstr->col_defs[rpgstr->row_just[r]].num_col; c++)
	    if (rpgstr->col_defs[rpgstr->row_just[r]].col_just[c] != 'a')
		fprintf(fp," \"%s\"",rpgstr->cell_values[c][r]);
        fprintf(fp,"\n");
    }
    fclose(fp);
    return(0);
}

char *Desc_rm_lf(char *s){
    static int len=0;
    static char *buf, *p;
    int n;
    
    if (! was_initialized) initialize_REPORT_DEF_STRUCT();

    if ((n = strlen(s) + 1) > len){
	if (len != 0) free_singarr(buf,char);
	len = n;
	alloc_singarr(buf,n,char);
    }
    p = buf;

    /* Copy the string in, if two /'s are found, convert it to a space */
    while (*s != '\0'){
	if (*s != '/') 
	    *(p++) = *(s++);
	else if (*(s+1) == '/'){
	    s+=2;
	    *(p++) = ' ';
	} else {
	    *(p++) = *(s++);
	}
    }
    *p = '\0';
    return(buf);
}
