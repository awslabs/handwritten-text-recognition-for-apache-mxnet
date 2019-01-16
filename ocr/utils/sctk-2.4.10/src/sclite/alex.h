/* File: alex.h                                                 */

typedef struct AUTO_LEX_struct{
    int max,                            /* total possible tokens */
        num;                            /* current number of stored tokens */
    TEXT **str;                         /* pointer to the list of texts */
    int *field_a;                       /* pointer to a list if intergers */
                                        /* used for any purpose what-so-ever, */
                                        /* Intitializd to ZERO */
    int *field_b;                       /* Same usage as field_a . . . */
    double *field_c;                       /* Same usage as field_a . . . */
    int *sort;                          /* Sorted list of the texts */
} AUTO_LEX;

void AUTO_LEX_init(AUTO_LEX *alex, int size);
void AUTO_LEX_free(AUTO_LEX *alex);
int AUTO_LEX_insert(AUTO_LEX *alex, TEXT *new);
void AUTO_LEX_dump(AUTO_LEX *alex, FILE *fp);
TEXT *AUTO_LEX_get(AUTO_LEX *alex, int ind);
void AUTO_LEX_printout(AUTO_LEX *alex, FILE *fp, char *title, int threshhold);
double AUTO_LEX_get_c(AUTO_LEX *alex, int ind);
int AUTO_LEX_find(AUTO_LEX *alex, TEXT *str);

