

/* TEXT definitions */
#define NULL_TEXT '\0'
#define TEXT_COLON ':'

#define ALT_BEGIN '{'
#define ALT_END   '}'
#define COMMENT_CHAR ';'
#define COMMENT_INFO_CHAR '*'

// ASCII - 7 bit
// EXTASCII - Extended ASCII - 8 bit
// GB - ASCII -+ 16 bit characters
// EUC - synonym for GB
// UTF-8 - Variable length encoding

enum TEXT_ENCODINGS { ASCII, EXTASCII, GB, UTF8 };
enum TEXT_COMPARENORM { CASE, NONE };
enum TEXT_LANG_PROFILE { LPROF_GENERIC, LPROF_BABEL_TURKISH, LPROF_BABEL_VIETNAMESE, LPROF_BABEL_LITHUANIAN, LPROF_BABEL_KURMANJI, LPROF_BABEL_CEBUANO, LPROF_BABEL_KAZAKH, LPROF_BABEL_MONGOLIAN, LPROF_BABEL_GUARANI};

typedef unsigned char TEXT;

#define TEXT_xnewline(_s) {int _l = TEXT_strlen(_s); if (_s[_l-1] == '\n') _s[_l-1] = '\0';}

// TEXT* (TEXT *)
TEXT *TEXT_strdup(TEXT *p);
TEXT *TEXT_skip_wspace(TEXT *ptr);

// void (TEXT **, int *)
TEXT *TEXT_str_to_master(TEXT *bufTEXT, int toLow);
void TEXT_str_case_change_with_mem_expand(TEXT **buf, int *len, int toLow);
void TEXT_str_case_change_with_mem_expand_from_array2(TEXT **buf, int *len, TEXT *arr2, int toLow);

//TEXT *TEXT_str_to_low(TEXT *buf);
//TEXT *TEXT_str_to_upp(TEXT *buf);

// TEXT* (TEXT *, TEXT **)
TEXT *TEXT_add(TEXT *p1, TEXT *p2);
TEXT *TEXT_strcat(TEXT *p, TEXT *p1);
TEXT *TEXT_strcpy(TEXT *p1, TEXT *p2);
TEXT *TEXT_strqtok(TEXT *buf, TEXT *set);
TEXT *TEXT_strstr(TEXT *p, TEXT *t);
TEXT *TEXT_strtok(TEXT *p, TEXT *t);
TEXT *tokenize_TEXT_first_alt(TEXT *p, TEXT *set);

// TEXT *(TEXT *, TEXT)
TEXT *TEXT_strchr(TEXT *p, TEXT t);
TEXT *TEXT_strrchr(TEXT *p, TEXT t);

// TEXT* (TEXT *, int)
TEXT *TEXT_strBdup(TEXT *p, int n);
TEXT *TEXT_strBdup_noEscape(TEXT *p, int n);

// TEXT* (TEXT *, TEXT *, int)
TEXT *TEXT_strCcpy(TEXT *p, TEXT *t, int n);
TEXT *TEXT_strBcpy(TEXT *p, TEXT *t, int n);

// TEXT* (TEXT *, TEXT *, TEXT)
TEXT *TEXT_strcpy_escaped(TEXT *p1, TEXT *p2, TEXT chr);
 
// TEXT* (TEXT *, int *, FILE *)
TEXT *TEXT_ensure_fgets(TEXT **arr, int *len, FILE *fp);

// TEXT* (TEXT *, int, FILE *)
TEXT *TEXT_fgets(TEXT *arr, int len, FILE *fp);

// TEXT* (int)
TEXT* TEXT_UTFCodePointToTEXT(long int c);

// float (TEXT *)
float TEXT_atof(TEXT *p);

// int (TEXT *)
int TEXT_chrlen(TEXT *text);
int TEXT_is_comment(TEXT *p);
int TEXT_is_comment_info(TEXT *p);
int TEXT_is_empty(TEXT *p);
int TEXT_is_wfrag(TEXT *text);
int TEXT_strlen(TEXT *p);
int TEXT_nbytes_of_char(TEXT *p);
long int TEXT_getUTFCodePoint(TEXT *buf);

// int (TEXT)
int end_of_TEXT(TEXT text);

// int (TEXT *, TEXT *)
int TEXT_strcasecmp(TEXT *p, TEXT *t);
int TEXT_strcmp(TEXT *p, TEXT *t);

// int (TEXT *, TEXT *, int)
int TEXT_strCcasecmp(TEXT *p1, TEXT *p2, int n);
int TEXT_strCcmp(TEXT *p, TEXT *t, int n);
int TEXT_strBcmp(TEXT *p, TEXT *t, int n);
int find_next_TEXT_alternation(TEXT **ctext, TEXT *token, int len);
int find_next_TEXT_token(TEXT **ctext, TEXT *end_token, int len);

int TEXT_nth_field(TEXT **to, int *to_len, TEXT *from, int field);
int TEXT_set_encoding(char *encoding);
enum TEXT_ENCODINGS TEXT_get_encoding();
int bsearch_TEXT_strcmp(const void *p, const void *p1);
int qsort_TEXT_strcmp(const void *p, const void *p1);

// size_t (TEXT *, TEXT *)
size_t TEXT_strcspn(TEXT *str, TEXT *set);
size_t TEXT_strspn(TEXT *str, TEXT *set);

// void (TEXT *)
void TEXT_free(TEXT *p);

// void (TEXT *, TEXT *, int *, int)
void TEXT_separate_chars(TEXT *from, TEXT **to, int *to_size, int not_ASCII);


/***********************************************************************/
/*   The TEXT_LIST utilities                                           */

typedef struct text_list_struct{
    char *file; /* filename read int */
    int max;   /* The max size for elem */
    int num;   /* The current number of elements in elem */
    TEXT **elem;
} TEXT_LIST;

TEXT_LIST *load_TEXT_LIST(char *file, int col);
TEXT_LIST *init_TEXT_LIST(void);
int add_TEXT_LIST(TEXT_LIST *tl, TEXT *str);
void free_TEXT_LIST(TEXT_LIST **tl);
void dump_TEXT_LIST(TEXT_LIST *tl, FILE *);
int in_TEXT_LIST(TEXT_LIST *tl, TEXT *str);
int WORD_in_TEXT_LIST(void *data, void *elem);
void TEXT_delete_chars(TEXT *arr, TEXT *set);

