/* file fcndcls.h              */
/* local function declarations */
/* last modified 11/8/96       */

#ifndef FCNDCLS_HEADER
#define FCNDCLS_HEADER

extern Char          *apply_rules2(Char *pb, Char *pa, RULESET2 *rset, int *perr);
extern boolean        atobool(Char *s);
extern Char          *bool_print(boolean x);
extern void          *calloc_safe(size_t nobj, size_t size, Char *calling_proc);
extern void           db_enter_msg(Char *proc, int level);
extern void           db_leave_msg(Char *proc, int level);
extern Char          *del_eol(Char *ps);
extern void           dump_rules2(RULESET2 *rset);
extern void           dump_rules3(RULESET2 *rset, FILE *fp);
extern Char          *expenv(Char *s, int slength);
extern void           fatal_error(Char *reporting_procedure, Char *msg, int error_level);
extern void           free_rules2(RULESET2 *rset);
extern void           free_str(Char *s);
extern void           get_comment_flag(Char *s, Char *comment_flag);
extern void           get_rules2(RULESET2 *rset, Char *path, Char *fname, int *perr);
extern Char          *make_full_fname(Char *sx, Char *path, Char *fname);
extern Char          *make_upper(Char *s);
extern Char          *pltrim(Char *s);
extern Char          *pltrimf(Char *s);
extern Char          *prtrim(Char *s);
extern Char          *prtrim2(Char *s);
extern Char          *remove_comments(Char *s, Char *comment_flag);
extern SUBSTRING      sstok2(Char *sx, Char *delimiters);
extern boolean        string_equal(Char *cs, Char *ct, int ignore_case);
#ifdef NEED_STRCMP
extern int            strcmpi(Char *ps1, Char *ps2);  /* BCD 4.2; not TURBO C */
extern int            strncmpi(Char *ps1, Char *ps2, int n);  /* BCD 4.2; not TURBO C */
#endif
extern Char          *strdup_safe(Char *ps, Char *calling_proc);
extern int            substr_length(SUBSTRING *substr);
extern Char          *substr_to_str(SUBSTRING *substr, Char *str, int lmax);
extern Char          *strcutr(Char *ps, int n);
extern int            textlen(Char *s);
extern boolean        valid_data_line(Char *s, Char *comment_flag);

#endif
/* end file fcndcls.h        */
