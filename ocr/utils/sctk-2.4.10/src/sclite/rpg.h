/***************************************************************/
/*    misc external function definitions                       */
/***************************************************************/
#if defined(__STDC__) || defined(__GNUC__) || defined(sgi)
#define PROTO(ARGS)	ARGS
#else
#define PROTO(ARGS)	()
#endif


#define BEFORE_ROW   20
#define AFTER_ROW    30

/* rpg.c */ void Desc_set_parse_verbose(int);
/* rpg.c */ void Desc_set_report_verbose(int);
/* rpg.c */ void Desc_erase(void);
/* rpg.c */ void Desc_set_page_center(int);
/* rpg.c */ void print_spaces(int , FILE *);
/* rpg.c */ void Desc_dump_report(int, FILE *);
/* rpg.c */ void print_line(int wid, FILE *);
/* rpg.c */ void print_blank_line(int, FILE *);
/* rpg.c */ void print_double_line(int, FILE *);
/* rpg.c */ void print_start_line(int, FILE *);
/* rpg.c */ void print_final_line(int, FILE *);
/* rpg.c */ void Desc_add_row_separation(char, int);
/* rpg.c */ void Desc_set_iterated_format(char *);
/* rpg.c */ void Desc_set_iterated_value(char *);
/* rpg.c */ void Desc_flush_iterated_row(void);
#ifdef __STDC__
/* rpg.c */ void Desc_add_row_values PROTO((char *format , ...));
#else
/* rpg.c */ void Desc_add_row_values PROTO((va_alist));
#endif
/* rpg.c */ int char_in_set(char, char *);
/* rpg.c */ int Desc_set_justification(char *);
/* rpg.c */ int is_last_span_col(int, int);
/* rpg.c */ int num_span_col(int, int);
/* rpg.c */ int Desc_dump_ascii_report(char *);
/* rpg.c */ char *get_next_string_value(char **, int);
/* rpg.c */ int is_last_just_col(int, int);
/* rpg.c */ char *center(char *str, int len);
/* rpg.c */ char *Desc_rm_lf(char *s);
