

#ifndef PROTO_HEADER
#define PROTO_HEADER

#if defined(__STDC__) || defined(__GNUC__) || defined(sgi)
#define PROTO(ARGS)	ARGS
#else
#define PROTO(ARGS)	()
#endif

#ifdef AT_NIST_IGNORE_THESE


extern int	fprintf PROTO(( FILE *fp, char *s, ... ));
int	printf PROTO(( char *s, ... ));
int	fclose PROTO(( FILE *f ));
int	strcasecmp PROTO(( char *s1, char *s2 ));
int	fflush PROTO(( FILE *f ));
int	fgetc PROTO((  FILE *f ));
int	fputc PROTO(( char c, FILE *f ));
int	fputs PROTO(( char *s, FILE *f ));
long	ftell PROTO(( FILE *f ));
int	fwrite PROTO(( char *p, int n, int n2, FILE *f ));
int	rewind PROTO(( FILE *f ));
int     fseek PROTO((FILE *, long int, int));
size_t  fread PROTO((void *, size_t, size_t, FILE *));
void *  malloc PROTO((size_t));
void    free PROTO((void *));
int     vsprintf PROTO((char *, const char *, ...));
int     fscanf PROTO((FILE *, const char *, ...));
char *  fgets PROTO((char *, int, FILE *));
/* void    qsort PROTO((void *, size_t, size_t, int (*) (const void *, const void *))); */
extern int                    _filbuf PROTO((FILE *));
extern int                    strcmp PROTO((const char *, const char *));
extern char *                 strchr PROTO((const char *, int));
extern char *                 strrchr PROTO((const char *, int));
extern int                    strcmp PROTO((const char *, const char *));
extern char *                 strdup PROTO((const char *));
extern int                    strncmp PROTO((const char *, const char *, size_t));

extern int                    strncasecmp PROTO((const char *, const char *, int));
extern int                    unlink PROTO((const char *));
extern int                    rename PROTO((const char *, const char *));
extern int                    __flsbuf PROTO((int, FILE *));
extern void *                 memset PROTO((void *, int, size_t));
extern int                    _flsbuf PROTO((unsigned int, FILE *));
extern int                    bcmp PROTO((const void *, const void *, int));
extern void                   bcopy PROTO((const void *, void *, int));
extern int                    system PROTO((const char *));
extern void                   perror PROTO((const char *));
extern char *                 index PROTO((const char *, int));
extern int		      pclose PROTO((FILE *));

extern size_t                 strlen PROTO((const char *s));
extern char *                 strcpy PROTO((char *s1, const char *s2));

/* time(), toupper(), and tolower() not in /usr/include/X.h */
extern time_t	time PROTO((time_t *t ));   /* from /usr/5include/time.h  */
extern int	toupper PROTO((int c));     /* from /usr/5include/ctype.h */
extern int	tolower PROTO((int c));     /* from /usr/5include/ctype.h */

extern int sscanf PROTO((const char *, const char *, ...));
extern int      isatty PROTO((int));
extern int      getpid PROTO((void));

#endif

#endif
