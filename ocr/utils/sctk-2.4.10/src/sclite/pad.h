void init_pad_util PROTO((int pr_width)) ;
int pad_pr_width PROTO((void)) ;
void set_pad PROTO((char *pad, char *str, int max)) ;
void set_pad_n PROTO((char *pad, int n, int max)) ;
void set_pad_cent_n PROTO((char *pad, int len, int max)) ;
char *center PROTO((char *str, int len)) ;
void strncpy_pad PROTO((char *to, char *from, int len, int max, char chr)) ;
