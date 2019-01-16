/**********************************************************************/
/*                                                                    */
/*             FILENAME:  order.h                                     */
/*             BY:  Jonathan G. Fiscus                                */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*           DESC:  This file contains defines for ordering arrays    */
/*                                                                    */
/**********************************************************************/
#define DECREASING	1
#define INCREASING	0

#if defined(__STDC__) || defined(__GNUC__) || defined(sgi)
#define PROTO(ARGS)	ARGS
#else
#define PROTO(ARGS)	()
#endif


/* order.c */    void rank_int_arr PROTO((int *arr, int num, int *ptr_arr, double *rank_arr, int order)) ;
/* order.c */    void sort_short_arr PROTO((short int *arr, int num, int *ptr_arr, int order)) ;
/* order.c */    void sort_double_arr PROTO((double *arr, int num, int *ptr_arr, int order)) ;
/* order.c */    void sort_int_arr PROTO((int *arr, int num, int *ptr_arr, int order)) ;
/* order.c */    void sort_strings_in_place PROTO((char **arr, int num, int order)) ;
/* order.c */    void sort_strings_using_index PROTO((char **arr, int *ind, int num, int order));
/* order.c */    void rank_double_arr PROTO((double *arr, int num, int *ptr_arr, double *rank_arr, int order)) ;
/* order.c */    int qsort_int_compare(const void *i, const void *j);
/* order.c */    int qsort_double_compare(const void *i, const void *j);
