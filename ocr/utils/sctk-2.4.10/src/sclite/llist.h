/**
 **  Mark Przybocki
 **  598 Unix/c
 **
 **  Project #7:  PART I  (creating a library)
 **
 **  Filename:  LLIST.H
 **             header file for the liblist.a library
 **             and other linked list functions.
 **
 **/

/** DEFINE THE BASIC NODE STRUCTURE **/

typedef struct lnode LList;
struct lnode
{
 void *data;  /* data element            */
 void *next;  /* pointer to next element */
};

/** FUNCTION PROTO-TYPING **/

    /* file: llist.c */

int LL_put_tail(LList **, void *);
int LL_put_front(LList **, void *);
int LL_get_first(LList **, void **);
void LL_init(LList **);
void LL_copy(LList **, LList **);
int LL_empty(LList *);

