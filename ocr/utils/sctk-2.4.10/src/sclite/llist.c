/**  Mark Przybocki
 **  Filename:  LLIST.C
 **             Module 1 used to build liblist.a (library).
 **
 **/

#include "sctk.h"

static LList *getLList(void *);
static void freeLList(LList **);
void recur(LList **t, LList *f);


/*
 *  Function GETLNODE returns a pointer to a lnode which will
 *  contain the data element taken as input.
 */
static LList *getLList (void *pdata)
{
    LList *ptemp;  /* temporary lnode being set with data element */

    /* ALLOCATE MEMORY for our TEMP LNODE */
    ptemp = malloc(sizeof(LList));

    if (ptemp != NULL) {
	/* we have successfully allocated memory so set the LList */
	ptemp -> data = pdata;  /** DATA ELEMENT                **/
	ptemp -> next = NULL;   /** NEXT LLIST NOT ASSIGNED YET  **/
    }

    return ptemp;  /** RETURNS either a Null LList or a set LList **/
}



/*
 * Function FREELLIST deallocates space occupied by a LList and
 * clears the LList from the list.
 */
static void freeLList (LList **p)
{
    free(*p);  /** DEALLOCATE SPACE  **/
    *p=NULL;   /** Clear value of *p **/
}



/*
 * Function INIT_LIST sets the starting point of a list NULL,
 * which infact gives existence to a list.
 */
void LL_init (LList **s)
{
    *s=NULL;  /** creates a NULL list **/
}

void recur(LList **t, LList *f){ 
  if (! LL_empty(f)){
    recur(t,f->next);
    LL_put_front(t,f->data);
  }
}

void LL_copy (LList **t, LList **f)
{
    LL_init(t);
    recur(t,*f);
}



/*
 * Function EMPTY test whether or not LLIST is empty.
 */
int LL_empty (LList *s)
{
    return ( s==NULL );  /** Test for existing list **/
}



/*
 * Function PUSH places one LList at the head (top) of the list.
 */
int LL_put_front (LList **s, void *pdata)
{
    LList *ptemp;  /** Temporary LList holder **/

    /* Get space and set a LList */
    ptemp = getLList(pdata);

    /* Check for error */
    if (ptemp == NULL)
	return 0;     /** FAILURE: Couldn't allocate space **/

    /* Add to list */
    ptemp -> next = *s;
    
    /** Set the begining of the list **/
    *s = ptemp;

    return 1;   /** SUCCESS **/
}

/*
 * Function LL_put_tail places one LList at the tail (bottom) of the list.
 */
int LL_put_tail (LList **s, void *pdata)
{
    LList *ptemp, *pt1 = *s;  /** Temporary LList holder **/

    if (*s == (LList *)0)
	return LL_put_front(s,pdata);

    /* Get space and set a LList */
    ptemp = getLList(pdata);

    /* Check for error */
    if (ptemp == NULL)
	return 0;     /** FAILURE: Couldn't allocate space **/

    /* traverse to the end of the list */
    while (pt1->next != NULL)
	pt1 = pt1->next;

    /* Add to list */
    pt1 -> next = ptemp;
    ptemp->next = (LList *)NULL;
    
    return 1;   /** SUCCESS **/
}



/*
 *  Function LL_get_first removes a LList for the head (top) of list.
 */
int LL_get_first (LList **s, void **apdata)
{
    LList *ptemp;  /* Temporary LList holder */

    if (LL_empty(*s) == 1)
	return 0;  /** list already empty nothing to pop **/

    /* Set ptemp to the first LList */
    ptemp = *s;

    /* Reset to point to the following LList*/
    *s = ptemp -> next;

    /* Grab the data element */
    *apdata = ptemp -> data;

    /* Free the space of the popped LList */
    freeLList (&ptemp);
    
    return 1; /** Success **/
}

