/* file delarc1.c */


#include "sctk.h"

ARC_LIST_ATOM *del_from_arc_list(ARC_LIST_ATOM **plist, ARC *arc, int *perr)
                                           

/*******************************************************************/
/* Plist is the address of a pointer to an arc list.  Deletes the  */
/* arc pointed to by *arc from this list, modifying *plist if      */
/* needed.                                                         */
/* DOES NOT free the arc itself, since it may be pointed to by     */
/* another arc pointer.                                            */
/* Returns pointer to the new list.                                */
/*******************************************************************/

 {char *proc = "del_from_arc_list";
  ARC_LIST_ATOM *p, *p_next;
/* code */
  db_enter_msg(proc,1); /* debug only */
  *perr = 0;
  for (p = *plist; p != NULL; p = p_next)
    {p_next = p->next;
     if (p->arc == arc)
       {if (p->next != NULL) p->next->prev = p->prev;
        if (p->prev != NULL) p->prev->next = p->next;
        else *plist = p->next;
        free(p);
    }  }
  db_leave_msg(proc,1); /* debug only */
  return *plist;
 } /* end del_from_arc_list() */
