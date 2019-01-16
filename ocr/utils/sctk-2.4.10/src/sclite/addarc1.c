/* file addarc1.c */

#include "sctk.h"

ARC_LIST_ATOM *add_to_arc_list(ARC_LIST_ATOM *list_atom, ARC *arc, int *perr)
                                              

/*******************************************************************/
/* List_atom is a pointer to either an atom of an arc list or NULL */
/* if the list is empty.  Adds arc *arc to end of *list_atom.      */
/* If list_atom is NULL on entry, returns point to newly-created   */
/* list item; otherwise returns list_atom.                         */
/*******************************************************************/

 {char *proc = "add_to_arc_list";
  ARC_LIST_ATOM *new_atom, *px,*first_atom,*last_atom; 
  boolean found;
/* code */
  db_enter_msg(proc,1); /* debug only */
  *perr = 0;
  first_atom = list_atom;
if (db_level > 1) {printf("%sadding arc:\n",pdb); arc->net->arc_func.print(arc);}
/* first check if arc is already in the set */
if (db_level > 2)
  {printf("%slooking for arc:",pdb);
   arc->net->arc_func.print(arc);
 }
  found = F;
  /* Added by Jon F. last_atom should be set to the first atom */
  last_atom = list_atom;
  for (px = list_atom; (!found)&&(px != NULL); px = px->next){
      /* This used to test the data in the arc, just check the addr now */
      /* if (arc->net->arc_func.equal(px->arc->data,arc->data)) */
      if (px->arc == arc)
       {found = T;
        if (db_level > 2) printf("%sarcs are equal\n",pdb);
       }
     last_atom = px;
  }
  if (!found)
    {/* make new atom */
     new_atom = (ARC_LIST_ATOM *)malloc_safe(sizeof(ARC_LIST_ATOM),proc);
     new_atom->arc = arc;
if (db_level > 2) printf("%s new_atom created.\n",pdb);
/* put it into list */
     if (list_atom == NULL) /* empty list */
       {new_atom->next = NULL;
        new_atom->prev = NULL;
        first_atom = new_atom;
if (db_level > 2) printf("%s empty list\n",pdb);
       }
     else                /* add new_atom after last_atom  */
       {new_atom->next = last_atom->next;
        new_atom->prev = last_atom;
        if (last_atom->next != NULL) last_atom->next->prev = new_atom;
        last_atom->next = new_atom;

if (db_level > 2) printf("%s non-empty list\n",pdb);
    }  }
  db_leave_msg(proc,1); /* debug only */
  return first_atom;
 } /* end add_to_arc_list() */




