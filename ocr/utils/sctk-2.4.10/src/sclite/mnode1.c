/* file mnode1.c */

#include "sctk.h"

NODE *make_node(char *name, NETWORK *net, ARC *from_arc, ARC *to_arc, int *highest_nnode_name, int *perr)

/************************************************************/
/*  Makes a new node, returns a pointer to it.              */
/* The new node is initialized with name, from_arc, to_arc. */
/* Set to be neither start not stop state.                  */
/* All boolean flags are initialized to F.                  */
/* If name is NULL, uses a char string version of bumped    */
/* *highest_nnode_name for the new node name.               */
/************************************************************/
 {char *proc = "make_node";
  char sxx[LINE_LENGTH], *sx = &sxx[0];
  NODE *p;
/* code */
  db_enter_msg(proc,1); /* debug only */
  p = (NODE *)malloc_safe(sizeof(NODE),proc);
  *perr = 0;
  if (name != NULL)
    {p->name = strdup_safe(name,proc);
    }
  else
    {/* make name from node number */
     *highest_nnode_name += 1;
     sprintf(sx,"%d",*highest_nnode_name);
     p->name = strdup_safe(sx,proc);
    }
  p->net      = net ;
  p->in_arcs  = NULL;
  p->out_arcs = NULL;
  p->start_state = F;
  p->stop_state  = F;
  p->flag1       = F;
  p->flag2       = F;
  if (from_arc != NULL){
      p->in_arcs  = add_to_arc_list(p->in_arcs,from_arc,perr);
      if (*perr > 0) {
	  printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	  return((NODE *)0);
      }
  }
  if (to_arc != NULL) {
      p->out_arcs = add_to_arc_list(p->out_arcs,to_arc,perr);
      if (*perr > 0) {
	  printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	  return((NODE *)0);
      }
  }
db_leave_msg(proc,1); /* debug only */
  return p;
 } /* end make_node() */

