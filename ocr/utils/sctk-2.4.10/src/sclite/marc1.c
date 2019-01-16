/* file marc1.c */


#include "sctk.h"

ARC *make_arc(void *dat, NODE *from_node, NODE *to_node, int *perr)
                                                   
/*********************************************************/
/* Makes an arc <from_node/symbol/to_node>, returning a  */
/* pointer to it.                                        */
/*********************************************************/
 {char *proc = "make_arc";
  ARC *p; 
/* code */
  db_enter_msg(proc,1); /* debug only */
  p = (ARC *)malloc_safe(sizeof(ARC),proc);
  *perr = 0;
  p->data = dat;
  p->weight = 0;
  p->net = NULL_NETWORK;
  p->from_node = from_node;
  p->to_node   = to_node;
  db_leave_msg(proc,1); /* debug only */
  return p;
 } /* end make_arc() */
