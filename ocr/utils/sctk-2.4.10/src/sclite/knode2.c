/* file knode2.c */

#include "sctk.h"

void kill_node2(NODE *node1)             

/*******************************************************************/
/* Deletes NODE *node1 and all its arc-lists.                      */
/*******************************************************************/

 {char *proc = "kill_node2";
  ARC_LIST_ATOM *p1, *p1_next; 
/* code */
db_enter_msg(proc,0); /* debug only */
if (db_level > 0) printf("%s killing node @%p, w/name '%s'\n",
  pdb,node1,node1->name);
if (db_level > 1) printf("%s killing in-arc list:\n",pdb);
  for (p1 = node1->in_arcs;  p1 != NULL; p1 = p1_next)
    {p1_next = p1->next;
     /* if (memory_trace) printf("%s FREEing %x\n",pdb,(int)p1); */
     free((void*)p1);
    }   
if (db_level > 1) printf("%s killing out-arc list:\n",pdb);
  for (p1 = node1->out_arcs;  p1 != NULL; p1 = p1_next)
    {p1_next = p1->next;
     /* if (memory_trace) printf("%s FREEing %x\n",pdb,(int)p1); */
     free((void*)p1);
    }
if (db_level > 1) printf("%s killing node name:\n",pdb);
  /* if (memory_trace) printf("%s FREEing %x\n",pdb,(int)node1->name); */
  free((void*)node1->name);
if (db_level > 1) printf("%s killing node:\n",pdb);
  /* if (memory_trace) printf("%s FREEing %x\n",pdb,(int)node1); */
  free((void*)node1);
db_leave_msg(proc,0); /* debug only */
  return;
 } /* end kill_node2() */
