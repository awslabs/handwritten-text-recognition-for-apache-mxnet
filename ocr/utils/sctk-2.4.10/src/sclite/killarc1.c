/* file killarc1.c */

#include "sctk.h"

void kill_arc(ARC *arc1)           

/*******************************************************************/
/* Deletes ARC *arc1.                                              */
/* *Perr > 0 on return to signal error.                            */
/*******************************************************************/

 {char *proc = "kill_arc";
/* code */
db_enter_msg(proc,0); /* debug only */

if (db_level > 0) printf("%s killing arc @%p\n",
			 pdb,arc1);

/*   if (memory_trace) printf("%s FREEing %x\n",pdb,(int)arc1->symbol); */
/*   free((void*)arc1->symbol); */
/*   if (memory_trace) printf("%s FREEing %x\n",pdb,(int)arc1); */
  free((void*)arc1);

db_leave_msg(proc,0); /* debug only */
  return;
 } /* end kill_arc() */
