/* file arcseq1.c */

#include "sctk.h"

boolean arcs_equal(ARC *arc1, ARC *arc2)                  
/***************************************************/
/* Determines if two arcs of a network are equal.  */
/***************************************************/
{if ((arc1->from_node == arc2->from_node) &&
/*      (streq(arc1->symbol,arc2->symbol)) && */
     (arc1->to_node   == arc2->to_node)) return T;
 else  return F;
}
