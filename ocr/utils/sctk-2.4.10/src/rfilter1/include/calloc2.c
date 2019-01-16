/* file calloc2.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 void *calloc_safe(size_t nobj, size_t size, Char *calling_proc)
 /*************************************************************/
 /*  Like calloc(nobj,size) except fatal err if calloc fails. */
 /*************************************************************/
 {Char *proc = "calloc_safe";
  void *x;
if (db_level > 3) printf("%sdoing %s\n",pdb,proc);
  x = calloc(nobj,size);
  if (x == NULL) fatal_error(calling_proc,"MEM ALLOC",-1);
  if (memory_trace) printf("%s CALLOC %x\n",pdb,(int)x);
  return x;
 }
/* end calloc2.c */
