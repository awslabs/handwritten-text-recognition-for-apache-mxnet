/* file fatalerr.c */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

 void fatal_error(Char *reporting_procedure, Char *msg, int error_level)

                                                  
/**************************************************************************/
/* Reports fatal error and exits.                                         */
/**************************************************************************/
 {Char *proc = "fatal_error";
/* code */
 db_enter_msg(proc,1); /* debug only */
  fprintf(stderr,"\n");
  fprintf(stderr,"*FATAL ERROR in procedure '%s'\n",reporting_procedure);
  fprintf(stderr,"*FAILURE: %s\n",msg);
  fprintf(stderr,"*ERROR LEVEL: %d\n",error_level);
  fflush(stderr);
  fprintf(stdout,"*FATAL ERROR in procedure '%s'\n",reporting_procedure);
  fprintf(stdout,"*FAILURE: %s\n",msg);
  fprintf(stdout,"*ERROR LEVEL: %d\n",error_level);
  fflush(stdout);
  exit(error_level);
 db_leave_msg(proc,1); /* debug only */
  return;
 }
