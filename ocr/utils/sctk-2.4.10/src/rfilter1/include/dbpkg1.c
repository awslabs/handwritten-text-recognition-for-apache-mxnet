/*  file dbpkg1.c  */

#if !defined(COMPILE_ENVIRONMENT)
#include "stdcenvf.h" /* std compile environment for functions */
#endif

/*********************************************************/
/* de-bug package #1                                     */
/* includes fcns:                                        */
/*   db_enter_msg() - displays a procedure-entering msg. */
/*   db_leave_msg() - displays a procedure-leaving msg.  */
/*********************************************************/

/***************************************************/
/*  db_enter_msg(proc,level)                       */
/*  Printf's "enter" msg for procedure whose name  */
/* is *proc, if <level> < <db_level>. Uses global  */
/* *pdb for header string.                         */
/***************************************************/
 void db_enter_msg(Char *proc, int level)
{if (db_level > level) {
   pdb = strcat(pdb,"  ");
   printf("%sentering %s.\n",pdb,proc);
   }
 }

/***************************************************/
/*  db_leave_msg(proc,level                        */
/*  Printf's "leave" msg for procedure whose name  */
/* is *proc, if <level> < <db_level>. Uses global  */
/* *pdb for header string.                         */
/***************************************************/
 void db_leave_msg(Char *proc, int level)
{if (db_level > level) {
   printf("%sleaving  %s.\n",pdb,proc);
   pdb = strcutr(pdb,2);
   }
 }

/* end of dbpkg1.c */
