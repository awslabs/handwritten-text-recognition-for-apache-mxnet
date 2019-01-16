/* Open a pipe to read from a program without intermediary sh.
   Copyright (C) 1992 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2, or (at your option)
   any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.  */

/* Written by David MacKenzie.  */
/* modified the non-ANSI C definitions of readpipe so as to compile */
/* on an AIX system  -Jon Fiscus                                    */

#include <sctk.h>

#if HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>

#if __STDC__
#include <stdarg.h>
#else
#include <varargs.h>
#endif

#if HAVE_UNISTD_H
# include <unistd.h>
#endif

/* Open a pipe to read from a program without intermediary sh.  Checks
   PATH.  Sample use:
   
   stream = readpipe ("progname", "arg1", "arg2", (char *) 0);
   
   Return 0 on error.  */

#if __STDC__
FILE *
readpipe (char *progname, ...)
#else
FILE *
readpipe (va_alist)
va_dcl
#endif
{
#if DIFF_ENABLED

#if ! __STDC__
  char *progname;
#endif
  int fds[2];
  va_list ap;
  char *args[100];
  int argno = 0;


  /* Copy arguments into `args'. */
#if __STDC__
  va_start (ap, progname);
#else
  va_start (ap);
  progname = va_arg (ap, char *);
#endif
  args[argno++] = progname;
  while ((args[argno++] = va_arg (ap, char *)) != NULL)
    ;
  va_end (ap);


  if (pipe (fds) == -1)
    return 0;

  switch (fork ())
    {
    case 0:			/* Child.  Write to pipe. */
      close (fds[0]);		/* Not needed. */
      if (fds[1] != 1)		/* Redirect 1 (stdout) only if needed.  */
	{
	  close (1);		/* We don't want the old stdout. */
	  if (dup (fds[1]) == 0)/* Maybe stdin was closed. */
	    {
	      dup (fds[1]);	/* Guaranteed to dup to 1 (stdout). */
	      close (0);
	    }
	  close (fds[1]);	/* No longer needed. */
	}
      execvp (args[0], args);	
      _exit (2);		/* 2 for `cmp'. */
    case -1:			/* Error. */
      return 0;
    default:			/* Parent.  Read from pipe. */
      close (fds[1]);		/* Not needed. */
      return (FILE *)fdopen (fds[0], "r");
    }
#else
  return 0;
#endif /* #if DIFF_ENABLED */

}


