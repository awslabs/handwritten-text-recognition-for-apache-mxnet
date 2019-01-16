/* booldef.h  - defines a "boolean" type  */
#ifndef BOOLDEF_HEADER
#define BOOLDEF_HEADER

#define boolean int
#define T 1
#define F 0
/* (K&R p. 38 equates non-zero with true,but most C's use 0 and 1.)  */

#endif

extern char *bool_print(boolean x);

/* end of booldef.h */
