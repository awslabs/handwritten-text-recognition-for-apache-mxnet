/********************************************/
/* file stdcenvp.h                          */
/* standard compile environment for progs   */
/********************************************/

#define COMPILE_ENVIRONMENT 1

#ifdef __cplusplus  /* if compiling with a C++ compiler */
extern "C" {
#endif

#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <sys/stat.h>

/* standard program parameter settings */
#include "stdpars.h"


  /* JGF  Redefined the character type  to be unsigned so that mandarin and
          japanese will parse correctly */
#define Char unsigned char

/* type definitions */
#include "booldef.h"
#include "rulestr2.h"
#include "ssstr1.h"

/* macros */
#include "strmacs.h"

/* global data declarations (NOT definitions) */
#include "gdatadef.h"

/* function declarations */
#include "fcndcls.h"

/* prototypes of system functions */
#ifdef AT_NIST
#include "/usr/local/nist/general/include/util/proto.h"
#endif

#ifdef __cplusplus
}
#endif
