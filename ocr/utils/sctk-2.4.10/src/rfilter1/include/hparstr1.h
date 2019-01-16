 /* File hparstr1.h --  hash function parameter structure #1 */
 /* Parameterized by HFCN_NPARAMS_MAX                        */
#ifndef HPARSTR1_HEADER
#define HPARSTR1_HEADER

 typedef struct
   {Char *hfcn_name;
    double recommended_load_factor;
    int nparams;
    int C[HFCN_NPARAMS_MAX];
   } HASH_PARAMETERS;

#endif
