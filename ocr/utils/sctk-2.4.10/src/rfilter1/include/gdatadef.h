/* file gdatadef.h         */
/* global data definitions */
#ifndef GDATADEF_HEADER
#define GDATADEF_HEADER

 int db_level = 0;
 int detail_level = 0;
 Char db[272] = "*DB: ", *pdb = &db[0];
 boolean memory_trace = F;
 Char null_char_string[1] = "", *EMPTY_STRING = &null_char_string[0];
 Char *NIL_STRING = "(nil)";
 int nstr_empty[1] = {0}, *EMPTY_INT_STRING = &nstr_empty[0];

#endif
