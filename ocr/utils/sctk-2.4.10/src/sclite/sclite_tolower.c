
#define MAIN
#include "sctk.h"

#define SCLITE_VERSION "2.9"

TEXT *prog = "sclite_tolower";

void do_exit(char *desc, char *prog, int ret){
    fprintf(stderr,"sclite <encoding> <flags>",prog);
    fprintf(stderr,"\n%s: Error, %s\n\n",prog,desc);
    exit(ret);
}

int main(int argc, char **argv){
  TEXT *in_buf;
  int in_buf_len = 1024;
  int toLower = 1;
  int splitChar = 0;

  if (!TEXT_set_encoding(argv[1]))
    do_exit(rsprintf("Unrecognized character encoding option '%s'",argv[1]),prog,1);
  // Parse the optional localization
  // printf("%d %s\n",argc, argv[2]);
  if (argc >= 3)
    if (TEXT_strcmp(argv[2], "toupper") == 0)
      toLower = 0;
    else if (TEXT_strcmp(argv[2], "tolower") == 0)
      toLower = 1;
    else if (TEXT_strcmp(argv[2], "splitChar") == 0)
      splitChar = 1;
    else if (!TEXT_set_lang_prof(argv[2]))
      do_exit(rsprintf("Optional case conversion localization failed /%s/\n", argv[2]),prog,1);

  if (argc >= 4)
    if (TEXT_strcmp(argv[3], "toupper") == 0)
      toLower = 0;
    else if (TEXT_strcmp(argv[3], "tolower") == 0)
      toLower = 1;
    else if (TEXT_strcmp(argv[3], "splitChar") == 0)
      splitChar = 1;
    else
      do_exit(rsprintf("Optional case direction ! tolower or toupper or splitChar /%s/\n", argv[3]),prog,1);
  //printf("tolower = %d\n",toLower);

  alloc_singZ(in_buf,in_buf_len,TEXT,'\0');

  while (!feof(stdin)){
    if (TEXT_ensure_fgets(&in_buf,&in_buf_len,stdin) == NULL)
      *in_buf = NULL_TEXT;
    if (!feof(stdin)){
      if (! splitChar){
	TEXT_str_case_change_with_mem_expand(&in_buf, &in_buf_len, toLower);
	printf("%s", in_buf);
      } else {
	TEXT *char_buf = NULL_TEXT;
	int size;
	TEXT_str_case_change_with_mem_expand(&in_buf, &in_buf_len, toLower);
	size = TEXT_chrlen(in_buf) * 2;
	alloc_singarr(char_buf, size, TEXT);
	TEXT_separate_chars(in_buf, &char_buf, &size, 0);
	printf("%s", char_buf);
      }
    }
  }
  exit(0);
}
