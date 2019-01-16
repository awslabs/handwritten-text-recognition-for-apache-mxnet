#define MAIN
#include "sctk.h"

int main(int argc, char **argv){
  int **table;

  alloc_2dimarr(table, 2, 2, int);
  table[0][0] = 100;
  table[0][1] = 200;
  table[1][0] = 300;
  table[1][1] = 400;

  double conf;
  do_McNemar(table, "Sys1", "Sys2", TRUE, stdout, &conf);



}
