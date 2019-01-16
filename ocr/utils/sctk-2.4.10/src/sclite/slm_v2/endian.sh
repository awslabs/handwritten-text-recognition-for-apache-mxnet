#!/bin/sh

# Dumb shell script to report endianness.

echo "  
  #define BIG_ENDIAN      0
  #define LITTLE_ENDIAN   1

  int little_endian(void)
  {
      short int w = 0x0001;
      char *byte = (char *) &w;
      return(byte[0] ? LITTLE_ENDIAN : BIG_ENDIAN);
  }

main () {  
  if(!little_endian()) {
     printf(\"Big-endian, DO NOT set -DSLM_SWAP_BYTES in Makefile\\n\");
  } 
  else {
     printf(\"Little-endian, set -DSLM_SWAP_BYTES in Makefile\\n\");
  } 
}" > test_endian.c
gcc test_endian.c -o test_endian

# Can use cc if gcc not available.

./test_endian
rm -f test_endian test_endian.c
