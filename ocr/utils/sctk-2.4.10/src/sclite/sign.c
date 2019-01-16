#define MAIN
#include "sctk.h"


int main(int argc, char **argv){
  double confidence;
  int tptr[2];

  /*  sign_test_analysis(sum_plus,sum_minus,sum_equal,"+","-",0,
				  0.05,verbose, rank->trt_name [ tptr[0] ] ,
				  rank->trt_name [ tptr[1] ] ,
				  tptr,zero_is_best,fp,confidence) */

  sign_test_analysis(20, 10, 0,"+","-",0,
		     0.05, TRUE, "sys1", "sys2", 
  	   	     tptr,TRUE,stdout,&confidence);
}

