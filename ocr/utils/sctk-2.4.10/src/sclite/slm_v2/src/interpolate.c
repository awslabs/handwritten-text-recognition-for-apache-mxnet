

/*=====================================================================
                =======   COPYRIGHT NOTICE   =======
Copyright (C) 1996, Carnegie Mellon University, Cambridge University,
Ronald Rosenfeld and Philip Clarkson.

All rights reserved.

This software is made available for research purposes only.  It may be
redistributed freely for this purpose, in full or in part, provided
that this entire copyright notice is included on any copies of this
software and applications and derivations thereof.

This software is provided on an "as is" basis, without warranty of any
kind, either expressed or implied, as to any matter including, but not
limited to warranty of fitness of purpose, or merchantability, or
results obtained from use of this software.
======================================================================*/

/* copyright (C) Roni Rosenfeld, 1990, 1991, 1992, 1993 */
/* Edited by Kristie Seymore, 4/16/97 */
/* Editied further by Philip Clarkson April 1997, in order to provide
   consistency with the rest of the toolkit */

/* interpolate_v2:
   Find maximum-likelihood weights for interpolating several probabilistic
models, where the models are described by their output on a common set of
items (.fprobs file), previously derived from a common text file.
   "-test_last nnn" means that the last nnn items of each model will be used
for testing.  "-test_all" means _all_ items will be used for testing.
"-test_first nnn" means that the first nnn items of each model will be used
for testing and the last nnn items of each model will be used for training.
"-test_cv" means to run in cross-validation mode, where each half of the
test items will be used for training and testing, and the resulting PPs
will be combined to give an overall PP value.
   If a partition of the items is induced via a "tags" mapping, separate
weights will be calculated for each tag.

In "test" mode, run thru the main loop only once, to compute the test PP.
   The initial lambdas may be read from a file.  For non-test-mode, they
default to 1/#-of-models.

  Roni Rosenfeld, 3/1/93
  Adapted from the version I wrote for my "lm" software.  */

#include <stdio.h>
#include <strings.h>
#include <math.h>
#include <stdlib.h>
#include "rr_libs/general.h"
#include "pc_libs/pc_general.h"
#include "toolkit.h"
#define  ITEM_T  float
#define  ITEM_FORMAT "%f"
#define  MCAPTION 20

/* update command line argument sequence */
void updateArgs( int *pargc, char **argv, int rm_cnt )
{
  int i ;             

  /* update the argument count */
  (*pargc)-- ;

  /* update the command line */
  for( i = rm_cnt ; i < *pargc ; i++ ) argv[i] = argv[i+1] ;
}       

void eval(double *sum_logprobs, double **fractions, int *tag_of, int *n_in_tag,
	  double *prob_components, double **lambdas, ITEM_T **model_probs,
	  int ntags, int from_item, int to_item, int nmodels, char **captions, 
	  double *p_new_pp, int iter_no, double old_pp, int verbosity, 
	  FILE *probs_fp) {
  int     itag, iitem, tag, imodel;
  double  total_prob, total_logprobs, new_pp;

  for (itag=0; itag<ntags; itag++) {
     sum_logprobs[itag] = 0.0;
     for (imodel=0; imodel<nmodels; imodel++) {
        fractions[imodel][itag] = 0.0;
     }
  }
  for (iitem=from_item; iitem<=to_item; iitem++) {
    tag = tag_of[iitem];
    total_prob = 0.0;
    for (imodel=0; imodel<nmodels; imodel++) {
      prob_components[imodel] =
	lambdas[imodel][tag] * model_probs[imodel][iitem];
      total_prob += prob_components[imodel];
    }
    for (imodel=0; imodel<nmodels; imodel++) {
      fractions[imodel][tag] += prob_components[imodel] / total_prob;
    }
    sum_logprobs[tag] += log(total_prob);
    pc_message(verbosity,3,"    item #%d (tag %d): ",iitem,tag);
    pc_message(verbosity,4,"\n        probs:  ");
    for (imodel=0; imodel<nmodels; imodel++) {
      pc_message(verbosity,4,"%.4f ",model_probs[imodel][iitem]);
    }
    pc_message(verbosity,4,"\n        comps:  ");
    for (imodel=0; imodel<nmodels; imodel++) {
      pc_message(verbosity,4,"%.4f ",prob_components[imodel]);
    }
    pc_message(verbosity,4,"\n        fracts: ");
    for (imodel=0; imodel<nmodels; imodel++) {
      pc_message(verbosity,4,"%.4f ",fractions[imodel][tag]);
    }
    pc_message(verbosity,4,"\n    ");

    pc_message(verbosity,3,
	       "total_prob=%.4f  logprob=%.3f  sum_logprob[%d]=%.3f\n",
	       total_prob, log(total_prob), tag, sum_logprobs[tag]);

    if (probs_fp) {
      fprintf(probs_fp,"%g\n",total_prob);
    }
  }
  pc_message(verbosity,2,"\n");
  total_logprobs = 0.0;
  for (itag=0; itag<ntags; itag++) {
    total_logprobs += sum_logprobs[itag];
    pc_message(verbosity,2,"%.*s weights: ",MCAPTION,captions[itag]);
    for (imodel=0; imodel<nmodels; imodel++) {
      pc_message(verbosity,2,"%.3f  ",lambdas[imodel][itag]);
    }
    pc_message(verbosity,2,"(%4d items)",n_in_tag[itag]);
    if (n_in_tag[itag]==0) {
      pc_message(verbosity,2,"\n");
    }
    else {
      pc_message(verbosity,2," --> PP=%f\n",
		 exp( -sum_logprobs[itag] / n_in_tag[itag]));
    }
  }
  new_pp = exp(-total_logprobs/(to_item-from_item+1));
  pc_message(verbosity,2,"\t\t\t=============>  TOTAL PP = %g",new_pp);
  if (iter_no>1) {
    pc_message(verbosity,2," (down %.4f)\n",(1.0-(new_pp/old_pp)));
  }
  else {
    pc_message(verbosity,2,"\n");
  }
  *p_new_pp = new_pp;
}


void main (int argc, char **argv) {

  int use_tags=0; 
  int use_captions=0; 
  int pure_test_mode=0; 
  int first_part = 0;
  int default_lambdas=1; 
  int verbosity=1; 
  int n_test_items=0; 
  int n_train_items=0; 
  int cv=0;
  int Mprobs = 60000; 
  int write_lambdas = 0;
  double stop_ratio = 0.999;

  static char *rname = "interpolate";

  char *tags_filename; 
  char *captions_filename; 
  char *lambdas_filename; 
  char *write_lambdas_filename;
  FILE *tags_fp; 
  FILE *captions_fp; 
  FILE *lambdas_fp; 
  FILE *probs_fp=NULL; 
  FILE *write_lambdas_fp = NULL;

  char   **model_filenames;      /* model_filenames[model]            */
  FILE   **model_fps;            /*       model_fps[model]            */
  Boolean *model_fixed_lambdas;  /*model_fixed_lambdas[model]         */
  ITEM_T **model_probs;          /*           probs[model][item]      */
  int    *tag_of;                /*              tag_of[item]         */
  double *prob_components;       /* prob_components[model]            */
  double **lambdas;              /*         lambdas[model][tag]       */
  double **fractions;            /*       fractions[model][tag]       */
  double *sum_logprobs;          /*           sum_logprobs[tag]       */
  int    *n_train_in_tag;        /*            n_in_tag[tag]          */
  int    *n_test_in_tag;         /*            n_in_tag[tag]          */
  int    nmodels=0; 
  int imodel; 
  int ntags; 
  int itag; 
  int tag; 
  int nitems; 
  int iitem;
  int iter_no;
  int half_point = 0; 
  int iter_num; 
  int first_test_items=0; 
  int second_test_items=0;
  double old_pp=0.0; 
  double new_pp; 
  double test_pp; 
  float dummyf;
  double first_part_pp=0.0; 
  double second_part_pp=0.0; 
  double total_pp; 
  double sum_logprob_1; 
  double sum_logprob_2;
  double total_logprob;
  char   **captions;
  FILE   *fp;
  ITEM_T *pitem;
  int    scanfrc; 
  int nnewitems;
  int temp_test_items;
  char *write_fprobs_filename;
  int i;

  /* Allocate memory for model data */

  model_filenames = (char **) rr_malloc(argc * (sizeof(char *)));
  model_fixed_lambdas = (Boolean *) rr_malloc(argc * (sizeof(Boolean)));

  /* Process command line */

  report_version(&argc,argv);

  if (argc == 1 || pc_flagarg(&argc,argv,"-help")) {

    fprintf(stderr,"Usage : interpolate +[-] model1.fprobs +[-] model2.fprobs ... \n");
    fprintf(stderr,"        [ -test_all | -test_first n | -test_last n | -cv ]\n");
    fprintf(stderr,"        [ -tags .tags ]\n");
    fprintf(stderr,"        [ -captions .captions ]\n");
    fprintf(stderr,"        [ -out_lambdas .lambdas ]\n");
    fprintf(stderr,"        [ -in_lambdas .lambdas ]\n");
    fprintf(stderr,"        [ -stop_ratio 0.999 ]\n");
    fprintf(stderr,"        [ -probs .fprobs ]\n");
    fprintf(stderr,"        [ -max_probs 6000000 ]\n");
    exit(1);

  }

  /* Grab all the model filename */

  i = 0;
  while (i<argc) {

    if (argv[i][0]=='+') {
      model_fixed_lambdas[nmodels] = (argv[i][1]=='-');
      model_filenames[nmodels++] = salloc(argv[i+1]);
      updateArgs( &argc, argv, i+1 ) ;
      updateArgs( &argc, argv, i ) ;
    }
    else {
      i++;
    }
  }

  /* Now process all the other switches */

  verbosity = pc_intarg(&argc,argv,"-verbosity",DEFAULT_VERBOSITY);

  pure_test_mode = pc_flagarg(&argc,argv,"-test_all");

  n_test_items = pc_intarg(&argc,argv,"-test_first",-1);
  if (n_test_items != -1) {
    first_part = 1;
  }

  temp_test_items = pc_intarg(&argc,argv,"-test_last",-1);
  if (n_test_items != -1 && temp_test_items != -1) {
    quit(-1,"Error : Cannot specify both -test_last and -test_first options.\n");
  }

  if (temp_test_items != -1) {
    n_test_items = temp_test_items;
    first_part = 0;
  }

  if (n_test_items == -1) {
    n_test_items = 0;
  }

  cv = pc_flagarg(&argc,argv,"-cv");

  tags_filename = salloc(pc_stringarg(&argc,argv,"-tags",""));
  if (strcmp(tags_filename,"")) {
    use_tags = 1;
  }

  captions_filename = salloc(pc_stringarg(&argc,argv,"-captions",""));
  if (strcmp(captions_filename,"")) {
    use_captions = 1;
  }

  if (use_captions && !use_tags) {
    pc_message(verbosity,1,"Warning - captions file specified, but no tags file.\n");
  }

  lambdas_filename = salloc(pc_stringarg(&argc,argv,"-in_lambdas",""));
  if (strcmp(lambdas_filename,"")) {
    default_lambdas = 0;
  }
  else {
    default_lambdas = 1;
  }

  write_lambdas_filename = salloc(pc_stringarg(&argc,argv,"-out_lambdas",""));
  if (strcmp(write_lambdas_filename,"")) {
    write_lambdas = 1;
  }

  stop_ratio = pc_doublearg(&argc,argv,"-stop_ratio",0.999);

  write_fprobs_filename = salloc(pc_stringarg(&argc,argv,"-probs",""));
  if (strcmp(write_fprobs_filename,"")) {
    if (n_test_items > 0 || cv) {
      probs_fp = rr_oopen(write_fprobs_filename);
    }
    else {
      pc_message(verbosity,2,"Warning : -write option ignored, as none of the data is used for testing.\n");
    }
  }

  Mprobs = pc_intarg(&argc,argv,"-max_probs",6000000);

  pc_report_unk_args(&argc,argv,verbosity);

  if (nmodels==0) quit(-1,"%s: no models specified\n",rname);
  if (pure_test_mode && default_lambdas)
    quit(-1,"%s: in pure test mode, initial lambdas must be supplied\n",rname);
  if (stop_ratio<0.0 || stop_ratio >1.0)
     quit(-1,"%s: illegal stop_ratio (%f) - must be a fraction\n",
              rname,stop_ratio);

  if (cv && pure_test_mode) {
    quit(-1,"%s : Error - cannot specify both -cv and -test_all.\n",rname);
  }

  if (cv && n_test_items != 0) {
    quit(-1,"%s : Error - cannot specify both -cv and -test_first or -test_last.\n",
	 rname);
  }

  if (pure_test_mode && n_test_items != 0) {
    quit(-1,"%s : Error - cannot specify both -test_all and -test_first or -test_last.\n",
	 rname);
  }


  model_fps       = (FILE   **) rr_malloc(nmodels * sizeof(FILE *));
  model_probs     = (ITEM_T **) rr_malloc(nmodels * sizeof(ITEM_T *));
  lambdas         = (double **) rr_malloc(nmodels * sizeof(double *));
  fractions       = (double **) rr_malloc(nmodels * sizeof(double *));
  prob_components = (double *)  rr_malloc(nmodels * sizeof(double));

  nitems = -1;

  pc_message(verbosity,2,"%s : Reading the probability streams....",rname);
  fflush(stderr);
  
  for (imodel=0; imodel<nmodels; imodel++) {
     model_fps[imodel] = rr_iopen(model_filenames[imodel]);
     model_probs[imodel] = (ITEM_T *) rr_malloc((Mprobs+1)*sizeof(ITEM_T));

     /* read in the models probabilities */
     fp=model_fps[imodel];
     pitem=model_probs[imodel];
     nnewitems = 0;
     for (iitem=0; iitem<Mprobs+1; iitem++) {
        if ((scanfrc=fscanf(fp,ITEM_FORMAT,pitem++)) != 1) break;
        nnewitems++;
     }
     if (nnewitems>Mprobs) quit(-1,
        "%s: more than %d probs on %s\n",rname,Mprobs,model_filenames[imodel]);
     if (imodel==0) nitems = nnewitems;
     else if (nnewitems != nitems)
        quit(-1,"%s: model '%s' has %d probs, but model '%s' has %d probs\n",
          rname,model_filenames[0],nitems,model_filenames[imodel],nnewitems);

     fclose(model_fps[imodel]);
  }

  pc_message(verbosity,2,"Done.\n");
  fflush(stderr);

  if (n_test_items >= nitems)
     quit(-1,"%s: \"-test_last %d\" was specified, but there are only %d items\n",
              rname, n_test_items, nitems);
  if (pure_test_mode) n_test_items=nitems;

  if (cv) half_point = (int) (nitems/2);

  if (write_lambdas == 1) {
      write_lambdas_fp = rr_oopen(write_lambdas_filename);
  }

  for (iter_num = 1; iter_num <= 2; iter_num++) {

     if (cv && iter_num == 1) {
        n_test_items = nitems - half_point;
        first_part = 0;
     }
     if (cv && iter_num == 2) {
        n_test_items = half_point;
        first_part = 1;
     }

     n_train_items = nitems - n_test_items;
     if (n_train_items>0 && n_test_items>0) {
       if (first_part) {
	 pc_message(verbosity,2,
	 "%s: %d models will be interpolated using the last %d data items\n",
		    rname, nmodels, n_train_items);
	 pc_message(verbosity,2,
	 "    The first %d data items will be used for testing\n",
		    n_test_items);
       }
       else {
	 pc_message(verbosity,2,
         "%s: %d models will be interpolated using the first %d data items\n",
		 rname, nmodels, n_train_items);
	 pc_message(verbosity,2,
	 "    The last %d data items will be used for testing\n",
	 n_test_items);
       }
     }
     else {
       if (n_train_items>0) {
	 pc_message(verbosity,2,
		    "%s: %d models will be interpolated using %d data items\n",
		    rname, nmodels, n_train_items);
       }
       else {
	 if (n_test_items>0) {
	   pc_message(verbosity,2,
		      "%s: %d models will be tested using %d data items\n",
		      rname, nmodels, n_test_items);
	 }

	 else {
	   if (cv) {
	     pc_message(verbosity,2,
	     "%s: %d models will be tested using cross validation\n",
			rname, nmodels);
	   }
	 }
       }
     }
     if (!default_lambdas)
       pc_message(verbosity,2,"%s: %sweights will be read from \"%s\"\n",
	  rname,(n_train_items ? "initial " : ""),
	  (strcmp(lambdas_filename,"-")==0) ? "stdin" : lambdas_filename);
     for (imodel=0; imodel<nmodels; imodel++) {
       if (model_fixed_lambdas[imodel])
	 pc_message(verbosity,2,"%s: weights of '%s' will be fixed\n",
		    rname, model_filenames[imodel]);
     }

     /* read the tags file, or set all tags to 1 */
     tag_of = (int *) rr_malloc(nitems * sizeof(int));
     if (use_tags) {
        int maxtag = -1;
        tags_fp = rr_iopen(tags_filename);
	pc_message(verbosity,2,"tags will be taken from \"%s\"\n",
		   tags_filename);
        for (iitem=0; iitem<nitems; iitem++) {
           if (fscanf(tags_fp,"%d",&tag_of[iitem]) != 1)
              quit(-1,"%s: problem reading %dth tag from %s\n",
                       rname, iitem, tags_filename);
           if ((tag_of[iitem]<0))
              quit(-1,"%s: illegal tag (%d)\n", rname, tag_of[iitem]);
           if (tag_of[iitem]>maxtag) maxtag = tag_of[iitem];
        }
        if (fscanf(tags_fp,"%d",&tag_of[iitem]) != EOF)
           quit(-1,"%s: %s contains more than %d items\n",
                 rname, tags_filename, nitems);
        ntags = maxtag+1;
	pc_message(verbosity,2,"%s: data is partitioned into %d tags\n", 
		   rname, ntags);
     }
     else {
        ntags = 1;
        for (iitem=0; iitem<nitems; iitem++)  tag_of[iitem] = 0;
	pc_message(verbosity,2,"%s: a single tag is used for all the data\n",
		   rname);
     }

     /* fill in the CAPTIONS strings */

     captions = (char **) rr_malloc(ntags*sizeof(char *));
     for (tag=0; tag<ntags; tag++)
         captions[tag] = (char *) rr_malloc((MCAPTION+1)*sizeof(char));
     if (use_captions) {
        char line[81];
        int len;
        captions_fp = rr_iopen(captions_filename);
	pc_message(verbosity,2,"captions will be taken from \"%s\"\n",
		   captions_filename);

        for (tag=0; tag<ntags; tag++) {
	  if (fgets(line,80,captions_fp) == NULL) {
	    quit(-1,"Error reading from captions file.\n");
	  }
              len=strlen(line); line[len-1]='\0'; len--; /* remove the '\n' */

	      sprintf(captions[tag],"%.*s",MCAPTION,line);
	      strncat(captions[tag],"                    ",MCAPTION-len);
        } 
     }
     else {
       for (tag=0; tag<ntags; tag++) {
	 sprintf(captions[tag],"          TAG %d",tag);
	 strncat(captions[tag],"                    ",
		 MCAPTION-strlen(captions[tag]));
       }
     }


     /* Allocate rest of arrays */
     n_train_in_tag  = (int *) rr_calloc(ntags,sizeof(int));
     n_test_in_tag  = (int *) rr_calloc(ntags,sizeof(int));
     for (iitem=0; iitem<n_train_items; iitem++) {
       n_train_in_tag[tag_of[iitem]]++;
     }
     for (iitem=n_train_items; iitem<nitems; iitem++) {
       n_test_in_tag[tag_of[iitem]]++;
     }
     sum_logprobs = (double *) rr_malloc(ntags * sizeof(double));
     for (imodel=0; imodel<nmodels; imodel++) {
        lambdas[imodel]   = (double *) rr_malloc(ntags*sizeof(double));
        fractions[imodel] = (double *) rr_malloc(ntags*sizeof(double));
     }

     /* Initialize the weights (lambdas) */
     if (default_lambdas) {
        for (itag=0; itag<ntags; itag++)
           for (imodel=0; imodel<nmodels; imodel++)
             lambdas[imodel][itag] = 1.0 / nmodels;
     }
     else {
        lambdas_fp = rr_iopen(lambdas_filename);
        if (strcmp(lambdas_filename,"-")==0)
          fprintf(stderr,"Enter initial weights, by tag order\n");
        for (itag=0; itag<ntags; itag++) {
           double sum_lambdas = 0.0;
           for (imodel=0; imodel<nmodels; imodel++) {
              if (fscanf(lambdas_fp,"%lf",&lambdas[imodel][itag])!=1)
                 quit(-1,"%s: problems reading from '%s'\n",
		      rname,lambdas_filename);
              sum_lambdas += lambdas[imodel][itag];
           }
           if (fabs(1.0-sum_lambdas) > 1e-8)
              quit(-1,"%s: weights for tag #%d sum to %g, not to 1\n",
                       rname, itag, sum_lambdas);
        }
        if (fscanf(lambdas_fp,"%f",&dummyf) != EOF)
           quit(-1,"%s: too many numbers found in '%s'\n", 
		rname, lambdas_filename);
        rr_iclose(lambdas_fp);
     }

     /* TRAINING: iterate the EM step */
     new_pp = 10e98;
     iter_no = 1;
     while (n_train_items>0 &&
            (iter_no==1 || (new_pp/old_pp < stop_ratio))) {
        old_pp = new_pp;

        /* re-estimate lambdas before all but the first iteration */
        if (iter_no > 1) {
           for (itag=0; itag<ntags; itag++) {
              double total_nonfixed_lambdas = 0.0;
              double total_nonfixed_fractions = 0.0;
              if (n_train_in_tag[itag] <= 0) continue;
              for (imodel=0; imodel<nmodels; imodel++) {
                 if (!model_fixed_lambdas[imodel]) {
                    total_nonfixed_lambdas += lambdas[imodel][itag];
                    total_nonfixed_fractions += fractions[imodel][itag];
                 }
              }
              for (imodel=0; imodel<nmodels; imodel++) {
                 if (!model_fixed_lambdas[imodel]) {
                    lambdas[imodel][itag] =
                             (fractions[imodel][itag] / n_train_in_tag[itag]);
		    /* correct s.t. the lambdas sum to 
		       'total_nonfixed_lambdas' */
                    lambdas[imodel][itag] *=
                        (total_nonfixed_lambdas /
                              (total_nonfixed_fractions/n_train_in_tag[itag]));
                 }
              }
           }
        }
        if (first_part) {     /* Train on last part and test on first part */
           eval(sum_logprobs, fractions, tag_of, n_train_in_tag,
                prob_components, lambdas, model_probs,
                ntags, n_test_items, nitems-1, nmodels, captions, &new_pp,
                iter_no, old_pp, verbosity, NULL);
           iter_no++;
        }
        else {  /* Train on first part and test on last part */
           eval(sum_logprobs, fractions, tag_of, n_train_in_tag,
                prob_components, lambdas, model_probs,
                ntags, 0, n_train_items-1, nmodels, captions, &new_pp,
                iter_no, old_pp, verbosity, NULL);
           iter_no++;
        }
     } /* e.o. while loop */
     /* (we avoid reestimating lambda after the last iteration, so that
        the PP we reported be accurate) */

     /* If training was done, write the weights to stdout as well */
     if (n_train_items>0) {
        fprintf(stderr,"\n");
        for (itag=0; itag<ntags; itag++) {
           for (imodel=0; imodel<nmodels; imodel++)
              printf("%12.10f ",lambdas[imodel][itag]);
           printf("\n");
        }
        fflush(stdout);
     }
     if (write_lambdas == 1) {
        for (itag=0; itag<ntags; itag++) {
           for (imodel=0; imodel<nmodels; imodel++)
              fprintf(write_lambdas_fp, "%s   %12.10f\n",
		      model_filenames[imodel],lambdas[imodel][itag]);
           printf("\n");
        }
     }

     if (n_test_items>0) {
        fprintf(stderr,"\nNOW TESTING ...\n");
        if (first_part) { /* Train on last part and test on first part */
           eval(sum_logprobs, fractions, tag_of, n_test_in_tag,
                prob_components, lambdas, model_probs,
                ntags, 0, n_test_items-1, nmodels,captions, &test_pp,
                1, 0.0, verbosity, probs_fp);
           fprintf(stderr,"\n");

        }
        else {                  /* Train on first part and test on last part */
           eval(sum_logprobs, fractions, tag_of, n_test_in_tag,
                prob_components, lambdas, model_probs,
                ntags, n_train_items, nitems-1, nmodels,captions, &test_pp,
                1, 0.0, verbosity, probs_fp);
           fprintf(stderr,"\n");
        }
     }


     if (iter_num == 1) {
        first_part_pp = test_pp;
        first_test_items = n_test_items;
     }
     else if (iter_num == 2) {
        second_part_pp = test_pp;
        second_test_items = n_test_items;
     }

     /* Free all memory allocated in the loop */
     free (tag_of);
     for (tag=0; tag<ntags; tag++)
         free (captions[tag]);
     free (captions);
     free (n_train_in_tag);
     free (n_test_in_tag);
     free (sum_logprobs);
     for (imodel=0; imodel<nmodels; imodel++) {
        free (lambdas[imodel]);
        free (fractions[imodel]);
     }

     if (!cv) { break; }

   }

   if (write_lambdas == 1) {
      fclose(write_lambdas_fp);
   }

   /* In cross-validation mode, calc total PP */
   if (cv) {
      sum_logprob_1 = -log(first_part_pp) * first_test_items;
      sum_logprob_2 = -log(second_part_pp) * second_test_items;
      total_logprob = sum_logprob_1 + sum_logprob_2;
      total_pp = exp(-total_logprob/nitems);

      fprintf(stderr, "Two-way cross validation: \n");
      fprintf(stderr, "     First half PP = %f\n", second_part_pp);
      fprintf(stderr, "     Second half PP = %f\n", first_part_pp);
      fprintf(stderr, "     =====> Total PP = %f\n", total_pp);

   }

   if (n_test_items>0) exit((int) test_pp);
}


