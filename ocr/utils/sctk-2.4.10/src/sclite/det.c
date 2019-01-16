#include "sctk.h"

static void Compute_ROC (double *true_scores, int num_true, double *false_scores, int num_false, double **Pdet);
static double ppndf (double p);

/* return true if all the paths in the score structure have hyp confidences */
int hyp_confidences_available(SCORES *scor){
    int g, p;
    int noconf = 0;

    /* search the score for pabiuld the target and non-target arrays */
    for (g=0; g<scor->num_grp; g++)
      for (p=0; p<scor->grp[g].num_path; p++) 
	if (! BF_isSET(scor->grp[g].path[p]->attrib,PA_HYP_CONF))
	  noconf ++;
    return(noconf == 0 ? 1 : 0);
}

int make_SCORES_DET_curve(SCORES *scor[], int nscor, char *outroot, int feedback, char *test_name){
    double *targ=(double *)0, *non_targ=(double *)0, **det = (double **)0;
    int n_targ = 0 , n_non_targ = 0;
    int max_targ = 10000, max_non_targ = 10000;
    int g, p, w, s;
    GRP *gp;
    PATH *pp;
    FILE *fp, *fpd;
    int rtn;
    char *gnutics = 
      "(\"0.1\" -3.08, \"0.5\" -2.57, \"2\" -2.05, \"5\" -1.64, "
      "\"10\" -1.28, \"20\" -0.84, \"30\" -0.52, \"40\" -0.25, "
      "\"50\" 0.0, \"60\" 0.25, \"70\" 0.52, \"80\" 0.84, \"90\" 1.28, "
      "\"95\" 1.64, \"98\" 2.05, \"99.5\" 2.57, \"99.9\" 3.08)";

    if (feedback >= 1)
      if (hyp_confidences_available(scor[0]))
        printf("    Writing DET Curve to '%s.det.[plt,dat]'\n",outroot);
      else {
        printf("    Skipping DET Curve, no confidence scores supplied.\n");
	return 0;
      }
    
    /* output the DET Curve GNUPLUT command file */
    if ((fpd = fopen(rsprintf("%s.det.plt",outroot),"w")) == (FILE *)0){
      fprintf(stderr,"Error: Unable to open DET Curve GNUPLOT file '%s' for"
	      " writing.",rsprintf("%s.det.plt",outroot));
      goto ERROR;
    }
    fprintf(fpd,"## GNUPLOT command file\n");
    fprintf(fpd,"set data style lines\n");
    fprintf(fpd,"set size 0.78, 1.0\n");
    fprintf(fpd,"set noxtics\n");
    fprintf(fpd,"set noytics\n");
    if (nscor == 1){
      fprintf(fpd,"set title 'DET plot for %s'\n",outroot);
      fprintf(fpd,"set nokey\n");
    } else {
      if (*test_name != (char)0)
	fprintf(fpd,"set title 'DET plot for %s Test'\n",test_name);
      else
	fprintf(fpd,"set title 'DET plot'\n");
    }
    fprintf(fpd,"set ylabel \"Correct Words Removed (in %%)\"\n");
    fprintf(fpd,"set xlabel \"Incorrect Words Retained (in %%)\"\n");
    fprintf(fpd,"set grid\n"); 
    fprintf(fpd,"set ytics %s\n",gnutics);
    fprintf(fpd,"set xtics %s\n",gnutics);
    fprintf(fpd,"plot [-3.290527:3.290527] [-3.290527:3.290527] \\\n");

    /* biuld the target and non-target arrays */
    for (s=0; s<nscor; s++){
      alloc_singZ(targ,max_targ,double,0.0);
      alloc_singZ(non_targ,max_non_targ,double,0.0);

      for (g=0; g<scor[s]->num_grp; g++){
        gp = &(scor[s]->grp[g]);
	for (p=0; p<gp->num_path; p++) {
	    pp = gp->path[p];
	    for (w=0; w<pp->num; w++) {
	        if (pp->pset[w].eval == P_CORR &&
		    *(((WORD*)pp->pset[w].b_ptr)->value) != (TEXT)'\0'){
		    if (n_targ >= max_targ) 
		        expand_singarr(targ,n_targ,max_targ,2,double);
		    targ[n_targ++] = ((WORD *)(pp->pset[w].b_ptr))->conf;
		} else if (pp->pset[w].eval == P_SUB || 
			   pp->pset[w].eval == P_INS){
		    if (n_non_targ >= max_non_targ) 
		        expand_singarr(non_targ,n_non_targ,
				       max_non_targ,2,double);
		    non_targ[n_non_targ++] =
		      ((WORD *)(pp->pset[w].b_ptr))->conf;
		}			  
	    }
	}
      }
      
      /* allocate the DET table */
      alloc_2dimZ(det,(n_targ+n_non_targ),2,double,0.0);
      /* calculate the DET Curve */
      Compute_ROC (targ, n_targ, non_targ, n_non_targ, det);

      /* output the DET Curve data points */
      if ((fp = fopen(rsprintf("%s.det.dat.%02d",outroot,s),"w"))== (FILE *)0){
        fprintf(stderr,"Error: Unable to open DET Curve data file '%s' for"
		" writing.",rsprintf("%s.det.dat.%02d",outroot,s));
	goto ERROR;
      }
      for (w=0; w<n_targ+n_non_targ-1; w++)
        fprintf(fp,"%f %f\n",det[w][0],det[w][1]);
      fclose(fp);

      /**************/
      fprintf(fpd," \"%s.det.dat.%02d\" using 2:1 title \"%s\" with lines %d",
	      outroot,s,scor[s]->title,s+1);
      if (s < nscor-1) fprintf(fpd,", \\");
      fprintf(fpd,"\n");

      if (det != (double**)0) free_2dimarr(det,(n_targ+n_non_targ),double);
      if (targ != (double *)0)     free_singarr(targ,double);
      if (non_targ != (double *)0) free_singarr(non_targ,double);
    }
    
    fprintf(fpd,"set ytics\n");
    fprintf(fpd,"set xtics\n");
    fprintf(fpd,"set size 1.0, 1.0\n");
    fprintf(fpd,"set key\n");
    
    fclose(fpd);

    rtn = 0;
    goto END;

  ERROR: 
    rtn = 1;
  END:
    if (targ != (double *)0)     free_singarr(targ,double);
    if (non_targ != (double *)0) free_singarr(non_targ,double);
    if (det != (double**)0)      free_2dimarr(det,(n_targ+n_non_targ),double);
    return(rtn);
}


/*

The Compute_ROC function that I sent to you has a slight error -- it
wasn't computing the last point on the Pmiss/Pfa probability trade-off
curve.  This is corrected in the version below.  Also, we've been
thinking about what to call these plots that makes better intuitive
sense.  I've sort of decided on "DET" or DErT" curves, which stands
for "Detection Error Trade-off" curves.  This really does express what
we are plotting, namely the Trade-off between the two kinds of Errors
(Pmiss/Pfa) for Detection tasks.

::::::::::::::
Compute_ROC.m -- corrected
::::::::::::::
function Pdet = Compute_ROC (true_scores, false_scores)
%function Pdet = Compute_ROC (true_scores, false_scores)
%
%  Compute_ROC computes the miss/false_alarm probability trade-off
%  for a set of scores for the true/false detection hypothesis.
%
%  true_scores (false_scores) are detection output scores for a set of
%  detection trials, given that the target hypothesis is true (false).
%  (It is assumed by convention that the more positive the score, the
%  more likely is the target hypothesis.)
%
%  Pdet is a two-column matrix containing the detection probability
%  trade-off.  The first column contains the miss probabilities and
%  the second column contains the corresponding false alarm
%  probabilities.
%
%  See also Make_pROC, Plot_pROC, and Set_DCF.

*/


#define PMIN    0.0005
#define PMAX    0.5
#define SMAX  9E99

#define Cmiss    1.0
#define Cfa      1.0
#define Ptrue    0.05
#define Pfalse   0.05

static void Compute_ROC (double *true_scores, int num_true, double *false_scores, int num_false, double **Pdet){


    int ntrue;
    int nfalse;
    int npts;
    
    qsort ((char *) true_scores, num_true, sizeof (double),
	   qsort_double_compare);
    true_scores[num_true] = SMAX;

    qsort ((char *) false_scores, num_false, sizeof (double),
	   qsort_double_compare);
    false_scores[num_false] = SMAX;

    ntrue = 1;
    nfalse = 1;
    npts = 0;
    Pdet[npts][0] = ppndf (PMIN);
    Pdet[npts][1] = ppndf (PMAX);
    
    while ((ntrue < num_true) || (nfalse < num_false)) {
	/* printf ("ntrue = %d, nfalse = %d, npts = %d, trsc = %lf, fasc =%lf\n",
	   ntrue, nfalse, npts, true_scores[ntrue], false_scores[nfalse]); */
	
      /* bug fix JGF, the previous expression assumed an highest score would
	 ALLWAYS be a TRUE score */
	if ((true_scores[ntrue] <= false_scores[nfalse]) &&
	    (ntrue < num_true))
	    ntrue = ntrue+1;
	else
	    nfalse = nfalse+1;
/*	end */
	npts = npts+1;
	Pdet[npts][0] =   ((double) (ntrue-1) / (double) num_true);
	Pdet[npts][1] =   ((double) (num_false - (nfalse-1)) / (double) num_false);
	
        Pdet[npts][0] = ppndf (Pdet[npts][0]);
        Pdet[npts][1] = ppndf (Pdet[npts][1]);
	
	/* printf ("point %f %f\n", Pdet[npts][0], Pdet[npts][1]); */
    }
}


/*

Convert lines with ROC coordinates to normal deviate scale.

Ignore lines beginning with "#" (9/26/94)

*/

#ifndef abs
#define abs(x)  (x < 0  ?  -x : x)
#endif

#define SPLIT      0.42

#define A0      2.5066282388
#define A1    -18.6150006252
#define A2     41.3911977353
#define A3    -25.4410604963
#define B1     -8.4735109309
#define B2     23.0833674374
#define B3    -21.0622410182
#define B4      3.1308290983
#define C0     -2.7871893113
#define C1     -2.2979647913
#define C2      4.8501412713
#define C3      2.3212127685
#define D1      3.5438892476
#define D2      1.6370678189

#define LL      140

#define eps     2.2204e-16

static double ppndf (double p){
    double q;  
    double r;
    double retval;


    /* printf ("p = %f\n", p); */

    if (p >= 1.0)
        p = 1 - eps;
    if (p <= 0.0)
        p = eps;

    q = p - 0.5;
    if (abs(q) <= SPLIT){
        r = q * q;
        retval = q * (((A3 * r + A2) * r + A1) * r + A0) /
	  ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.0);
    } else  {
        r = (q > 0.0  ?  1.0 - p : p);
	if (r <= 0.0){
	    fprintf (stderr,"Warning: Found r = %f\n", r);
	    return(0.0);
	}
    
	r = sqrt ((-1.0) * log (r));
	
	retval = (((C3 * r + C2) * r + C1) * r + C0) /
	    ((D2 * r + D1) * r + 1.0);
	if (q < 0)
	    retval *= -1.0;
    }

    return (retval);

}

/****************************************************************************
 *
 *   The functions below create the binned confidence graphs
 *
 *
 *
 ****************************************************************************/

#define H_TO           0
#define H_FROM         1
#define H_PCT_CORR     2
#define H_EXP_PCT_CORR 3
#define H_N_CORR       4
#define H_N_INCORR     5

#define NUM_H_ELEM     6

static void binned_confidences(SCORES *scor, double from, double to, int nbin, double **hist);

int make_confidence_histogram(SCORES *scor, char *outroot, int feedback){
    double **hist = (double **)0;
    FILE *fp_dat1, *fp_plt;
    int rtn, b, nbin = 100;

    if (feedback >= 1)
      if (hyp_confidences_available(scor))
        printf("    Writing Confidence Histogram '%s.hist.[plt,dat]'\n",
	       outroot);
      else {
        printf("    Skipping Confidence Histogram, no confidence scores supplied.\n");
	return 0;
      }
    
    /* allocate memory */
    alloc_2dimZ(hist,nbin,NUM_H_ELEM,double,0.0);

    binned_confidences(scor, 0.0, 1.0, nbin, hist);

    /* output the Binned histogram data points */
    if ((fp_dat1 = fopen(rsprintf("%s.hist.dat",outroot),"w")) == (FILE *)0){
        fprintf(stderr,"Error: Writing Binned Histogram file '%s' for"
		" writing.",rsprintf("%s.bhist.dat1",outroot));
	goto ERROR;
    }
    for (b=0; b<nbin; b++){
        fprintf(fp_dat1,"%f %f %f %f\n",hist[b][H_FROM],
		hist[b][H_N_CORR]+hist[b][H_N_INCORR],
		hist[b][H_N_CORR],hist[b][H_N_INCORR]);
        fprintf(fp_dat1,"%f %f %f %f\n",hist[b][H_TO],
		hist[b][H_N_CORR]+hist[b][H_N_INCORR],
		hist[b][H_N_CORR],hist[b][H_N_INCORR]);
    }
    fclose(fp_dat1);

    /* output the Binned histogram GNUPLUT command file */
    if ((fp_plt = fopen(rsprintf("%s.hist.plt",outroot),"w")) == (FILE *)0){
        fprintf(stderr,"Error: Unable to open Binned Histogram"
		"GNUPLOT file '%s' for writing.",
		rsprintf("%s.bhist.plt",outroot));
	goto ERROR;
    }

    fprintf(fp_plt,"set samples 1000\n");
    fprintf(fp_plt,"set xrange [0.000000:1.000000]\n");
    fprintf(fp_plt,"set autoscale y\n");
    fprintf(fp_plt,"set size 0.78, 1.0\n");
    fprintf(fp_plt,"set nogrid\n");
    fprintf(fp_plt,"set ylabel 'Counts'\n");
    fprintf(fp_plt,"set xlabel 'Confidence Measure'\n");
    fprintf(fp_plt,"set title  'Confidence scores for %s'\n",outroot);
    fprintf(fp_plt,
	    "plot '%s.hist.dat' using 1:2 '%%f%%f' title 'All Conf.'"
	    " with lines, \\\n"
	    "     '%s.hist.dat' using 1:2 '%%f%%*s%%f' title 'Correct Conf.'"
	    " with lines, \\\n"
	    "     '%s.hist.dat' using 1:2 '%%f%%*s%%*s%%f' title 'Incorrect"
	    " Conf.' with lines\n",outroot,outroot,outroot); 
    fprintf(fp_plt,"set size 1.0, 1.0\n");
    
    fclose(fp_plt);

    rtn = 0;
    goto END;

  ERROR: 
    rtn = 1;
  END:
    if (hist != (double**)0)      free_2dimarr(hist,nbin,double);
    return(rtn);
}

int make_binned_confidence(SCORES *scor, char *outroot, int feedback){
    double **hist = (double **)0;
    FILE *fp_dat1, *fp_plt;
    int rtn, b, nbin = 10;

    if (feedback >= 1)
      if (hyp_confidences_available(scor))
        printf("    Writing Binned Histogram '%s.bhist.[plt,dat1]'\n",
	       outroot);
      else {
        printf("    Skipping Binned Histogram, no confidence scores supplied.\n");
	return 0;
      }

    /* allocate memory */
    alloc_2dimZ(hist,nbin,NUM_H_ELEM,double,0.0);

    binned_confidences(scor, 0.0, 1.0, nbin, hist);

    /* output the Binned histogram data points */
    if ((fp_dat1 = fopen(rsprintf("%s.bhist.dat1",outroot),"w")) == (FILE *)0){
        fprintf(stderr,"Error: Writing Binned Histogram file '%s' for"
		" writing.",rsprintf("%s.bhist.dat1",outroot));
	goto ERROR;
    }
    for (b=0; b<nbin; b++){
        fprintf(fp_dat1,"%.4f %f 1 1 0.1\n",(b*0.1) + 0.05,hist[b][H_PCT_CORR]);
    }
    fclose(fp_dat1);

    /* output the Binned histogram GNUPLUT command file */
    if ((fp_plt = fopen(rsprintf("%s.bhist.plt",outroot),"w")) == (FILE *)0){
        fprintf(stderr,"Error: Unable to open Binned Histogram"
		"GNUPLOT file '%s' for writing.",
		rsprintf("%s.bhist.plt",outroot));
	goto ERROR;
    }
    fprintf(fp_plt,"## GNUPLOT command file\n");
    fprintf(fp_plt,"set samples 1000\n");
    fprintf(fp_plt,"set key 30.000000,90.000000\n");
    fprintf(fp_plt,"set xrange [0:1]\n");
    fprintf(fp_plt,"set yrange [0:100]\n");
    fprintf(fp_plt,"set nogrid\n");
    fprintf(fp_plt,"set ylabel '%% Hypothesis Correct'\n");
    fprintf(fp_plt,"set xlabel 'Confidence Scores'\n");
    fprintf(fp_plt,"set title  'Binned Confidence scores for %s'\n",outroot);
    fprintf(fp_plt,"set size 0.78,1\n");
    fprintf(fp_plt,"set nolabel\n");
    fprintf(fp_plt,"plot '%s.bhist.dat1'  title 'True' with boxes,\\\n",
	    outroot);
    fprintf(fp_plt,"     x*100  title 'Predicted' with lines\n");
    fprintf(fp_plt,"set size 1.0, 1.0\n");

    fclose(fp_plt);

    rtn = 0;
    goto END;

  ERROR: 
    rtn = 1;
  END:
    if (hist != (double**)0)      free_2dimarr(hist,nbin,double);
    return(rtn);
}


int make_scaled_binned_confidence(SCORES *scor, char *outroot, int bins, int feedback){
  int curlen = 1000, nword=0;
  int *eval, i, p, w, *sort, sum_corr = 0, sum_err = 0, rtn, b, l;
  double *conf;
  FILE *fp_dat, *fp_plt;

  if (feedback >= 1)
    if (hyp_confidences_available(scor))
      printf("    Writing Scaled Binned Histogram '%s.sbhist.[plt,dat]'\n",
	     outroot);
    else{
      printf("    Skipping Scaled Binned Histogram, no confidence scores supplied.\n");
      return 0;
    }

  /* algo, :
     1: make a list of all confidences, with either corr or incorr flags
     2: sort the confidences 
     3: divide it up into N chunks to make the histogram
     */
  /* step 1 */
  alloc_singZ(conf,curlen,double,0.0);
  alloc_singZ(eval,curlen,int,0);

  for (i=0; i<scor->num_grp; i++)
    for (p=0; p<scor->grp[i].num_path; p++){
      PATH *path = scor->grp[i].path[p];
      for (w=0; w<path->num; w++)
	if ((path->pset[w].eval != P_DEL) &&
	    (*(((WORD*)path->pset[w].b_ptr)->value) != (TEXT)'\0')){
	  if (nword+1 == curlen){
	    expand_1dimZ(conf,nword,curlen,2,double,0.0,0);
	    expand_1dimZ(eval,nword,curlen,2,int,0,1);
	  } 
	  conf[nword] = ((WORD *)path->pset[w].b_ptr)->conf;
	  eval[nword] = path->pset[w].eval;
	  nword++;
	}
    }

  /* step 2 */
  alloc_singZ(sort,curlen,int,0);
  sort_double_arr(conf, nword, sort, INCREASING);

  /* write the graph */
  if ((fp_dat = fopen(rsprintf("%s.sbhist.dat",outroot),"w")) == (FILE *)0){
    fprintf(stderr,"Error: Writing Scaled Binned Histogram file '%s' for"
	    " writing.",rsprintf("%s.sbhist.dat",outroot));
    goto ERROR;
  }

  for (b=nword / bins, l=0; l < b && l < nword; ){
    double min, max;
    for (w=l, min=99999.0, max=(-99999.9), sum_err=sum_corr=0; w<b; w++){
      if (eval[sort[w]] == P_CORR)
	sum_corr ++;
      else
	sum_err ++;
      if (min > conf[sort[w]]) min = conf[sort[w]];
      if (max < conf[sort[w]]) max = conf[sort[w]];
    }
    fprintf(fp_dat,"%f %f 1 1 %f\n",(min+max)/2,
	    (float)sum_corr/(sum_err+sum_corr)*100.0,(max-min));

    l=b;
    b += nword/bins;
    /* on the last bin, add the leftover words,
       (from doing modulo arithmetic) to it */
    if (b + (nword/bins) > nword) b = nword;
  }
  fclose(fp_dat);
  
  /* output the Scaled Binned histogram GNUPLUT command file */
  if ((fp_plt = fopen(rsprintf("%s.sbhist.plt",outroot),"w")) == (FILE *)0){
    fprintf(stderr,"Error: Unable to open Scaled Binned Histogram"
	    "GNUPLOT file '%s' for writing.",
	    rsprintf("%s.sbhist.plt",outroot));
    goto ERROR;
  }
  fprintf(fp_plt,"## GNUPLOT command file\n");
  fprintf(fp_plt,"set samples 1000\n");
  fprintf(fp_plt,"set key 30.000000,90.000000\n");
  fprintf(fp_plt,"set xrange [0:1]\n");
  fprintf(fp_plt,"set yrange [0:100]\n");
  fprintf(fp_plt,"set ylabel '%% Hypothesis Correct'\n");
  fprintf(fp_plt,"set xlabel 'Confidence Scores'\n");
  fprintf(fp_plt,"set title  'Scaled Binned Confidence scores for %s'\n",
	  outroot);
  fprintf(fp_plt,"set nogrid\n");
  fprintf(fp_plt,"set size 0.78,1\n");
  fprintf(fp_plt,"set nolabel\n");
  fprintf(fp_plt,"plot '%s.sbhist.dat'  title 'True' with boxes, x*100 title 'Expected'\n",
	  outroot);
  fprintf(fp_plt,"set size 1.0, 1.0\n");
  fprintf(fp_plt,"set key\n");

  fclose(fp_plt);
  
  rtn = 0;
  goto END;
  
ERROR: 
  rtn = 1;
END:
  
  free_singarr(conf,double);
  free_singarr(eval,int);
  free_singarr(sort,int);

  return(rtn);
}

/* 
   hist[x][H_PCT_CORR]  is the percent correct in bin 'x'
   hist[x][H_EXP_PCT_CORR]  is the expected correct in bin 'x'
   hist[x][H_N_CORR]  is the number of corrent words in bin 'x'
   hist[x][H_N_INCORR]  is the number of incorrect words in bin 'x'
   */

static void binned_confidences(SCORES *scor, double c_from, double c_to, int nbin, double **hist){
    int i, p, w, b, index;
    double index_d;
    double range = c_to - c_from;
    int range_errors = 0;


    for (b=0; b<nbin; b++){
        hist[b][H_FROM] = b * (range / (double)nbin);
	hist[b][H_TO] = (b + 1) * (range / (double)nbin);
	hist[b][H_N_CORR] = hist[b][H_N_INCORR] = 0;
    }
	
    for (i=0; i<scor->num_grp; i++){
        for (p=0; p<scor->grp[i].num_path; p++){
	    PATH *path = scor->grp[i].path[p];
	    for (w=0; w<path->num; w++) {
	      if ((path->pset[w].eval != P_DEL) &&
		  (*(((WORD*)path->pset[w].b_ptr)->value) != (TEXT)'\0')){
		    index_d = (((WORD *)path->pset[w].b_ptr)->conf - c_from)
		               / range * (double)(nbin);
		    index = (int)index_d;
		    if (index_d <= (double)nbin && index == nbin)
		        index = nbin - 1;

		    if (index < 0 || index >= nbin){
		      range_errors ++;
		    } else 
		        if (path->pset[w].eval == P_CORR)
			    hist[index][H_N_CORR] ++;
			else
			    hist[index][H_N_INCORR]++;
		}
	    }
	}
    }
    if (range_errors > 0){
      fprintf(stderr,"Warning: %d Confidence scores out of binning range %f:%f\n",
	      range_errors,c_from,c_to);
    }

    for (b=0; b<nbin; b++){
        if ((hist[b][H_N_CORR] + hist[b][H_N_INCORR]) == 0)
	    hist[b][H_PCT_CORR] = 0.0;
	else 
	    hist[b][H_PCT_CORR] = hist[b][H_N_CORR] / 
	      (hist[b][H_N_CORR]+hist[b][H_N_INCORR]) * 100.0;
	hist[b][H_EXP_PCT_CORR] = (hist[b][H_TO]+hist[b][H_FROM])/0.02;
       
	/* 
	   printf("Bin= %d  Range= %4.2f %4.2f #corr= %5.0f #incorr= %5.0f   "
	       "%%corr=%5.2f   %%exp= %5.2f\n",
	       b,hist[b][H_FROM],hist[b][H_TO],hist[b][H_N_CORR],
	       hist[b][H_N_INCORR],
	       hist[b][H_PCT_CORR],hist[b][H_EXP_PCT_CORR]);
	       */
    }

}


