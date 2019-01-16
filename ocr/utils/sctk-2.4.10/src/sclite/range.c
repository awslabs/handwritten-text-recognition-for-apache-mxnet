/**********************************************************************/
/*                                                                    */
/*             FILENAME:  range.c                                     */
/*             BY:  Jonathan G. Fiscus                                */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*             USAGE: this file contains programs to print a range    */
/*                    analysis table and graph for the systems and the*/
/*                    speakers                                        */
/*                                                                    */
/**********************************************************************/

#include "sctk.h"


static void print_gnu_range_graph  (RANK *, char *, char *, char *, int, int);
static void print_gnu_range_graph2  (RANK *, char *, char *, char *, int, int);
static void print_range_graph(int, double *, double *, double *, double *, double *, char **, int , char *, int *, FILE *);
static void do_blk_ranges(RANK *, int , char *, char *, FILE *);
static void do_trt_ranges(RANK *, int , char *, char *, FILE *);

/**********************************************************************/
/*  controlling program to print the ranges                           */
/**********************************************************************/
void print_gnu_rank_ranges(RANK *rank, char *percent_desc, char *testname,char *basename, int feedback)
{
    print_gnu_range_graph(rank,percent_desc,testname,basename,1, feedback);
    print_gnu_range_graph(rank,percent_desc,testname,basename,0, feedback);
}

/**********************************************************************/
/*  controlling program to print the ranges                           */
/**********************************************************************/
void print_gnu_rank_ranges2(RANK *rank, char *percent_desc, char *testname,char *basename, int feedback)
{
    print_gnu_range_graph2(rank,percent_desc,testname,basename,1,feedback);
    print_gnu_range_graph2(rank,percent_desc,testname,basename,0,feedback);
}

/**********************************************************************/
/*  controlling program to print the ranges                           */
/**********************************************************************/
void print_rank_ranges(RANK *rank, char *percent_desc, char *testname, char *outroot, int feedback)
{
    int scale;
    FILE *fp = stdout;
    
    if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(outroot,"w")) ==
	(FILE *)0){
	fprintf(stderr,"Warning: Open of %s for write failed.  "
		"Using stdout instead.\n",outroot);
	fp = stdout;
    } else
	if (feedback >= 1) printf("        Output written to '%s'\n",outroot);


    /* scale is whether to base graphs on 50(2) or 100(1) columns */
    if (pad_pr_width() > 100)
        scale = 1;
    else 
        scale = 2;

    do_blk_ranges(rank,scale,percent_desc,testname,fp);
    form_feed(fp);
    do_trt_ranges(rank,scale,percent_desc,testname,fp);
    if (fp == stdout) form_feed(fp);

    if (fp != stdout) fclose(fp);
}

/***********************************************************************/
/*                                                                     */
/*                          STATIC FUNCTIONS                           */



/**********************************************************************/
/*  do the block ranges, alias the speakers                           */
/**********************************************************************/
static void do_blk_ranges(RANK *rank, int scale, char *p_desc, char *tname, FILE *fp)
{
    int i,j;
    double Z_stat, *high, *low, *mean, *std_dev,variance,*pcts,median;
    char *title = "RANGE ANALYSIS ACROSS SPEAKERS FOR THE TEST:";
    char *title1= "by ";
    char pad[FULL_SCREEN];

    alloc_singarr(high, rank->n_blk, double);
    alloc_singarr(low, rank->n_blk, double);
    alloc_singarr(std_dev, rank->n_blk, double);
    alloc_singarr(mean, rank->n_blk, double);
    alloc_singarr(pcts, rank->n_trt, double);

    set_pad(pad,title, FULL_SCREEN);
    fprintf(fp,"\n\n\n%s%s\n",pad,title);
    set_pad(pad,tname, FULL_SCREEN);
    fprintf(fp,"%s%s\n",pad,tname);
    set_pad_cent_n(pad,strlen(title1)+strlen(p_desc), FULL_SCREEN);
    fprintf(fp,"%s%s%s\n\n\n\n\n",pad,title1,p_desc);

    /**** find the high, low, and the standard deviation for each block */
    for (i=0;i< rank->n_blk ;i++){
        high[i] = 0.0;
        low[i] = 100.0;
        mean[i] = 0.0;
        for (j=0;j< rank->n_trt ;j++){
            if (rank->pcts [ i ][ j ]  > high[i])
               high[i] = rank->pcts [ i ][ j ] ;
            if (rank->pcts [ i ][ j ]  < low[i])
                low[i] = rank->pcts [ i ][ j ] ;
            pcts[j] = rank->pcts [ i ][ j ] ;
        }
        calc_mean_var_std_dev_Zstat_double(pcts, rank->n_trt ,&(mean[i]),
                                    &(variance),&(std_dev[i]),&median,&Z_stat);
    }

    print_range_graph(scale,high,low,mean,std_dev, rank->blk_Ovr_ranks ,
		      rank->blk_name , rank->n_blk ,"  SPKR",
		      rank->trt_srt_ind,fp);

    free_singarr(high, double);
    free_singarr(low, double);
    free_singarr(std_dev, double);
    free_singarr(mean, double);
    free_singarr(pcts, double);
}

/**********************************************************************/
/*  do the treatments ranges, alias the systems                       */
/**********************************************************************/
static void do_trt_ranges(RANK *rank, int scale, char *p_desc, char *tname, FILE *fp)
{
    int i,j;
    double Z_stat, variance, *high, *low, *mean, *std_dev, *pcts, median;
    char *title = "RANGE ANALYSIS ACROSS RECOGNITION SYSTEMS FOR THE TEST:";
    char *title1= "by ";
    char pad[FULL_SCREEN];

    alloc_singarr(high, rank->n_trt, double);
    alloc_singarr(low, rank->n_trt, double);
    alloc_singarr(std_dev, rank->n_trt, double);
    alloc_singarr(mean, rank->n_trt, double);
    alloc_singarr(pcts, rank->n_blk, double);


    set_pad(pad,title, FULL_SCREEN);
    fprintf(fp,"\n\n\n%s%s\n",pad,title);
    set_pad(pad,tname, FULL_SCREEN);
    fprintf(fp,"%s%s\n",pad,tname);
    set_pad_cent_n(pad,strlen(title1)+strlen(p_desc), FULL_SCREEN);
    fprintf(fp,"%s%s%s\n\n\n\n\n",pad,title1,p_desc);

    /**** find the high, low, and the standard deviation for each treatment*/
    for (i=0;i< rank->n_trt ;i++){
        high[i] = 0.0;
        low[i] = 100.0;
        mean[i] = 0.0;
        for (j=0;j< rank->n_blk ;j++){
            if (rank->pcts [ j ][ i ]  > high[i])
               high[i] = rank->pcts [ j ][ i ] ;
            if (rank->pcts [ j ][ i ]  < low[i])
                low[i] = rank->pcts [ j ][ i ] ;
            pcts[j] = rank->pcts [ j ][ i ] ;
        }
        calc_mean_var_std_dev_Zstat_double(pcts, rank->n_blk ,&(mean[i]),
					  &(variance),&(std_dev[i]),&median,
					  &(Z_stat));
    }
    print_range_graph(scale,high,low,mean,std_dev, rank->trt_Ovr_ranks ,
		      rank->trt_name,rank->n_trt,"  SYS",rank->blk_srt_ind,fp);
    free_singarr(high, double);
    free_singarr(low, double);
    free_singarr(std_dev, double);
    free_singarr(mean, double);
    free_singarr(pcts, double);
}

/**********************************************************************/
/*   this program prints a graph showing the ranges in descending     */
/*   order of the means                                               */
/**********************************************************************/
static void print_range_graph(int scale, double *high, double *low, double *mean, double *std_dev, double *ovr_rank, char **r_names, int num_ranges, char *r_label, int *ptr_arr, FILE *fp)
{
    int i,j,pad_1, pad_2;
    int max_range_len=6, tmp;
    char pad[FULL_SCREEN], *pct_str = "PERCENTAGES";
    char range_format[20];

    for (i=0;i<num_ranges;i++)
        if ((tmp=strlen(r_names[i])) > max_range_len)
            max_range_len = tmp;
    max_range_len++;
    sprintf(range_format,"%%-%1ds",max_range_len);

    /************************************************************/
    /*   make the range table                                   */
    /************************************************************/
    /* print the header */
    set_pad_cent_n(pad,42+max_range_len, FULL_SCREEN);
    fprintf(fp,"%s|-",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp,"-");
    for (i=0;i<4;i++)
       fprintf(fp,"---------");
    fprintf(fp,"---|\n");
    fprintf(fp,"%s| ",pad);
    fprintf(fp,range_format,r_label);
    fprintf(fp,"|  high  |   low  || std dev |   mean  |\n");

    fprintf(fp,"%s|-",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp,"-");
    for (i=0;i<2;i++)
        fprintf(fp,"+--------");
    fprintf(fp,"+");
    for (i=0;i<2;i++)
        fprintf(fp,"+---------");
    fprintf(fp,"|\n");
    for (i=0;i<num_ranges;i++){
        fprintf(fp,"%s| ",pad);
        fprintf(fp,range_format,r_names[ptr_arr[i]]);
        fprintf(fp,"| %5.1f  ",high[ptr_arr[i]]);
        fprintf(fp,"| %5.1f  ",low[ptr_arr[i]]);
        fprintf(fp,"||  %5.1f  ",std_dev[ptr_arr[i]]);
        fprintf(fp,"|  %5.1f  |\n",mean[ptr_arr[i]]);
    }
    fprintf(fp,"%s|-",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp,"-");
    for (i=0;i<4;i++)
        fprintf(fp,"---------");
    fprintf(fp,"---|\n\n\n\n\n\n\n");

    /************************************************************/
    /*   make the range graph                                   */
    /************************************************************/
    set_pad_cent_n(pad,(100/scale) + (4+max_range_len), FULL_SCREEN);
    fprintf(fp,"%s|---",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp,"-");
    for (i=0;i<100;i+=(5*scale))
        fprintf(fp,"-----");
    fprintf(fp,"|\n");


    /*   center the percent string in the table */
    fprintf(fp,"%s| ",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp," ");
    fprintf(fp,"|");

    pad_1 =(((100/scale +1) - strlen(pct_str)) / 2);
    pad_2 =(100/scale +1) - (pad_1 + strlen(pct_str));
    set_pad_n(pad,pad_1, FULL_SCREEN);
    fprintf(fp,"%s%s",pad,pct_str);
    set_pad_n(pad,pad_2, FULL_SCREEN);
    fprintf(fp,"%s|\n",pad);

    /* print the scale */
    set_pad_cent_n(pad,(100/scale) + (4+max_range_len), FULL_SCREEN);
    fprintf(fp,"%s|-",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp,"-");
    fprintf(fp,"+-");
    for (i=0;i<100;i+=(5*scale))
        fprintf(fp,"-----");
    fprintf(fp,"|\n");

    fprintf(fp,"%s| ",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp," ");
    fprintf(fp,"|0");
    for (i=(5*scale);i<101;i+=(5*scale))
        fprintf(fp,"  %3d",i);
    fprintf(fp,"|\n");

    fprintf(fp,"%s| ",pad);
    fprintf(fp,range_format,r_label);
    fprintf(fp,"||");
    for (i=(5*scale);i<101;i+=(5*scale))
        fprintf(fp,"    |");
    fprintf(fp,"|\n");

    fprintf(fp,"%s|-",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp,"-");
    fprintf(fp,"+-");
    for (i=0;i<100;i+=(5*scale))
        fprintf(fp,"-----");
    fprintf(fp,"|\n");

    /* begin to make the graphs of the ranges */
    for (i=0;i<num_ranges;i++){
        fprintf(fp,"%s| ",pad);
        fprintf(fp,range_format,r_names[ptr_arr[i]]);
        fprintf(fp,"|");
        for (j=0;j<101;j+=scale){
            if ((j > (mean[ptr_arr[i]]-(0.0001+(double)(scale)/2.00))) &&
                (j < (mean[ptr_arr[i]]+(0.0001+(double)(scale)/2.00))))
                fprintf(fp,"|");
            else if (((j>(mean[ptr_arr[i]]+
                         std_dev[ptr_arr[i]]-(0.0001+(double)(scale)/2.00))) &&
                      (j<(mean[ptr_arr[i]]+
                         std_dev[ptr_arr[i]]+(0.0001+(double)(scale)/2.00))))||
                     ((j>(mean[ptr_arr[i]]-
                         std_dev[ptr_arr[i]]-(0.0001+(double)(scale)/2.00))) &&
                      (j<(mean[ptr_arr[i]]-
                         std_dev[ptr_arr[i]]+(0.0001+(double)(scale)/2.00)))))
                fprintf(fp,"+");
          
            else if ((j > high[ptr_arr[i]]) || 
                     (j < low[ptr_arr[i]]))
                fprintf(fp," ");
            else
                fprintf(fp,"-");
	}
        fprintf(fp,"|\n");
    }
    fprintf(fp,"%s|---",pad);
    for (i=0;i<max_range_len;i++)
        fprintf(fp,"-");
    for (i=0;i<100;i+=(5*scale))
        fprintf(fp,"-----");
    fprintf(fp,"|\n\n");

    fprintf(fp,"%s      |-> shows the mean\n",pad);
    fprintf(fp,"%s      +-> shows plus or minus one standard deviation\n",pad);
}

/**********************************************************************/
/*   this program prints a graph showing the ranges in descending     */
/*   order of the means                                               */
/**********************************************************************/
static void print_gnu_range_graph(RANK *rank, char *percent_desc, char *testname, char *base, int for_blks, int feedback)
{
    int b,t;
    FILE *fp_dat, *fp_mean, *fp_plt, *fp_median;
    double sum;
    TEXT *mean_name = (TEXT *)0, *dat_name = (TEXT *)0;
    TEXT *plt_name = (TEXT *)0, *basename = (TEXT *)0;
    TEXT *median_name = (TEXT *)0;
    double *pctt, *rnkk;
    int *ind;

    alloc_singarr(pctt,MAX( rank->n_blk , rank->n_trt ),double);
    alloc_singarr(rnkk,MAX( rank->n_blk , rank->n_trt ),double);
    alloc_singarr(ind,MAX( rank->n_blk , rank->n_trt ),int);

    basename = TEXT_strdup((TEXT *)rsprintf("%s%s",(strcmp(base,"-")==0) ? "STATS" : base,
	    (for_blks) ? ".spk" : ".sys"));
    dat_name = TEXT_strdup((TEXT *)rsprintf("%s%s",basename,".dat"));
    /* make the datafiles for the treatements */
    if ((fp_dat = fopen(dat_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT data file %s\n",
		dat_name);
	exit(1);
    } else
	if (feedback >= 1) printf("        Output written to '%s.*'\n",basename);

    mean_name = TEXT_strdup((TEXT *)rsprintf("%s%s",basename,".mean"));
    if ((fp_mean = fopen(mean_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT data file %s\n",
		mean_name);
	exit(1);
    }
    median_name = TEXT_strdup((TEXT *)rsprintf("%s%s",basename,".median"));
    if ((fp_median = fopen(median_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT data file %s\n",
		median_name);
	exit(1);
    }
    plt_name = TEXT_strdup((TEXT *)rsprintf("%s%s",basename,".plt"));
    if ((fp_plt = fopen(plt_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT file %s\n",
		plt_name);
	exit(1);
    } 
    fprintf(fp_plt,"set yrange [%d:0]\n",
	    1 + ((for_blks) ?rank->n_blk  :rank->n_trt ));
    fprintf(fp_plt,"set xrange [0:100]\n");
    fprintf(fp_plt,"set title \"%s\"\n",testname);
    fprintf(fp_plt,"set key\n");
    fprintf(fp_plt,"set ylabel \"%s\"\n",(for_blks) ? "Speaker ID" :"Systems");
    fprintf(fp_plt,"set xlabel \"%s\"\n",percent_desc);
    fprintf(fp_plt,"set ytics (");

    if (for_blks){
	for (b=0;b< rank->n_blk ;b++){
	    double medianval=0.0;
	    sum = 0.0;
	    for (t=0;t< rank->n_trt ;t++){
		fprintf(fp_dat,"%d %f\n",
			b+1,rank->pcts [ rank->trt_srt_ind [ b ]  ][ t ] );
		pctt[t] = rank->pcts [ rank->trt_srt_ind [ b ]  ][ t ] ;
		sum += rank->pcts [ rank->trt_srt_ind [ b ]  ][ t ] ;
	    }

	    rank_double_arr(pctt, rank->n_trt ,ind,rnkk,INCREASING);
	    if ( rank->n_trt  % 2 == 0) { /* handle the even arr len */
		medianval = (pctt[ind[ rank->n_trt  / 2]] + 
			     pctt[ind[( rank->n_trt  / 2) - 1]]) / 2.0;
	    } else { /* handle the odd arr len */
		medianval = pctt[ind[ rank->n_trt  / 2]];
	    }
	    fprintf(fp_mean,"%d %f\n",b+1,sum / (double)( rank->n_trt ));
	    fprintf(fp_median,"%d %f\n",b+1,medianval);
	    fprintf(fp_plt,"\"%s\" %d",rank->blk_name[rank->trt_srt_ind[b]],
		    b+1);
	    if (b !=rank->n_blk -1)
		fprintf(fp_plt,",");
	}
	fprintf(fp_plt,")\n");
    } else {
	for (t=0;t< rank->n_trt ;t++){
	    double medianval=0.0;
	    sum = 0.0;

	    for (b=0;b< rank->n_blk ;b++){
		fprintf(fp_dat,"%d %f\n",
			t+1,rank->pcts [ b ][ rank->blk_srt_ind [ t ]  ] );
		pctt[b] = rank->pcts [ b ][ rank->blk_srt_ind [ t ]  ] ;
		sum += rank->pcts [ b ][ rank->blk_srt_ind [ t ]  ] ;
	    }
    rank_double_arr(pctt, rank->n_blk ,ind,rnkk,INCREASING);
	    if ( rank->n_blk  % 2 == 0) { /* handle the even arr len */
		medianval = (pctt[ind[ rank->n_blk  / 2]] + 
			     pctt[ind[( rank->n_blk  / 2) - 1]]) / 2.0;
	    } else { /* handle the odd arr len */
		medianval = pctt[ind[ rank->n_blk  / 2]];
	    }
	    fprintf(fp_mean,"%d %f\n",t+1,sum / (double)(rank->n_blk ));
	    fprintf(fp_median,"%d %f\n",t+1,medianval);
	    fprintf(fp_plt,"\"%s\" %d",rank->trt_name[rank->blk_srt_ind[t]],
		    t+1);
	    if (t !=rank->n_trt -1)
		fprintf(fp_plt,",");
	}
	fprintf(fp_plt,")\n");
    }

    fprintf(fp_plt,"plot \"%s\" using 2:1 title \"Mean %s\" with lines,",
	    mean_name, percent_desc);
    fprintf(fp_plt,"\"%s\" using 2:1 title \"Median %s\" with lines,",
	    median_name, percent_desc);
    fprintf(fp_plt,"\"%s\" using 2:1 title \"Individual %s\"\n",dat_name,
	    percent_desc);
    fclose(fp_dat);
    fclose(fp_mean);
    fclose(fp_median);
    fclose(fp_plt);

    free_singarr(mean_name  , TEXT);
    free_singarr(dat_name   , TEXT);
    free_singarr(plt_name   , TEXT);
    free_singarr(basename   , TEXT);
    free_singarr(median_name, TEXT);

    free_singarr(pctt,double);
    free_singarr(rnkk,double);
    free_singarr(ind,int);
}

/**********************************************************************/
/*   this program prints a graph showing the ranges in descending     */
/*   order of the means                                               */
/**********************************************************************/
static void print_gnu_range_graph2(RANK *rank, char *percent_desc, char *testname, char *base, int for_blks, int feedback)
{
    int b,t;
    FILE *fp_dat, *fp_mean, *fp_plt, *fp_median;
    double sum;
    char mean_name[50], dat_name[50], plt_name[50], basename[50];
    char median_name[50];
    double *pctt, *rnkk;
    int *ind;

    alloc_singarr(pctt,MAX( rank->n_blk , rank->n_trt ),double);
    alloc_singarr(rnkk,MAX( rank->n_blk , rank->n_trt ),double);
    alloc_singarr(ind,MAX( rank->n_blk , rank->n_trt ),int);

    sprintf(basename,"%s%s",(strcmp(base,"-")==0) ? "STATS" : base,
	    (for_blks) ? ".spk" : ".sys");
    sprintf(dat_name,"%s%s",basename,".dat");
    /* make the datafiles for the treatements */
    if ((fp_dat = fopen(dat_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT data file %s\n",
		dat_name);
	exit(1);
    } else
	if (feedback >= 1) printf("        Output written to '%s.*'\n",basename);

    sprintf(mean_name,"%s%s",basename,".mean");
    if ((fp_mean = fopen(mean_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT data file %s\n",
		mean_name);
	exit(1);
    }
    sprintf(median_name,"%s%s",basename,".median");
    if ((fp_median = fopen(median_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT data file %s\n",
		median_name);
	exit(1);
    }
    sprintf(plt_name,"%s%s",basename,".plt");
    if ((fp_plt = fopen(plt_name,"w")) == (FILE *)0){
	fprintf(stderr,"Error: unable to open GNUPLOT file %s\n",
		plt_name);
	exit(1);
    } 
    fprintf(fp_plt,"set yrange [%d:0]\n",
	    1 + ((for_blks) ?rank->n_blk  :rank->n_trt ));
    fprintf(fp_plt,"set xrange [0:100]\n");
    fprintf(fp_plt,"set title \"%s\"\n",testname);
    fprintf(fp_plt,"set key\n");
    fprintf(fp_plt,"set ylabel \"%s\"\n",(for_blks) ? "Speaker ID" :"Systems");
    fprintf(fp_plt,"set xlabel \"%s\"\n",percent_desc);
    fprintf(fp_plt,"set ytics (");

    if (for_blks){
	/* build the data file */
	for (b=0;b< rank->n_blk ;b++){
	    fprintf(fp_dat,"%d",b+1);
	    for (t=0;t< rank->n_trt ;t++){
		fprintf(fp_dat," %f",
			rank->pcts [ rank->trt_srt_ind [ b ]  ][ t ] );
	    }
	    fprintf(fp_dat,"\n");
	}
	for (b=0;b< rank->n_blk ;b++){
	    double medianval=0.0;
	    sum = 0.0;
	    for (t=0;t< rank->n_trt ;t++){
		pctt[t] = rank->pcts [ rank->trt_srt_ind [ b ]  ][ t ] ;
		sum += rank->pcts [ rank->trt_srt_ind [ b ]  ][ t ] ;
	    }

	    rank_double_arr(pctt, rank->n_trt ,ind,rnkk,INCREASING);
	    if ( rank->n_trt  % 2 == 0) { /* handle the even arr len */
		medianval = (pctt[ind[ rank->n_trt  / 2]] + 
			     pctt[ind[( rank->n_trt  / 2) - 1]]) / 2.0;
	    } else { /* handle the odd arr len */
		medianval = pctt[ind[ rank->n_trt  / 2]];
	    }
	    fprintf(fp_mean,"%d %f\n",b+1,sum / (double)( rank->n_trt ));
	    fprintf(fp_median,"%d %f\n",b+1,medianval);
	    fprintf(fp_plt,"\"%s\" %d",rank->blk_name[rank->trt_srt_ind[b]],
		    b+1);
	    if (b !=rank->n_blk -1)
		fprintf(fp_plt,",");
	}
	fprintf(fp_plt,")\n");
	fprintf(fp_plt,"plot \"%s\" using 2:1 title \"Mean %s\" with lines,\\\n",
		mean_name, percent_desc);
	fprintf(fp_plt,"     \"%s\" using 2:1 title \"Median %s\" with lines",
		median_name, percent_desc);
	for (t=0;t< rank->n_trt ;t++){
	    char fmt[100]; int x;
	    strcpy(fmt,"%f");
	    for (x=0; x<t; x++)
		strcat(fmt,"%*s");
	    strcat(fmt,"%f");
	    fprintf(fp_plt,",\\\n     \"%s\" using 2:1 \"%s\" title \"%s\"",
		    dat_name,fmt,rank->trt_name[t]);
	}
	fprintf(fp_plt,"\n");
    } else {
	for (t=0;t< rank->n_trt ;t++){
	    fprintf(fp_dat,"%d",t+1);
	    for (b=0;b< rank->n_blk ;b++){
		fprintf(fp_dat," %f",
			rank->pcts [ b ][ rank->blk_srt_ind [ t ]  ] );
	    }
	    fprintf(fp_dat,"\n");
	}

	for (t=0;t< rank->n_trt ;t++){
	    double medianval=0.0;
	    sum = 0.0;

	    for (b=0;b< rank->n_blk ;b++){
		pctt[b] = rank->pcts [ b ][ rank->blk_srt_ind [ t ]  ] ;
		sum += rank->pcts [ b ][ rank->blk_srt_ind [ t ]  ] ;
	    }	
	    rank_double_arr(pctt, rank->n_blk ,ind,rnkk,INCREASING);
	    if ( rank->n_blk  % 2 == 0) { /* handle the even arr len */
		medianval = (pctt[ind[ rank->n_blk  / 2]] + 
			     pctt[ind[( rank->n_blk  / 2) - 1]]) / 2.0;
	    } else { /* handle the odd arr len */
		medianval = pctt[ind[ rank->n_blk  / 2]];
	    }
	    fprintf(fp_mean,"%d %f\n",t+1,sum / (double)(rank->n_blk ));
	    fprintf(fp_median,"%d %f\n",t+1,medianval);
	    fprintf(fp_plt,"\"%s\" %d",rank->trt_name[rank->blk_srt_ind[t]],
		    t+1);
	    if (t !=rank->n_trt -1)
		fprintf(fp_plt,",");
	}
	fprintf(fp_plt,")\n");
	fprintf(fp_plt,"plot \"%s\" using 2:1 title \"Mean %s\" with lines,\\\n",
		mean_name, percent_desc);
	fprintf(fp_plt,"     \"%s\" using 2:1 title \"Median %s\" with lines",
		median_name, percent_desc);
	for (b=0;b< rank->n_blk ;b++){
	    char fmt[100]; int x;
	    strcpy(fmt,"%f");
	    for (x=0; x<b; x++)
		strcat(fmt,"%*s");
	    strcat(fmt,"%f");
	    fprintf(fp_plt,",\\\n     \"%s\" using 2:1 \"%s\" title \"%s\"",
		    dat_name,fmt,rank->blk_name[b]);
	}
	fprintf(fp_plt,"\n");
    }

    fclose(fp_dat);
    fclose(fp_mean);
    fclose(fp_median);
    fclose(fp_plt);

    free_singarr(pctt,double);
    free_singarr(rnkk,double);
    free_singarr(ind,int);
}

