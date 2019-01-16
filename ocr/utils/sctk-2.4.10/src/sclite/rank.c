/**********************************************************************/
/*                                                                    */
/*           FILE: rank.c                                             */
/*           WRITTEN BY: Jonathan G. Fiscus                           */
/*                  NATIONAL INSTITUTE OF STANDARDS AND TECHNOLOGY    */
/*                  SPEECH RECOGNITION GROUP                          */
/*                                                                    */
/*           USAGE: for general procedures to manipulate a RANK       */
/*                  structure                                         */
/*                                                                    */
/**********************************************************************/

#include "sctk.h"

/**********************************************************************/
/*   initialize and allocate memory for the RANK structure from the   */
/*   SYS_ALIGN_LIST                                                   */
/**********************************************************************/
void init_RANK_struct_from_SCORES(RANK *rank, SCORES *scor[], int nscor, char *calc_formula)
{
    int blk=0,trt,i,n,mlen=0,msclen=0,g;
    int c,s,d;

    /* set it to the max number of speakers for all systems */
    for (n=0; n<nscor; n++){
	if (blk < scor[n]->num_grp)
	    blk = scor[n]->num_grp;
	if (msclen < strlen(scor[n]->title))
		msclen = strlen(scor[n]->title);
	for (g=0; g<scor[n]->num_grp; g++)
	    if (mlen < strlen(scor[n]->grp[g].name))
		mlen = strlen(scor[n]->grp[g].name);
    }
    trt = nscor;

    rank->n_blk  = blk;
    rank->n_trt  = trt;
    alloc_2dimarr(rank->blk_name,blk,mlen+1,char);
    /* copy in only what's available from the first system */
    for (i=0; i<blk; i++)
	if (i < scor[0]->num_grp)
	    strcpy(rank->blk_name[i] ,scor[0]->grp[i].name);
    alloc_2dimarr(rank->trt_name,trt,msclen+1,char);
    for (i=0; i<trt; i++){
        strcpy(rank->trt_name[i],scor[i]->title);
    }
    alloc_2dimZ(rank->trt_ranks ,blk,trt,double,0.0);
    alloc_2dimZ(rank->blk_ranks ,blk,trt,double,0.0);
    alloc_2dimarr(rank->pcts ,blk,trt,double);
    alloc_singarr(rank->trt_Ovr_ranks ,trt,double);
    alloc_singarr(rank->blk_Ovr_ranks ,blk,double);
    alloc_singarr(rank->blk_srt_ind ,trt,int);
    alloc_singarr(rank->trt_srt_ind ,blk,int);

    for (n=0; n<nscor; n++)
	for (g=0; g<scor[n]->num_grp; g++){
	    c = scor[n]->grp[g].corr;
	    s = scor[n]->grp[g].sub; 
	    d = scor[n]->grp[g].del;
	    i = scor[n]->grp[g].ins;
            switch (*calc_formula) {
	      case 'R':
		rank->pcts[g][n] = pct(c, (c + s + d));
		break;
	      case 'E':
		rank->pcts [g][n] = pct((i + s + d),(c + d + s));
		break;
	      case 'W':
		rank->pcts[g][n] = 100.0000 - 
		    pct((i + s + d),(c + d + s));
		break;
	      default:
		fprintf(stderr,"You did not give calc_pct a formula\n");
		break;
	    }
	}

    if (*calc_formula == 'E' )
        rank_on_pcts(rank,INCREASING);
    else
        rank_on_pcts(rank,DECREASING);
}

void free_RANK(RANK *rank)
{
    free_2dimarr(rank->blk_name,rank->n_blk,char);
    free_2dimarr(rank->trt_name,rank->n_trt,char);

    free_2dimarr(rank->trt_ranks,rank->n_blk,double);
    free_2dimarr(rank->blk_ranks,rank->n_blk,double);
    free_2dimarr(rank->pcts,rank->n_blk,double);

    free_singarr(rank->trt_Ovr_ranks ,double);
    free_singarr(rank->blk_Ovr_ranks ,double);
    free_singarr(rank->blk_srt_ind ,int);
    free_singarr(rank->trt_srt_ind ,int);


}

/**********************************************************************/
/*  given the RANK structure rank the percentages across blocks and   */
/*  treatments                                                        */
/**********************************************************************/
void rank_on_pcts(RANK *rank, int ordering)
{
    int t, b;
    int *t_ptr_arr;
    double *b_pcts, *b_ranks, *t_pcts, *t_ranks;

    alloc_singZ(t_ptr_arr, rank->n_trt ,int,0);
    alloc_singZ(b_pcts, rank->n_blk ,double,0.0);
    alloc_singZ(b_ranks, rank->n_blk ,double,0.0);
    alloc_singZ(t_pcts, rank->n_trt ,double,0.0);
    alloc_singZ(t_ranks, rank->n_trt ,double,0.0);

    /**** ranks the blocks */
    /**** i.e. rank this speaker's results over all the systems */
    for (b=0;b< rank->n_blk ;b++){
        rank_double_arr( rank->pcts[ b ] ,
			 rank->n_trt ,
			 t_ptr_arr,
                         rank->trt_ranks [ b ] ,
			 ordering);
    }

    /****  rank the treatments */
    /****  i.e. rank this systems speakers */	
    for (t=0;t< rank->n_trt ;t++){
        /**** since treatment pcts are the same index into n arrays, */
        /**** use a temporary array                                  */
        for (b=0;b< rank->n_blk ;b++)
            b_pcts[b] =   rank->pcts [ b ][ t ] ;

        rank_double_arr(b_pcts,
                        rank->n_blk ,
                        rank->trt_srt_ind ,
			b_ranks,
			ordering);
        for (b=0;b< rank->n_blk ;b++){
              rank->blk_ranks [ b ][ t ]  = b_ranks[b];
	}
    }

    /*  rank the systems in an overall sense */
    for (b=0;b< rank->n_blk ;b++)
        b_pcts[b] = 0.0;
    for (t=0; t< rank->n_trt ; t++)
        t_pcts[t] = 0.0;

    for (t=0; t< rank->n_trt ; t++)
        for (b=0;b< rank->n_blk ;b++){
            t_pcts[t] +=   rank->pcts [ b ][ t ] ;
            b_pcts[b] +=   rank->pcts [ b ][ t ] ;
	}

    rank_double_arr(t_pcts, rank->n_trt , rank->blk_srt_ind ,t_ranks,ordering);
    rank_double_arr(b_pcts, rank->n_blk , rank->trt_srt_ind ,b_ranks,ordering);

    free_singarr(b_ranks,double)
    free_singarr(b_pcts,double)
    free_singarr(t_ranks,double)
    free_singarr(t_pcts,double)
    free_singarr(t_ptr_arr,int)
}

#define FULL_SCREEN 132
/**********************************************************************/
/*   print to stdout the full rank structure                          */
/**********************************************************************/
void dump_full_RANK_struct(RANK *rank, char *t_name, char *b_name, char *blk_label, char *trt_label, char *formula, char *test_name, char *blk_desc, char *trt_desc)
{
    char *title 	= "RANKING TABLE BY PERCENTAGES FOR THE TEST";
/*    char *title_3	= "         System ranks for the speaker";
    char *title_4	= "         Speaker ranks for the system";*/
    double sum_blk;

    int i,j, *ts, *bs;
    char mfmt[50], tfmt[50], trfmt[50];

    /* use temporary pointers to point to the sorted indexes arrays */
    ts =  rank->blk_srt_ind ;
    bs =  rank->trt_srt_ind ;


    strcpy(mfmt,"ll");
    strcpy(tfmt,"la");
    strcpy(trfmt,"cc");
    for (i=0;i< rank->n_trt ;i++){
	strcat(mfmt,"|c");
	strcat(tfmt,"|c");
	strcat(trfmt,(i==0) ? "|c" : "|a");
    }
    strcat(mfmt,"|c");
    strcat(tfmt,"|c");
    strcat(trfmt,"|c");

    Desc_erase();
    Desc_set_page_center(SCREEN_WIDTH);
    Desc_add_row_values("c",title);
    Desc_add_row_values("rl","Showing:",formula_str(formula));
    Desc_add_row_values("rl","",trt_desc);
    Desc_add_row_values("rl","",blk_desc);
    Desc_add_row_separation('-',BEFORE_ROW);
    Desc_add_row_separation(' ',AFTER_ROW);
    Desc_add_row_values(trfmt,"","",t_name,"Mean Pct");

    Desc_set_iterated_format(tfmt);
    Desc_set_iterated_value("");
    for (j=0;j< rank->n_trt ;j++)
	Desc_set_iterated_value(rank->trt_name[ts[j]]);
    Desc_set_iterated_value("");
    Desc_flush_iterated_row();

    Desc_set_iterated_format(tfmt);
    Desc_set_iterated_value(b_name);
    for (i=0;i< rank->n_trt ;i++)
	Desc_set_iterated_value("");
    Desc_set_iterated_value("Mean Rank");
    Desc_flush_iterated_row();

    for (i=0; i<rank->n_blk; i++){
	Desc_add_row_separation('-',BEFORE_ROW);
	/* the percents */
	sum_blk = 0;
	Desc_set_iterated_format(mfmt);
	Desc_set_iterated_value(rank->blk_name[bs[i]]);
	Desc_set_iterated_value("Percent");
	for (j=0;j< rank->n_trt ;j++){
	    Desc_set_iterated_value(rsprintf("%5.1f",rank->pcts[bs[i]][ts[j]]));
	    sum_blk += rank->pcts[bs[i]][ts[j]];
	}
	Desc_set_iterated_value(rsprintf("%5.1f",sum_blk/(double)rank->n_trt));
	Desc_flush_iterated_row();

	/* the system ranks */
	Desc_set_iterated_format(mfmt);
	Desc_set_iterated_value("");
	Desc_set_iterated_value("Sys rnk");
	for (j=0;j< rank->n_trt ;j++)
	    Desc_set_iterated_value(rsprintf("%5.1f",rank->trt_ranks[bs[i]][ts[j]]));
	Desc_set_iterated_value("");
	Desc_flush_iterated_row();

	/* the Block ranks */
	Desc_set_iterated_format(mfmt);
	Desc_set_iterated_value("");
	Desc_set_iterated_value("Spkr rnk");
	sum_blk = 0;
	for (j=0;j< rank->n_trt ;j++){
	    Desc_set_iterated_value(rsprintf("%5.1f",rank->blk_ranks[bs[i]][ts[j]]));
	    sum_blk += rank->blk_ranks[bs[i]][ts[j]];
	}
	Desc_set_iterated_value(rsprintf("%5.1f",sum_blk/(double)rank->n_trt));
	Desc_flush_iterated_row();
    }
    Desc_add_row_separation('-',BEFORE_ROW);
    /* line 1 */
    Desc_set_iterated_format(tfmt);
    Desc_set_iterated_value("Mean pct");
    for (j=0;j< rank->n_trt ;j++){
	for (i=0, sum_blk = 0; i<rank->n_blk; i++)
	    sum_blk += rank->pcts[bs[i]][ts[j]];
	Desc_set_iterated_value(rsprintf("%5.1f",sum_blk/(double)rank->n_blk));
    }
    Desc_set_iterated_value("");
    Desc_flush_iterated_row();
    /* line 2 */
    Desc_add_row_separation(' ',AFTER_ROW);
    Desc_set_iterated_format(tfmt);
    Desc_set_iterated_value("Mean ranks");
    for (j=0;j< rank->n_trt ;j++){
	for (i=0, sum_blk = 0; i<rank->n_blk; i++)
	    sum_blk += rank->trt_ranks[bs[i]][ts[j]];
	Desc_set_iterated_value(rsprintf("%5.1f",sum_blk/(double)rank->n_blk));
    }
    Desc_set_iterated_value("");
    Desc_flush_iterated_row();

    Desc_dump_report(0,stdout);

    form_feed(stdout);
}

void print_n_winner_comp_matrix(RANK *rank, int **wins[], char *win_ids[], int win_cnt, int page_width,FILE *fp)
{

    int report_width, max_name_len=0, len_per_trt, trt, trt2, i, wid;
    int max_win_ids_len=0;
    char sysname_fmt[40], statname_fmt[40], sysname_fmt_center[40];

    /* find the largest treat length */
    for (trt=0;trt< rank->n_trt ;trt++)
      if ((i=strlen(  rank->trt_name [ trt ] )) > max_name_len)
         max_name_len = i;
    /* find the largest id length */
    for (wid=0; wid<win_cnt; wid++)
      if ((i=strlen(win_ids[wid])) > max_win_ids_len)
         max_win_ids_len=i;
    wid=0;
    sprintf(statname_fmt," %%%ds",max_win_ids_len);
    sprintf(sysname_fmt_center," %%%ds |",max_name_len+1+max_win_ids_len);
    sprintf(sysname_fmt," %%%ds |",max_name_len);

  
    len_per_trt = 3 + max_name_len + 1 + max_win_ids_len;
    report_width=( rank->n_trt +1) * len_per_trt + 1; 
     

    /* begin to print the report matrix */
    /* just a line */
    fprintf(fp,"%s",center("",(page_width-report_width)/2));
    fprintf(fp,"|");
    for (i=0; i<len_per_trt * ( rank->n_trt +1) - 1; i++)
        fprintf(fp,"-");
    fprintf(fp,"|\n");
    /* the treatment titles */
    fprintf(fp,"%s",center("",(page_width-report_width)/2));

    fprintf(fp,"|");
    fprintf(fp,sysname_fmt_center,"");
    for (trt=0; trt< rank->n_trt ; trt++){
        fprintf(fp,sysname_fmt_center,center(  rank->trt_name [ trt ] ,len_per_trt-3));
    }
    fprintf(fp,"\n");
    for (trt2=0; trt2< rank->n_trt ; trt2++){
        /* spacer lines */
        fprintf(fp,"%s",center("",(page_width-report_width)/2));
	fprintf(fp,"|");
        for (trt=0; trt< rank->n_trt +1; trt++){
            for (i=0; i<len_per_trt - 1; i++)
                fprintf(fp,"-");
            if (trt <  rank->n_trt )
                fprintf(fp,"+");
        }
        fprintf(fp,"|\n");
        for (wid=0; wid<win_cnt; wid++){
            fprintf(fp,"%s",center("",(page_width-report_width)/2));
            fprintf(fp,"|");
            if (wid == 0)
                fprintf(fp,sysname_fmt_center,center(  rank->trt_name [ trt2 ] ,len_per_trt-3));
            else
                fprintf(fp,sysname_fmt_center,"");
            for (trt=0; trt< rank->n_trt ; trt++){
                if (trt2 >= trt)
                    fprintf(fp,statname_fmt,"");
                else
                    fprintf(fp,statname_fmt,win_ids[wid]);
 
                if (trt2 >= trt)
                    fprintf(fp,sysname_fmt,"");
                else if (wins[wid][trt2][trt] == NO_DIFF)
                    fprintf(fp,sysname_fmt,"same");
                else if (wins[wid][trt2][trt] < 0)
                    fprintf(fp,sysname_fmt,  rank->trt_name [ trt2 ] );
                else
                    fprintf(fp,sysname_fmt,  rank->trt_name [ trt ] );
            }
            fprintf(fp,"\n");
        }
    }   
    /* just a line */
    fprintf(fp,"%s",center("",(page_width-report_width)/2));
    fprintf(fp,"|");
    for (i=0; i<len_per_trt * ( rank->n_trt +1) - 1; i++)
        fprintf(fp,"-");
    fprintf(fp,"|\n");
    form_feed(fp);

}

void print_composite_significance(RANK *rank, int pr_width, int num_win, int ***wins, char **win_desc, char **win_str, int matrix, int report, char *test_name, char *outroot, int feedback, char *outdir)
{
    char title[200];
    int i,max_desc_len=9,max_str_len=7,wid;
    FILE *fp = stdout;
    
    if (report || matrix){
	char *f = rsprintf("%s.unified",outroot);
	if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(f,"w")) ==
	    (FILE *)0){
	    fprintf(stderr,"Warning: Open of %s for write failed.  "
		           "Using stdout instead.\n",f);
	    fp = stdout;
	} else
	    if (feedback >= 1) printf("        Output written to '%s'\n",f);
    }

    for (wid=0; wid<num_win; wid++){
        if ((i=strlen(win_str[wid])) > max_str_len)
            max_str_len=i;
        if ((i=strlen(win_desc[wid])) > max_desc_len)
            max_desc_len=i;
    }

    if (matrix){
        fprintf(fp,"%s\n",
		center("Composite Report of All Significance Tests",pr_width));
        sprintf(title,"For the %s Test",test_name);
        fprintf(fp,"%s\n\n",center(title,pr_width));
        fprintf(fp,"%s",
		center("",(pr_width - (max_desc_len+5+max_str_len)) / 2));
        fprintf(fp,"%s",center("Test Name",max_desc_len));
        fprintf(fp,"%s",center("",5));
        fprintf(fp,"%s\n",center("Abbrev.",max_str_len));
        /* a line of dashes */
        fprintf(fp,"%s",
		center("",(pr_width - (max_desc_len+5+max_str_len)) / 2));
        for (i=0; i<max_desc_len; i++)
            fprintf(fp,"-");
        fprintf(fp,"%s",center("",5));
        for (i=0; i<max_str_len; i++)
            fprintf(fp,"-");
        fprintf(fp,"\n");
        for (wid=0; wid<num_win; wid++){
            fprintf(fp,"%s",
		    center("",(pr_width-(max_desc_len+5+max_str_len)) / 2));
            fprintf(fp,"%s",center(win_desc[wid],max_desc_len));
            fprintf(fp,"%s",center("",5));
            fprintf(fp,"%s\n",center(win_str[wid],max_str_len));
        }
        fprintf(fp,"\n\n");
        print_n_winner_comp_matrix(rank,wins,win_str,num_win,pr_width,fp);
    }
    if (report)
        print_n_winner_comp_report(rank,wins,win_str,win_str,win_desc,
				   num_win,pr_width,test_name,outdir);

    if (fp != stdout) fclose(fp);
}


void print_composite_significance2(RANK *rank, int pr_width, int num_win, int ***wins, double ***conf, char **win_desc, char **win_str, int matrix, int report, char *test_name, char *outroot, int feedback, char *outdir)
{
    char buff[100];
    int i,max_desc_len=9,max_str_len=7,wid;
    FILE *fp = stdout;
    
    if (report || matrix){
	char *f = rsprintf("%s.unified",outroot);
	if ((fp=(strcmp(outroot,"-") == 0) ? stdout : fopen(f,"w")) ==
	    (FILE *)0){
	    fprintf(stderr,"Warning: Open of %s for write failed.  "
		           "Using stdout instead.\n",f);
	    fp = stdout;
	} else
	    if (feedback >= 1) printf("        Output written to '%s'\n",f);
    }

    for (wid=0; wid<num_win; wid++){
        if ((i=strlen(win_str[wid])) > max_str_len)
            max_str_len=i;
        if ((i=strlen(win_desc[wid])) > max_desc_len)
            max_desc_len=i;
    }

    if (matrix){
	char title_fmt[100], row_fmt[100], header_fmt[100];
	int trt1, trt2;

	/* build the titles format */
	*title_fmt = *row_fmt = '\0';
	strcpy(title_fmt,"c=c");
	strcpy(row_fmt,"c=c");
	strcpy(header_fmt,"ca");
	for (trt1=0; trt1< rank->n_trt ; trt1++){
	    strcat(title_fmt,"|caa");
	    strcat(row_fmt,"|rrl");
	    strcat(header_fmt,"aaa");
	}
	strcat(title_fmt,"=c");
	strcat(row_fmt,"=c");
	strcat(header_fmt,"a");

	Desc_erase();
	Desc_set_page_center(pr_width);
	Desc_add_row_values(header_fmt,
			    "Composite Report of All Significance Tests");
	Desc_add_row_values(header_fmt,rsprintf("For the %s Test",test_name));
	Desc_add_row_separation(' ',BEFORE_ROW);

	/* cheat on this to make it look nice */
	strcpy(buff,center("Test Name",max_desc_len));
	strcat(buff,center("",5));
        strcat(buff,center("Abbrev.",max_str_len));
	Desc_add_row_values(header_fmt,buff);

	/* a line of dashes */
	*buff = '\0';        
        for (i=0; i<max_desc_len; i++)
            strcat(buff,"-");
	strcat(buff,center("",5));
        for (i=0; i<max_str_len; i++)
            strcat(buff,"-");
	Desc_add_row_values(header_fmt,buff);

	/* add the descriptions */
        for (wid=0; wid<num_win; wid++){
	    if (wid == num_win-1){
		Desc_add_row_separation(' ',AFTER_ROW);
		Desc_add_row_separation(' ',AFTER_ROW);
	    }
	    *buff = '\0';        
	    strcpy(buff,center(win_desc[wid],max_desc_len));
	    strcat(buff,center("",5));
	    strcat(buff,center(win_str[wid],max_str_len));
	    Desc_add_row_values(header_fmt,buff);
        }
	Desc_add_row_separation('-',BEFORE_ROW);
	
	/* begin the matrix */


	/* the treatment titles */
	Desc_set_iterated_format(title_fmt);
	Desc_set_iterated_value("Test//Abbrev.");
	Desc_set_iterated_value("");
	for (trt1=0; trt1< rank->n_trt ; trt1++)
	    Desc_set_iterated_value(rank->trt_name[trt1]);
	Desc_set_iterated_value("Test//Abbrev.");
	Desc_flush_iterated_row();

	/* add the row information */
	for (trt1=0; trt1< rank->n_trt ; trt1++){
	    Desc_add_row_separation('-',BEFORE_ROW);
	    for (wid=0; wid<num_win; wid++){
		Desc_set_iterated_format(row_fmt);		
		Desc_set_iterated_value(win_str[wid]);
		Desc_set_iterated_value((wid == 0) ?
					rank->trt_name[trt1] : "");
		for (trt2=0; trt2< rank->n_trt ; trt2++){
		    if (trt1 >= trt2){
			Desc_set_iterated_value("");
			Desc_set_iterated_value("");
			Desc_set_iterated_value("");
		    } else {
		      /* Desc_set_iterated_value(win_str[wid]); */
			if (wins[wid][trt1][trt2] == NO_DIFF)
			    Desc_set_iterated_value("~"/*"same"*/);
			else if (wins[wid][trt1][trt2] < 0)
			    Desc_set_iterated_value(rank->trt_name[trt1]);
			else
			    Desc_set_iterated_value(rank->trt_name[trt2]);
			if (conf[wid][trt1][trt2] < 0.001) 
			    Desc_set_iterated_value("<0.001");
			else 
			    Desc_set_iterated_value(rsprintf("%.3f",conf[wid][trt1][trt2]));
			if (conf[wid][trt1][trt2] <= 0.001 )
			    Desc_set_iterated_value("***");
			else if (conf[wid][trt1][trt2] <= 0.01)
			    Desc_set_iterated_value("**");
			else if (conf[wid][trt1][trt2] <= 0.05)
			    Desc_set_iterated_value("*");
			else
			    Desc_set_iterated_value("");
		    }
		}
		Desc_set_iterated_value(win_str[wid]);
		Desc_flush_iterated_row();
	    }
	}
	Desc_add_row_separation('-',BEFORE_ROW);
	/* 
	Desc_add_row_values(header_fmt,"Element Legend");
	Desc_add_row_separation(' ',BEFORE_ROW);
	Desc_add_row_values(header_fmt,
		    " TEST         LOWER        TEST        *   (0.01  < signficance level <= 0.05) ");
	Desc_add_row_values(header_fmt,
		    "ABBREV      ERR. RATE      SIGNIF.     **  (0.001 < signficance level <= 0.01) ");
	Desc_add_row_values(header_fmt,
		    "             SYSTEM        LEVL        *** (        signficance level <= 0.001)");
		    */
	Desc_add_row_values(header_fmt,
			    "These significance tests are all two-tailed tests with the null hypothesis");
	Desc_add_row_values(header_fmt,
			    "that there is no performance difference between the two systems.          ");
	Desc_add_row_values(header_fmt,
			    "                                                                          ");
	/*	Desc_add_row_values(header_fmt,
		"The first column is an abreviation of the type of significance test   ");
		Desc_add_row_values(header_fmt,
		"employed.                                                             ");
		Desc_add_row_values(header_fmt,
		"                                                                      ");
		*/
	Desc_add_row_values(header_fmt,
			    "The first column indicates if the test finds a significant difference     ");
	Desc_add_row_values(header_fmt,
			    "at the level of p=0.05.  It consists of '~' if no difference is           ");
	Desc_add_row_values(header_fmt,
			    "found at this significance level.  If a difference at this level is       ");
	Desc_add_row_values(header_fmt,
			    "found, this column indicates the system with the higher value on the      ");
	Desc_add_row_values(header_fmt,
			    "performance statistic utilized by the particular test.                    ");
	Desc_add_row_values(header_fmt,
			    "                                                                          ");
	Desc_add_row_values(header_fmt,
			    "The second column specifies the minimum value of p for which the test     ");
	Desc_add_row_values(header_fmt,
			    "finds a significant difference at the level of p.                         ");
	Desc_add_row_values(header_fmt,
			    "                                                                          ");
	Desc_add_row_values(header_fmt,
			    "The third column indicates if the test finds a significant difference     ");
	Desc_add_row_values(header_fmt,
			    "at the level of p=0.001 (\"***\"), at the level of p=0.01, but not          ");
	Desc_add_row_values(header_fmt,
			    "p=0.001 (\"**\"), or at the level of p=0.05, but not p=0.01 (\"*\").          ");
	Desc_add_row_values(header_fmt,
			    "                                                                          ");
	Desc_add_row_values(header_fmt,
			    "A test finds significance at level p if, assuming the null hypothesis,    ");
	Desc_add_row_values(header_fmt,
			    "the probability of the test statistic having a value at least as          ");
	Desc_add_row_values(header_fmt,
			    "extreme as that actually found, is no more than p.                        ");

	Desc_dump_report(1,fp);

        /* print_n_winner_comp_matrix(rank,wins,win_str,num_win,pr_width,fp); */
    }
    if (report)
        print_n_winner_comp_report(rank,wins,win_str,win_str,win_desc,
				   num_win,pr_width,test_name,outdir);

    if (fp != stdout) fclose(fp);
}

void print_n_winner_comp_report(RANK *rank, int ***wins, char **win_ids, char **win_str, char **win_desc, int win_cnt, int page_width, char *testname, char *outdir)
{

    int max_name_len=0, trt1, trt2, trt, t1, t2, i, wid;
    int max_win_ids_len=0, max_str_len=7, max_desc_len=0, pr_width=79;
    char sysname_fmt[40], title_line[200], fname[200], *trtfile;
    FILE *fp;

    /* find the largest treat length */
    for (trt=0;trt< rank->n_trt ;trt++)
      if ((i=strlen(  rank->trt_name [ trt ] )) > max_name_len)
         max_name_len = i;
    /* find the largest id length */
    for (wid=0; wid<win_cnt; wid++){
      if ((i=strlen(win_ids[wid])) > max_win_ids_len)
         max_win_ids_len=i;
      if ((i=strlen(win_str[wid])) > max_str_len)
         max_str_len=i;
      if ((i=strlen(win_desc[wid])) > max_desc_len)
          max_desc_len=i;
    }
    wid=0;
    sprintf(sysname_fmt," %%%ds ",max_name_len);
    pr_width = (max_name_len + 2) * (win_cnt + 1);
    sprintf(title_line,"for the %s Test with other Systems:",testname);
    pr_width = MAX(pr_width,strlen(title_line));
    pr_width = MAX(pr_width,max_desc_len+5+max_str_len);

    for (trt1=0; trt1< rank->n_trt ; trt1++){
	if ((trtfile = strrchr(rank->trt_name[trt1],'/')) == NULL)
	    trtfile = rank->trt_name[trt1];
        sprintf(fname,"%s%s%s.sts", outdir,
		(strcmp(outdir,"")==0) ? "" : "/",
		(strcmp(outdir,"")==0) ? rank->trt_name[trt1] : trtfile);
        if ((fp=fopen(fname,"w")) == (FILE *)0){
            fprintf(stderr,"Warning: Unable to open output file %s, using stdout\n",fname);
            fp=stdout;
        }
        sprintf(title_line,"Summary of Statistical Significance Tests:");
        fprintf(fp,"%s\n",center(title_line,pr_width));
        sprintf(title_line,"Comparing the System");
        fprintf(fp,"%s\n",center(title_line,pr_width));
        sprintf(title_line,"%s",  rank->trt_name [ trt1 ] );
        fprintf(fp,"%s\n",center(title_line,pr_width));
        sprintf(title_line,"for the %s Test with other Systems:",testname);
        fprintf(fp,"%s\n\n",center(title_line,pr_width));
       
        fprintf(fp,sysname_fmt,"");
        for (wid=0; wid<win_cnt; wid++)
            fprintf(fp,sysname_fmt,center(win_str[wid],max_name_len));
        fprintf(fp,"\n");
        fprintf(fp,sysname_fmt,"");
        for (wid=0; wid<win_cnt; wid++){
            fprintf(fp," ");
            for (i=0; i<max_name_len; i++)
                fprintf(fp,"-");
            fprintf(fp," ");
        }
        fprintf(fp,"\n");        
        for (trt2=0; trt2< rank->n_trt ; trt2++){
            if (trt1 == trt2)
                continue;
            fprintf(fp,sysname_fmt,  rank->trt_name [ trt2 ] );
            for (wid=0; wid<win_cnt; wid++){
                if (trt1 < trt2)
                    t1 = trt1, t2 = trt2;
                else
                    t1 = trt2, t2 = trt1;
                if (wins[wid][t1][t2] == NO_DIFF)
                    fprintf(fp,sysname_fmt,center("same",max_name_len));
                else if (wins[wid][t1][t2] < 0)
                    fprintf(fp,sysname_fmt,center(  rank->trt_name [ t1 ] ,max_name_len));
                else
                    fprintf(fp,sysname_fmt,center(  rank->trt_name [ t2 ] ,max_name_len));
            }
            fprintf(fp,"\n");
	}
        fprintf(fp,"\n\n\n");
        fprintf(fp,"%s",center("",(pr_width - (max_desc_len+5+max_str_len)) / 2));
        fprintf(fp,"%s",center("Test Name",max_desc_len));
        fprintf(fp,"%s",center("",5));
        fprintf(fp,"%s\n",center("Abbrev.",max_str_len));

        /* a line of dashes */
        fprintf(fp,"%s",center("",(pr_width - (max_desc_len+5+max_str_len)) / 2));
        for (i=0; i<max_desc_len; i++)
             fprintf(fp,"-");
        fprintf(fp,"%s",center("",5));
        for (i=0; i<max_str_len; i++)
            fprintf(fp,"-");
        fprintf(fp,"\n");
        for (wid=0; wid<win_cnt; wid++){
            fprintf(fp,"%s",center("",(pr_width - (max_desc_len+5+max_str_len)) / 2));
            fprintf(fp,"%s",center(win_desc[wid],max_desc_len));
            fprintf(fp,"%s",center("",5));
            fprintf(fp,"%s\n",center(win_str[wid],max_str_len));
        }
        fprintf(fp,"\n");
        fprintf(fp,"Note: System ID with lower error rate is printed if the Null\n");
        fprintf(fp,"      Hypothesis is rejected at the 95%% Confidence Level,\n");
        fprintf(fp,"      if not, 'same' is printed.\n");
        if (fp != stdout)
            fclose(fp);
        else
            form_feed(fp);
    }   
}

/**********************************************************************/
/*   print to stdout a comparison matrix based on the treatments      */
/**********************************************************************/
void print_trt_comp_matrix_for_RANK_one_winner(int **winner, RANK *rank, char *title, char *formula_str, char *block_id, FILE *fp)
{
    int report_width, max_name_len=0, len_per_trt, trt, trt2, i;
    char sysname_fmt[40], title_line[200];
    int page_width=79;

    /* find the largest treat length */
    for (trt=0;trt< rank->n_trt ;trt++)
      if ((i=strlen(  rank->trt_name [ trt ] )) > max_name_len)
         max_name_len = i;
    sprintf(sysname_fmt," %%%ds |",max_name_len);

    len_per_trt = 3 + max_name_len;
    report_width=( rank->n_trt +1) * len_per_trt + 1; 
     
    fprintf(fp,"%s\n",center(title,page_width));
    sprintf(title_line,"Using the %s Percentage per %s",formula_str,block_id);
    fprintf(fp,"%s\n",center(title_line,page_width));
    fprintf(fp,"%s\n\n",center("as the Comparison Metric",page_width));
                       

    /* begin to print the report matrix */
    /* just a line */
    fprintf(fp,"%s",center("",(page_width-report_width)/2));
    fprintf(fp,"|");
    for (i=0; i<len_per_trt * ( rank->n_trt +1) - 1; i++)
        fprintf(fp,"-");
    fprintf(fp,"|\n");
    /* the treatment titles */
    fprintf(fp,"%s",center("",(page_width-report_width)/2));
    fprintf(fp,"|");
    fprintf(fp,sysname_fmt,"");
    for (trt=0; trt< rank->n_trt ; trt++)
        fprintf(fp,sysname_fmt,  rank->trt_name [ trt ] );
    fprintf(fp,"\n");
    for (trt2=0; trt2< rank->n_trt ; trt2++){
        /* spacer lines */
        fprintf(fp,"%s",center("",(page_width-report_width)/2));
	fprintf(fp,"|");
        for (trt=0; trt< rank->n_trt +1; trt++){
            for (i=0; i<len_per_trt - 1; i++)
                fprintf(fp,"-");
            if (trt <  rank->n_trt )
                fprintf(fp,"+");
        }
        fprintf(fp,"|\n");
        fprintf(fp,"%s",center("",(page_width-report_width)/2));
        fprintf(fp,"|");
        fprintf(fp,sysname_fmt,  rank->trt_name [ trt2 ] );
        for (trt=0; trt< rank->n_trt ; trt++)
            if (trt2 >= trt)
                fprintf(fp,sysname_fmt,"");
            else if (winner[trt2][trt] == NO_DIFF)
                fprintf(fp,sysname_fmt,"same");
            else if (winner[trt2][trt] < 0)
                fprintf(fp,sysname_fmt,  rank->trt_name [ trt2 ] );
            else
                fprintf(fp,sysname_fmt,  rank->trt_name [ trt ] );
        fprintf(fp,"\n");
    }   
    /* just a line */
    fprintf(fp,"%s",center("",(page_width-report_width)/2));
    fprintf(fp,"|");
    for (i=0; i<len_per_trt * ( rank->n_trt +1) - 1; i++)
        fprintf(fp,"-");
    fprintf(fp,"|\n");
    if (fp == stdout) form_feed(fp);
}

int formula_index(char *str)
{
    if (*str == 'R' )
        return('R' );
    if (*str == 'E' )
        return('E' );
    if (*str == 'W' )
        return('W' );
    return('W' );
}

char *formula_str(char *str)
{
    if (*str == 'R' )
        return("Speaker Percent Correctly Recognized" );
    if (*str == 'E' )
        return("Speaker Word Error Rate (%)" );
    if (*str == 'W' )
        return("Speaker Word Accuracy Rate (%)" );
    return("Speaker Word Accuracy Rate (%)" );
}
