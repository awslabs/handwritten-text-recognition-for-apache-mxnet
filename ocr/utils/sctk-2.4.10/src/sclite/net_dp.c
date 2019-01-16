/*******************************************************************/
/*    File: net_dp.c                                               */
/*    Desc: Dynamic programming alignment routines for networks.   */
/*                                                                 */
/*  Revisions:                                                     */
/*     960319 - JGF - Changed Network_dp_align to optionally       */
/*                    output NULLS.                                */
/*     961009 - JGF - added an option to output a Network of the   */
/*                    aligned networks.                            */
/*******************************************************************/


#include "sctk.h"

static NET_ALIGN *net_ali = (NET_ALIGN *)0;

PATH *extract_best_NET_ALIGN_result(NET_ALIGN *, NETWORK *, NETWORK *,float wwd(void *, void *, int (*cmp)(void *, void *)));
void find_minimal_paths(NET_ALIGN *, int, int, int *,float wwd(void *, void *, int (*cmp)(void *, void *)),NETWORK *, PATH *, int *);
void create_alinged_NETWORK(NET_ALIGN *, NETWORK *, NETWORK *, void *, int);

void calc_minimum_insert(NET_ALIGN *, int, int, float *, int *);
void calc_minimum_delete(NET_ALIGN *, int, int, float *, int *);
void calc_minimum_substi(NET_ALIGN *, int, int, float *, int *, int *);

static int setup_NET_ALIGN(NET_ALIGN **,NETWORK *,NETWORK *);
static void destroy_NET_ALIGN(NET_ALIGN **net_ali);
PATH *extract_NET_ALIGN_result(NET_ALIGN *,NETWORK *,NETWORK *, int);

static int add_null_network_heads(NETWORK *, char *);
static void calculate_margin_cells(NET_ALIGN *, NETWORK *net_a,NETWORK *net_b,float wwd(void *, void *, int (*cmp)(void *, void *)));
static void calculate_internal_cells(NET_ALIGN *, NETWORK *,NETWORK *net_b,float wwd(void *, void *, int (*cmp)(void *, void *)));

int find_arcset_id(ARCSET *arcset, ARC *arc, int from){
    int a;
    ARC **arclist;
    if (from == arcset->num) from --;
    for (a=from,arclist = (arcset->arcs) + a; a>=0; a--, arclist--){
	if (*arclist == arc)
	    return(a);
    }

    for (a=from+1; a<arcset->num; a++){
	if (arcset->arcs[a] == arc){
	    fprintf(scfp,"Error: find_arcset_id found the arc after from\n");
	    return(-1);
	}
    }
    return(-1);
}

void add_to_arcset(ARC *arc, void *ptr){
    ARCSET *arcset = (ARCSET *)ptr;
 
    arcset->arcs[arcset->num++] = arc;
}

void calc_minimum_delete(NET_ALIGN *net_ali,int a, int b, float *min_arc_d, int *min_ind_a){
    int arc_a_id;
    ARC *arc_b, *arc_a;
    ARC_LIST_ATOM *arcpa;

    *min_arc_d = 999999999.9;
    *min_ind_a = -1;
    arc_a = net_ali->arcset_a.arcs[a];
    arc_b = net_ali->arcset_b.arcs[b];
    if (db >= 15) {
	printf("    Computing TREE DELETE COST [%d][%d]:\n",a,b);
	printf("        ");
	print_arc(arc_a,0);
	printf("        ");
	print_arc(arc_b,0);
    }

    for (arcpa = arc_a->from_node->in_arcs; arcpa != NULL;arcpa = arcpa->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),arcpa->arc,a)) <0){
	    fprintf(stderr,"Error: find_arcset_id failed in\n");
	    exit(1);
	}
	if (db >= 15) {printf("       look back to cell: %d,%d\n",arc_a_id, b); 
		       printf("           ");
		       print_arc(arcpa->arc,0);
		       printf("           ");
		       print_arc(arc_b,0);}
	if (net_ali->cell[arc_a_id][b].min_d < *min_arc_d){
	    *min_arc_d = net_ali->cell[arc_a_id][b].min_d;
	    *min_ind_a = arc_a_id;
	}
    }
    if (db >= 15) 
	printf("         Min_d=%4.2f   cell [%d][%d]\n",
	       *min_arc_d, *min_ind_a, b);
}
 

void calc_minimum_insert(NET_ALIGN *net_ali,int a, int b, float *min_arc_d, int *min_ind_b){
    int arc_b_id;
    ARC *arc_b, *arc_a;
    ARC_LIST_ATOM *arcpb;

    *min_arc_d = 999999999.9;
    *min_ind_b = -1;
    arc_a = net_ali->arcset_a.arcs[a];
    arc_b = net_ali->arcset_b.arcs[b];
    if (db >= 15) {
	printf("    Computing TREE INSERT COST [%d][%d]:\n",a,b);
	printf("        ");
	print_arc(arc_a,0);
	printf("        ");
	print_arc(arc_b,0);
    }

    for (arcpb = arc_b->from_node->in_arcs; arcpb != NULL;arcpb = arcpb->next){
	if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),arcpb->arc,b)) <0){
	    fprintf(stderr,"Error: find_arcset_id failed\n");
	    exit(1);
	}
	if (db >= 15) {printf("       look back to cell: %d,%d\n",a, arc_b_id); 
		       printf("           ");
		       print_arc(arc_a,0);
		       printf("           ");
		       print_arc(arcpb->arc,0);}
	if (net_ali->cell[a][arc_b_id].min_d < *min_arc_d){
	    *min_arc_d = net_ali->cell[a][arc_b_id].min_d;
	    *min_ind_b = arc_b_id;
	}
    }
    if (db >= 15) 
 	printf("         Min_d=%4.2f   cell [%d][%d]\n",
	       *min_arc_d, a, *min_ind_b);

}

void calc_minimum_substi(NET_ALIGN *net_ali,int a, int b, float *min_arc_d, int *min_ind_a, int *min_ind_b){
    int arc_a_id, arc_b_id;
    ARC *arc_a, *arc_b;
    ARC_LIST_ATOM *arcpa, *arcpb;
	    
    *min_arc_d = 999999999.9; 
    *min_ind_b = *min_ind_a = -1;
    arc_a = net_ali->arcset_a.arcs[a];
    arc_b = net_ali->arcset_b.arcs[b];

    if (db >= 15) { 
	printf("    Computing TREE SUBSITUTION Cost [%d][%d]:\n",a,b);
	printf("        ");print_arc(arc_a,0);printf("        ");
	print_arc(arc_b,0);
    }
    for (arcpa = arc_a->from_node->in_arcs; arcpa != NULL;arcpa = arcpa->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),arcpa->arc,a)) <0){
	    fprintf(stderr,"Error: netA find_arcset_id failed\n");
	    exit(1);
	}
	for (arcpb = arc_b->from_node->in_arcs; arcpb != NULL;
	     arcpb = arcpb->next){
	    if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),
					   arcpb->arc,b)) <0){
		fprintf(stderr,"Error: netB find_arcset_id failed\n");
		exit(1);
	    }
	    if (db >= 15) {printf("       look back to cell: %d,%d\n",
				  arc_a_id, arc_b_id); 
			   printf("          ");print_arc(arcpa->arc,0);
			   printf("          ");print_arc(arcpb->arc,0);}
	    if (net_ali->cell[arc_a_id][arc_b_id].min_d < *min_arc_d){
		*min_arc_d = net_ali->cell[arc_a_id][arc_b_id].min_d;
		*min_ind_b = arc_b_id;
		*min_ind_a = arc_a_id;
	    }
	}
    }
    if (db >= 15) 
	printf("         Min_d=%4.2f   cell [%d][%d]\n\n",
	       *min_arc_d, *min_ind_a, *min_ind_b);
}


static int add_null_network_heads(NETWORK *net, char *proc){
    if (Network_add_arc_to_head(net,(void *)0) > 0){
	fprintf(stderr,"%s: Unable to add (void *)0 to Net A\n",proc);
	return(1);
    }
    return(0);
}

static void calculate_margin_cells(NET_ALIGN *net_ali, NETWORK *net_a,NETWORK *net_b,float wwd(void *, void *, int (*cmp)(void *, void *))){
    int a,b;
    float min_arc_d;
    int min_ind_a, min_ind_b;

    /* Pre-Calculate the margin Cells */
    for (a=0,b=1; b<net_ali->arcset_b.num; b++){
	calc_minimum_insert(net_ali,0,b,&min_arc_d,&min_ind_b);

	net_ali->cell[0][b].min_d  = min_arc_d + 
	    wwd(NULL_WORD,net_ali->arcset_b.arcs[b]->data,
		net_a->arc_func.equal);
	net_ali->cell[0][b].back_a = 0;
	net_ali->cell[0][b].back_b = min_ind_b;
	if (db >= 15) 
	    printf("            Min_d=%4.2f   back_a=%d  back_b=%d\n\n",
		   net_ali->cell[a][b].min_d,
		   net_ali->cell[a][b].back_a,
		   net_ali->cell[a][b].back_b);
    }

    for (b=0,a=1; a<net_ali->arcset_a.num; a++){
	calc_minimum_delete(net_ali,a,0,&min_arc_d,&min_ind_a);
	net_ali->cell[a][0].min_d  = min_arc_d + 
	    wwd(net_ali->arcset_a.arcs[a]->data,NULL_WORD,
		net_a->arc_func.equal);
	net_ali->cell[a][0].back_a = min_ind_a;
	net_ali->cell[a][0].back_b = 0;
	if (db >= 15) 
	    printf("            Min_d=%4.2f   back_a=%d  back_b=%d\n\n",
		   net_ali->cell[a][b].min_d,
		   net_ali->cell[a][b].back_a,
		   net_ali->cell[a][b].back_b);
    }
}

static void calculate_internal_cells(NET_ALIGN *net_ali, NETWORK *net_a,NETWORK *net_b,float wwd(void *, void *, int (*cmp)(void *, void *))){    
    int a,b;

    for (a=1; a<net_ali->arcset_a.num; a++){
	for (b=1; b<net_ali->arcset_b.num; b++){ 
	    /* for (b=1; b<2; b++){ */
	    int ins_min_ind_b, del_min_ind_a, sub_min_ind_a, sub_min_ind_b;
	    float ins_min_d, del_min_d, sub_min_d;

	    if (db >= 15){
		printf("Computing minimized cost of [%d][%d]:\n",a,b);
		print_arc(net_ali->arcset_a.arcs[a],0);
		print_arc(net_ali->arcset_b.arcs[b],0);
	    }

	    calc_minimum_insert(net_ali,a,b,&ins_min_d,&ins_min_ind_b);
	    calc_minimum_delete(net_ali,a,b,&del_min_d,&del_min_ind_a);
	    calc_minimum_substi(net_ali,a,b,&sub_min_d,&sub_min_ind_a,
				&sub_min_ind_b);
	    if (db >= 15)
		printf("       Tree:     Ins = %f   Del = %f   Sub = %f\n",
		       ins_min_d, del_min_d, sub_min_d);

	    sub_min_d += wwd(net_ali->arcset_a.arcs[a]->data,
			     net_ali->arcset_b.arcs[b]->data,
			     net_a->arc_func.equal);
	    del_min_d += wwd(net_ali->arcset_a.arcs[a]->data,0,
			     net_a->arc_func.equal);
	    ins_min_d += wwd(0,net_ali->arcset_b.arcs[b]->data,
			     net_a->arc_func.equal);

	    if (db >= 15)
		printf("       Tree+wwd: Ins = %f   Del = %f   Sub = %f\n",
		       ins_min_d, del_min_d, sub_min_d);

	    if (sub_min_d <= del_min_d && sub_min_d <= ins_min_d){
		net_ali->cell[a][b].min_d  = sub_min_d;
		net_ali->cell[a][b].back_a = sub_min_ind_a;
		net_ali->cell[a][b].back_b = sub_min_ind_b;
	    } else if (del_min_d < ins_min_d) {
		net_ali->cell[a][b].min_d  = del_min_d;
		net_ali->cell[a][b].back_a = del_min_ind_a;
		net_ali->cell[a][b].back_b = b;
	    } else {
		net_ali->cell[a][b].min_d  = ins_min_d;
		net_ali->cell[a][b].back_a = a;
		net_ali->cell[a][b].back_b = ins_min_ind_b;
	    }		

	    if (db >= 15) 
		printf("       Min_d=%4.2f   back_a=%d  back_b=%d\n",
		       net_ali->cell[a][b].min_d,
		       net_ali->cell[a][b].back_a,
		       net_ali->cell[a][b].back_b);
	    
	    if (db >= 15) printf("\n");	   
	}
    }
}

void cleanup_NET_ALIGN(void){
    char *proc="cleanup_NET_ALIGN";

    if (db >= 1) printf("Entering %s:\n",proc);
    Network_dpalign((NETWORK *)0,(NETWORK *)0, 0, (PATH **)0, 0);
}

static int setup_NET_ALIGN(NET_ALIGN **net_ali,NETWORK *net_a,NETWORK *net_b){
    char *proc="setup_NET_ALIGN";

    /* make sure there is enough room in the data structure */
    if (((*net_ali) = alloc_NET_ALIGN((*net_ali),net_a->arc_count,
				   net_b->arc_count)) == NET_ALIGN_NULL){
	fprintf(stderr,"%s: alloc_NET_ALIGN failed\n",proc);
	return(1);
    }
    /* link the arcs into the structure */

    Network_traverse(net_a,0,0,add_to_arcset,(void *)&((*net_ali)->arcset_a),
		     NT_CA_For + NT_Inorder);
    Network_traverse(net_b,0,0,add_to_arcset,(void *)&((*net_ali)->arcset_b),
		     NT_CA_For + NT_Inorder);

    /* initialize the [0][0] Cell */
    (*net_ali)->cell[0][0].back_a = -1;
    (*net_ali)->cell[0][0].back_b = -1;
    (*net_ali)->cell[0][0].min_d  = 0.0;

    if (db >= 15) dump_NET_ALIGN((*net_ali),stdout);
    return(0);
}

PATH *extract_NET_ALIGN_result(NET_ALIGN *net_ali,NETWORK *net_a,NETWORK *net_b, int include_nulls){
    PATH *t_path;
    ARC_LIST_ATOM *arcpa, *arcpb;
    int back_path, min_a, min_b, hops;
    int a, b;
    int arc_a_id, arc_b_id;
    float min_arc_d;

    /*********************************/
    /*     Beginning Back Trace      */

    /* Find the minimal network termination node, */
    /* !!!!  It's not just max x,y any more   !!!!*/
    min_arc_d = 999999.9;
    a = b = -1;
    for (arcpa = net_a->stop_node->in_arcs; arcpa != NULL;arcpa =arcpa->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),arcpa->arc,
				       net_ali->arcset_a.num)) <0){
	    fprintf(stderr,"Error: find_arcset_id for Net A failed\n");
	    exit(1);
	}
	for (arcpb = net_b->stop_node->in_arcs; arcpb != NULL;
	     arcpb = arcpb->next){
	    if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),arcpb->arc,
					   net_ali->arcset_b.num)) < 0){

		fprintf(stderr,"Error: find_arcset_id for Net B failed\n");
		exit(1);
	    }
	    if (min_arc_d > net_ali->cell[arc_a_id][arc_b_id].min_d){
		min_arc_d = net_ali->cell[arc_a_id][arc_b_id].min_d;
		a = arc_a_id; b = arc_b_id;
	    }
	}
    }
    min_a = a;
    min_b = b;

    {if (getenv("DBL") != NULL) { db=(int)atof(getenv("DBL")); }}

    if (db >= 25) dump_NET_ALIGN(net_ali,stdout);

    /* first do a back trace, counting the hops */
    a = min_a ; b = min_b; hops = 0;
    while (a > 0 || b > 0){
	int la, lb;
	if (db >= 15)
	    printf("Counting hops  a=%d b=%d   %d,%d\n",a,b,
		   net_ali->cell[a][b].back_b,net_ali->cell[a][b].back_a);

	hops ++;
	/* check for an insertion of "@", and ignore it */
	if (b == net_ali->cell[a][b].back_b) {
	    if (!include_nulls &&
		net_a->arc_func.is_null_alt(net_ali->arcset_a.arcs[a]->data)){
		if (db >= 15) printf("Insertion of nulls\n");
		hops --;
	    }
	} else if (a == net_ali->cell[a][b].back_a) {
	    /* check for a deletion of "@", and ignore it */
	    if (!include_nulls &&
		net_b->arc_func.is_null_alt(net_ali->arcset_b.arcs[b]->data)){
		if (db >= 15) printf("deletion of null\n");
		hops --;
	    }
	} else {
	    /* check for a substitution of "@", and ignore it */
	    if (!include_nulls &&
		(net_a->arc_func.is_null_alt(net_ali->arcset_a.arcs[a]->data))&&
		(net_b->arc_func.is_null_alt(net_ali->arcset_b.arcs[b]->data))){
		printf("substition of to nulls\n");
		hops --;
	    }
	}
	la = a;	lb = b;
	a = net_ali->cell[la][lb].back_a;
	b = net_ali->cell[la][lb].back_b;
    }
    a = min_a ; b = min_b;
    if (db >= 10)
	printf("Minimal Start point: D(%d,%d) = %f  hops(%d)\n",
	       a,b,min_arc_d,hops);
    
    t_path = PATH_alloc(hops);
    back_path = hops - 1;

    a = min_a ; b = min_b;
    while (a > 0 || b > 0){
	int la, lb;
	enum edits back;
	back = (a == net_ali->cell[a][b].back_a) ? INS : 
	            ((b == net_ali->cell[a][b].back_b) ? DEL : SUB);
	switch (back){
	  case DEL:
	    if (include_nulls ||
		!net_a->arc_func.is_null_alt(net_ali->arcset_a.arcs[a]->data)){
	      if (! net_a->arc_func.is_opt_del(net_ali->arcset_a.arcs[a]->data)){
		t_path->pset[back_path].a_ptr = net_ali->arcset_a.arcs[a]->data;
		net_a->arc_func.use_count(net_ali->arcset_a.arcs[a]->data,1);
		t_path->pset[back_path].b_ptr = (void *)0;
		if (db >= 15){
		  printf("Del  %d %d  \n",a,b);
		  print_arc(net_ali->arcset_a.arcs[a],0);
		  printf("     *\n");
		}
		t_path->pset[back_path].eval = P_DEL;
	      } else { 
		/**** OPTIONALLY deletable, make it correct */
		t_path->pset[back_path].a_ptr = net_ali->arcset_a.arcs[a]->data;
		net_a->arc_func.use_count(net_ali->arcset_a.arcs[a]->data,1);
		t_path->pset[back_path].b_ptr = 
		  net_a->arc_func.make_empty((void*)0);
		if (db >= 15){
		  printf("Corr Del  %d %d  \n",a,b);
		  print_arc(net_ali->arcset_a.arcs[a],0);
		}
		t_path->pset[back_path].eval = P_CORR;
	      }
	      
	      --back_path; 	t_path->num ++;
	    }
	    break;
	  case SUB:
	    if(include_nulls ||
	       !(net_b->arc_func.is_null_alt(net_ali->arcset_a.arcs[a]->data)&&
		 net_b->arc_func.is_null_alt(net_ali->arcset_b.arcs[b]->data))){
		t_path->pset[back_path].a_ptr = net_ali->arcset_a.arcs[a]->data;
		net_a->arc_func.use_count(net_ali->arcset_a.arcs[a]->data,1);
		t_path->pset[back_path].b_ptr = net_ali->arcset_b.arcs[b]->data;
		net_a->arc_func.use_count(net_ali->arcset_b.arcs[b]->data,1);
		if (db >= 15){	
		    printf("Sub  %d %d  \n",a,b);
		    print_arc(net_ali->arcset_a.arcs[a],0);
		    print_arc(net_ali->arcset_b.arcs[b],0);
		}
		if (net_a->arc_func.equal(t_path->pset[back_path].a_ptr,
					  t_path->pset[back_path].b_ptr) == 0)
		    t_path->pset[back_path].eval = P_CORR;
		else
		    t_path->pset[back_path].eval = P_SUB;
		--back_path; 	t_path->num ++;
	    }
	    break;
	  case INS:
	    if (include_nulls  ||
		!net_b->arc_func.is_null_alt(net_ali->arcset_b.arcs[b]->data)){
	      if (! net_b->arc_func.is_opt_del(net_ali->arcset_b.arcs[b]->data)){
		t_path->pset[back_path].a_ptr = (void *)0;
		t_path->pset[back_path].b_ptr = net_ali->arcset_b.arcs[b]->data;
		net_a->arc_func.use_count(net_ali->arcset_b.arcs[b]->data,1);
	    
		if (db >= 15){
		    printf("INS  %d %d \n",a,b);
		    printf("     *\n");
		    print_arc(net_ali->arcset_b.arcs[b],0);
		}
		t_path->pset[back_path].eval = P_INS;
	      } else {
		t_path->pset[back_path].a_ptr = 
		  net_b->arc_func.make_empty((void*)0);
		t_path->pset[back_path].b_ptr = net_ali->arcset_b.arcs[b]->data;
		net_a->arc_func.use_count(net_ali->arcset_b.arcs[b]->data,1);
	    
		if (db >= 15){
		    printf("CORR INS  %d %d \n",a,b);
		    printf("     *\n");
		    print_arc(net_ali->arcset_b.arcs[b],0);
		}
		t_path->pset[back_path].eval = P_CORR;
	      }
		--back_path; 	t_path->num ++;
	    }
	}
	la = a;
	lb = b;
	a = net_ali->cell[la][lb].back_a;
	b = net_ali->cell[la][lb].back_b;
    }
    return(t_path);
}

int Network_dpalign(NETWORK *net_a,NETWORK *net_b,float wwd(void *, void *, int (*cmp)(void *, void *)), PATH **out_path, int include_nulls){
    char *proc = "Network_dpalign";
    int prev_db;

    prev_db = db;
    if (getenv("NET_DP_DBL") != NULL) { db=(int)atof(getenv("NET_DP_DBL")); }

    if (db >= 1) printf("Entering %s:\n%s",proc,proc);

    if (net_a == (NETWORK *)0 && net_b == (NETWORK *)0){
        if (db >= 1) printf("destroying %s's data\n",proc);
        destroy_NET_ALIGN(&net_ali);
	return (1);
    }

    /* Add Null words to the beginning of Net A and B */
    if (add_null_network_heads(net_a,proc) != 0) return(1);
    if (add_null_network_heads(net_b,proc) != 0) return(1);

    if (setup_NET_ALIGN(&net_ali,net_a,net_b) != 0)
	return(1);

    calculate_margin_cells(net_ali,net_a,net_b,wwd);
    calculate_internal_cells(net_ali,net_a,net_b,wwd);

    if (db >= 15) dump_NET_ALIGN(net_ali,stdout);

    *out_path = extract_NET_ALIGN_result(net_ali, net_a, net_b, include_nulls);

    db=prev_db;
    
    return(0);
}

int Network_dpalign_n_networks(NETWORK **in_nets, int n_nets, float wwd(void *, void *, int (*cmp)(void *, void *)), NETWORK **out_net, void *null_alt){
    char *proc = "Network_dpalign_n_networks"; 
    NETWORK **copy_in_nets;
    int new_db = db, n, rtn=0;

    if (getenv("NET_DP_DBL") != NULL) 
        new_db =(int)atof(getenv("NET_DP_DBL"));

    alloc_singZ(copy_in_nets,n_nets,NETWORK *,(NETWORK *)0);

    if (new_db >= 1) printf("Entering %s:\n",proc);

    /* make a copy of the networks and align to those.   This allows */
    /* the return network function to work from the copy to make the */
    /* output network */
    for (n=0; n<n_nets; n++){
        if ((copy_in_nets[n] = Network_copy(in_nets[n])) == (NETWORK *)0){
	    fprintf(scfp,"Error: %s unable to copy networks\n",proc);
	    goto ERROR;
	}
	if (add_null_network_heads(copy_in_nets[n],proc) != 0) {
	    goto ERROR;
	}
	if (new_db >= 10) {
	    printf("%s: ----------------------------------------\n",proc);
	    printf("%s: Input network %d\n",proc,n);
	    if (n == 0) printf("%s: STARTING OUTPUT NETOWRK\n",proc);
	    Network_traverse(copy_in_nets[n],0,0,print_arc,0,
			     NT_CA_For+NT_Verbose);
	}
    }
    if (new_db >= 10) 
        printf("%s: ----------------------------------------\n",proc);

    /* now loop through the networks, aligning them, then merging them */
    for (n=1; n<n_nets; n++){
        if (setup_NET_ALIGN(&net_ali,copy_in_nets[0],copy_in_nets[n]) != 0){
	    goto ERROR;
	}

	calculate_margin_cells(net_ali,copy_in_nets[0],copy_in_nets[n],wwd);
	calculate_internal_cells(net_ali,copy_in_nets[0],copy_in_nets[n],wwd);

	if (new_db >= 15) dump_NET_ALIGN(net_ali,stdout);

        create_alinged_NETWORK(net_ali,copy_in_nets[0],copy_in_nets[n],
			       null_alt,n);
	if (new_db >= 10) {
	    printf("%s: ----------------------------------------\n",proc);
	    printf("%s: Output Network %d\n",proc,n);
	    Network_traverse(copy_in_nets[0],0,0,print_arc,0,
			     NT_CA_For+NT_Verbose);
	}
    }
    Network_delete_null_arcs(copy_in_nets[0]);
    *out_net = copy_in_nets[0];
    rtn = 0;
    goto CLEANUP;

  ERROR:
    rtn = 1;

  CLEANUP:
    for (n=1; n<n_nets; n++)
        Network_destroy(copy_in_nets[n]);
    free_singarr(copy_in_nets,NETWORK *);
    
  
    return(rtn);
}

/*** THIS CODE MAKES THE ASSUMPTION THE net_b network is a linear network */
/*** without branching */

void create_alinged_NETWORK(NET_ALIGN *net_ali, NETWORK *net_a, NETWORK *net_b, void *null_alt, int iteration){
    char *proc = "create_alinged_NETWORK";
    ARC_LIST_ATOM *arcpa, *arcpb;
    void *copy_b_arc_data;
    int min_a, min_b;
    int a, b, i;
    int arc_a_id, arc_b_id;
    float min_arc_d;
    int new_db = 0;


    if (getenv("ALIGN_NET_DBL") != NULL) 
        new_db=(int)atof(getenv("ALIGN_NET_DBL"));

    if (new_db >= 10) printf("Entering %s:\n",proc);
    
    /* Find the minimal network termination node, */
    /* !!!!  It's not just max x,y any more   !!!!*/
    min_arc_d = 999999.9;
    a = b = -1;
    for (arcpa = net_a->stop_node->in_arcs; arcpa != NULL;arcpa =arcpa->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),arcpa->arc,
				       net_ali->arcset_a.num)) <0){
	    fprintf(stderr,"Error: find_arcset_id for Net A failed\n");
	    exit(1);
	}
	for (arcpb = net_b->stop_node->in_arcs; arcpb != NULL;
	     arcpb = arcpb->next){
	    if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),arcpb->arc,
					   net_ali->arcset_b.num)) < 0){

		fprintf(stderr,"Error: find_arcset_id for Net B failed\n");
		exit(1);
	    }
	    if (min_arc_d > net_ali->cell[arc_a_id][arc_b_id].min_d){
		min_arc_d = net_ali->cell[arc_a_id][arc_b_id].min_d;
		a = arc_a_id; b = arc_b_id;
	    }
	}
    }
    min_a = a;
    min_b = b;

    /*******************************************************************/
    /*     Beginning the Back Trace copying the B-arcs onto A_net      */

    if (new_db >= 10)
        printf("Minimal Start point: D(%d,%d) = %f\n",min_a,min_b,min_arc_d);

    a = min_a ; b = min_b;
    while (a > 0 || b > 0){
	int la = a, lb = b;
	ARC *a_arc, *b_arc;
	enum edits back;

	back = (a == net_ali->cell[a][b].back_a) ? INS : 
	            ((b == net_ali->cell[a][b].back_b) ? DEL : SUB);
	b_arc = net_ali->arcset_b.arcs[b];
	a_arc = net_ali->arcset_a.arcs[a];

	switch (back){
	  case SUB:
 	    /* insert a copy the B_arc onto the from_node and to_node of the 
	       A_arc */
	    if (new_db >= 10){
	        printf("%s: SUB\n",proc);
		printf("%s:   Insert copy of B_arc",proc);
		print_arc(b_arc,stdout);
		printf("%s:   between A_arcs(from)",proc);
		print_node(a_arc->from_node,stdout);
		printf("%s:           A_arcs(to)  ",proc);
		print_node(a_arc->to_node,stdout);
	    }
	    if (NETWORK_insert_arc_between_node(net_a, a_arc->from_node,
						a_arc->to_node,
				    b_arc->net->arc_func.copy(b_arc->data)) != 0)
	        goto ERROR;

	    break;
	  case DEL:
	    /* insert a @ arc onto the from_node and to_node of the A_arc */
	    if (new_db >= 10){
	        printf("%s: DEL\n",proc);
		printf("%s:   Insert copy of null_alt\n",proc);
		printf("%s:   between A_arcs(from)",proc);
		print_node(a_arc->from_node,stdout);
		printf("%s:           A_arcs(to)  ",proc);
		print_node(a_arc->to_node,stdout);
	    }
	    if (NETWORK_insert_arc_between_node(net_a, a_arc->from_node,
				    a_arc->to_node,
 				    a_arc->net->arc_func.copy(null_alt)) != 0)
	        goto ERROR;

	    break;
  	  case INS:
	    /* insert a subnetwork of { @ / B_arc } onto the from_node    */
	    /* and to_node of the A_arc,  There will be 'iteration' @'s   */
	    /* inside this network */

	    if (new_db >= 10){
	        printf("%s: INS\n",proc);
		printf("%s:   Insert a subnet with %d null arcs and a "
		       "copy of b_arc\n",proc,iteration);
		printf("%s:  Insert copy of B_arc",proc);
		print_arc(b_arc,stdout);
		printf("%s:  At A_arcs(to_node)  ",proc);
		print_node(a_arc->to_node,stdout);
	    }
	    /* first, insert the copy of b_arc's data into the A_net */
	    copy_b_arc_data = a_arc->net->arc_func.copy(b_arc->data);
	    if (NETWORK_insert_arc_before_node(a_arc->net,
					       a_arc->to_node,
					       copy_b_arc_data) != 0)
	        goto ERROR;
	    /* Find, in A_net, the new arc */
	    for (arcpa = a_arc->to_node->out_arcs; arcpa != NULL;
		 arcpa = arcpa->next)
	        if (arcpa->arc->data == copy_b_arc_data){
		    break;
		}
	    if (arcpa == NULL){
	        fprintf(scfp,"Internal Error\n");
		goto ERROR;
	    }
	    for (i=0; i<iteration; i++)
	      if (NETWORK_insert_arc_between_node(arcpa->arc->net,
						  arcpa->arc->from_node,
						  arcpa->arc->to_node,
				       a_arc->net->arc_func.copy(null_alt)) != 0)
	        goto ERROR;
	    break;
	}
	la = a;	                           lb = b;
	a = net_ali->cell[la][lb].back_a;  b = net_ali->cell[la][lb].back_b;
    }
    goto CLEANUP;

  ERROR:
    printf("ERROR OCCURED\n");
  CLEANUP:
    return;
}

void dump_NET_ALIGN(NET_ALIGN *net_ali,FILE *fp){
    int aa, bb;
    printf("Dump of net_ali:\n");
    
    printf("\nArc List from Network A\n");
    for (aa=0; aa<net_ali->arcset_a.num; aa++){
	printf("  %2d: ",aa);
	print_arc(net_ali->arcset_a.arcs[aa],0);
    }
      
    printf("\nArc List from Network B\n");
    for (bb=0; bb<net_ali->arcset_b.num; bb++){
	printf("  %2d: ",bb);
	print_arc(net_ali->arcset_b.arcs[bb],0);
    }
    
    printf("\n       ");
    for (bb=0; bb<net_ali->arcset_b.num; bb++)
	printf("   %3d   ",bb);
    printf("\n");
    for (aa=0; aa<net_ali->arcset_a.num; aa++) {
	printf("  %3d  ",aa);
	for (bb=0; bb<net_ali->arcset_b.num; bb++)
	    printf("D%6.3f  ",net_ali->cell[aa][bb].min_d);
	printf("\n");
	printf("       ");
	for (bb=0; bb<net_ali->arcset_b.num; bb++){
	    if (aa == net_ali->cell[aa][bb].back_a)
		printf("I=");
	    else if (bb == net_ali->cell[aa][bb].back_b)
		printf("D=");
	    else
		printf("S=");
	    printf("%2d,%2d  ",net_ali->cell[aa][bb].back_a,
		   net_ali->cell[aa][bb].back_b);
	}
	printf("\n\n");
    }
}

void alloc_ARCSET(ARCSET *arcset, int n){
    arcset->max = n;
    arcset->num = 0;
    alloc_singarr(arcset->arcs,arcset->max,ARC *);
}

void free_ARCSET(ARCSET *arcset){
    arcset->max = 0;
    arcset->num = 0;
    free_singarr(arcset->arcs,ARC *);
}

static void destroy_NET_ALIGN(NET_ALIGN **net_ali){
    char *proc = "destroy_NET_ALIGN";

    if (db >= 1) printf("Entering %s:\n",proc);

    /* free the cell array */
    free_2dimarr((*net_ali)->cell,(*net_ali)->arcset_a.max,CELL);

    /* the arcsets */
    free_ARCSET(&((*net_ali)->arcset_a));
    free_ARCSET(&((*net_ali)->arcset_b));

    /* the structure */
    free_singarr((*net_ali),NET_ALIGN);

    *net_ali = (NET_ALIGN *)0;
}

NET_ALIGN *alloc_NET_ALIGN(NET_ALIGN *net_ali, int na, int nb){
    char *proc = "alloc_NET_ALIGN";
    int alloc_data, init_size=100;
    
    if (db >= 1) printf("Entering %s:\n",proc);
    if (na < init_size) na = init_size;
    if (nb < init_size) nb = init_size;
    if (db >= 2) 
	printf("NET_ALIGN: %p  na=%d  nb=%d\n",net_ali,na,nb);

    if (net_ali == NET_ALIGN_NULL) {
	/* do the first initialization */
	alloc_singarr(net_ali,1,NET_ALIGN);
	alloc_data = T;
    } else if (net_ali->arcset_a.max < na || net_ali->arcset_b.max < nb) {
	/* Then ... Re-allocate the structure */
	if (db >= 5) 
	    printf("%s: Free-ing previous data structure\n",proc);

	/* free the old memory */
	free_2dimarr(net_ali->cell,net_ali->arcset_a.max,CELL);
	free_ARCSET(&(net_ali->arcset_a));
	free_ARCSET(&(net_ali->arcset_b));

	alloc_data = T;
    } else
	alloc_data = F;

    if (alloc_data){
	alloc_ARCSET(&(net_ali->arcset_a),(int)(na * 1.05));
	alloc_ARCSET(&(net_ali->arcset_b),(int)(nb * 1.05));
	
	alloc_2dimarr(net_ali->cell,
		      net_ali->arcset_a.max,net_ali->arcset_b.max,CELL);
    }

    /* erase the data cells and arcs */
#ifdef full
    clear_sing(net_ali->arcset_a.arcs,net_ali->arcset_a.max,(ARC *)0);
    clear_sing(net_ali->arcset_b.arcs,net_ali->arcset_b.max,(ARC *)0);
    for (a=0; a<net_ali->arcset_a.max; a++)
	for (b=0; b<net_ali->arcset_b.max; b++){
	    c = &(net_ali->cell[a][b]);
	    c->min_d = 0.0;
	    c->back_a = c->back_b = 0;
	}    
#endif
    net_ali->arcset_a.num = 0;
    net_ali->arcset_b.num = 0;

    return(net_ali);
}


/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/
/*************************************************************************/

PATH *extract_best_NET_ALIGN_result(NET_ALIGN *net_ali,NETWORK *net_a,NETWORK *net_b,float wwd(void *, void *, int (*cmp)(void *, void *))){
    ARC_LIST_ATOM *arcpa, *arcpb;
    int min_a, min_b, hops=0;
    int a, b;
    int arc_a_id, arc_b_id;
    float min_arc_d;
    int r=0, count=0;
    PATH *path;


    path = PATH_alloc(100);

    /*********************************/
    /*     Beginning Back Trace      */

    /* Find the minimal network termination node, */
    /* !!!!  It's not just max x,y any more   !!!!*/
    min_arc_d = 999999.9;
    a = b = -1;
    for (arcpa = net_a->stop_node->in_arcs; arcpa != NULL;arcpa =arcpa->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),arcpa->arc,
				       net_ali->arcset_a.num)) <0){
	    fprintf(stderr,"Error: find_arcset_id for Net A failed\n");
	    exit(1);
	}
	for (arcpb = net_b->stop_node->in_arcs; arcpb != NULL;
	     arcpb = arcpb->next){
	    if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),arcpb->arc,
					   net_ali->arcset_b.num)) < 0){
		fprintf(stderr,"Error: find_arcset_id for Net B failed\n");
		exit(1);
	    }
	    if (min_arc_d > net_ali->cell[arc_a_id][arc_b_id].min_d){
		min_arc_d = net_ali->cell[arc_a_id][arc_b_id].min_d;
		a = arc_a_id; b = arc_b_id;
	    }
	}
    }
    min_a = a;
    min_b = b;

/*
    dump_NET_ALIGN(net_ali,stdout);
*/

    for (arcpa = net_a->stop_node->in_arcs; arcpa != NULL;arcpa =arcpa->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),arcpa->arc,
				       net_ali->arcset_a.num)) <0){
	    fprintf(stderr,"Error: find_arcset_id for Net A failed\n");
	    exit(1);
	}
	for (arcpb = net_b->stop_node->in_arcs; arcpb != NULL;
	     arcpb = arcpb->next){
	    if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),arcpb->arc,
					   net_ali->arcset_b.num)) < 0){
		fprintf(stderr,"Error: find_arcset_id for Net B failed\n");
		exit(1);
	    }
	    if (min_arc_d == net_ali->cell[arc_a_id][arc_b_id].min_d){
		printf("Minimal Start point: D(%d,%d) = %f  hops(%d)\n",
		       arc_a_id,arc_b_id,min_arc_d,hops);
		find_minimal_paths(net_ali,arc_a_id,arc_b_id,
				   &r,wwd, net_a, path,&count);
		printf("Count = %d\n\n\n",count);
		(count) = 0;
	    }
	}
    }
      

    db = 0;
    return((PATH *)0);
}

void find_minimal_paths(NET_ALIGN *net_ali, int a, int b, int *rlev,float wwd(void *, void *, int (*cmp)(void *, void *)),NETWORK *net_a, PATH *path,int *count)
{

    ARC_LIST_ATOM *oarc_a, *oarc_b;
    ARC *arc_a, *arc_b;
    int la, lb, cost;
    int arc_a_id, arc_b_id,d;
    int db = 0;
    char buf[100];

    sprintf(buf,rsprintf("%%0%ds",*rlev),"");
    if (a == 0 && b == 0){
	if (db > 5) printf("END recursion\n");
	if (db > 2) {
	    printf("Count = %d\n",*count);
	    PATH_print(path,stdout,100);
	}
	(*count)++;
	return;
    }

    arc_a = net_ali->arcset_a.arcs[a];
    arc_b = net_ali->arcset_b.arcs[b];
    cost  = net_ali->cell[a][b].min_d;

    if (db > 5){
	printf("%sCurrent position: D(%d,%d) = %d\n",buf,a,b,cost);
	printf("%s    ",buf); print_arc(arc_a,0);
	printf("%s    ",buf); print_arc(arc_b,0);
    }

    /* loop through the net A in_arcs, seaching for possible paths */
    for (oarc_a = arc_a->from_node->in_arcs; oarc_a != NULL;
	 oarc_a=oarc_a->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),oarc_a->arc,
				       a)) <0){
	    fprintf(stderr,"Error: find_arcset_id for Net A failed\n");
	    exit(1);
	}
	if (db > 5) {
	    printf("%s    Checking deletion arcs:\n",buf);
	    printf("%s        A:",buf); print_arc(oarc_a->arc,0);
	}
	if (net_ali->cell[arc_a_id][b].min_d <= cost && 
	    (wwd(arc_a->data,NULL_WORD,net_a->arc_func.equal) +
	     net_ali->cell[arc_a_id][b].min_d) <= cost){
	    if (db > 5) printf("%s            Possible PATH\n",buf);
	    if (db > 5) printf("%s            *****  Delete %s\n",buf,
			       ((WORD*)(arc_b->data))->value);
	    *rlev += 2;
	    PATH_append(path,arc_a->data,0,P_DEL);
	    find_minimal_paths(net_ali,arc_a_id,b,rlev,wwd,net_a,path,count);
	    PATH_remove(path);
	    *rlev -= 2;
	}
    }
    for (oarc_b = arc_b->from_node->in_arcs; oarc_b != NULL;
	 oarc_b = oarc_b->next){
	if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),oarc_b->arc,
				       b)) < 0){
	    fprintf(stderr,"Error: find_arcset_id for Net B failed\n");
	    exit(1);
	}
	if (db > 5) {
	    printf("%s    Checking insertion arcs:\n",buf);
	    printf("%s        B:",buf); print_arc(oarc_b->arc,0);
	}
	if (net_ali->cell[a][arc_b_id].min_d <= cost &&
	    (wwd(NULL_WORD,arc_b->data,net_a->arc_func.equal) +
	     net_ali->cell[a][arc_b_id].min_d) <= cost){
	    if (db > 5) printf("%s            Possible PATH\n",buf);
	    if (db > 5) printf("%s            *****  Insert %s\n",buf,
			       ((WORD*)(arc_b->data))->value);
	    *rlev += 2;
	    PATH_append(path,0,arc_b->data,P_INS);
	    find_minimal_paths(net_ali,a,arc_b_id,rlev,wwd,net_a,path,count);
	    PATH_remove(path);
	    *rlev -= 2;
	}
    }


    for (oarc_a = arc_a->from_node->in_arcs; oarc_a != NULL;
	 oarc_a=oarc_a->next){
	if ((arc_a_id = find_arcset_id(&(net_ali->arcset_a),oarc_a->arc,
				       a)) <0){
	    fprintf(stderr,"Error: find_arcset_id for Net A failed\n");
	    exit(1);
	}
	for (oarc_b = arc_b->from_node->in_arcs; oarc_b != NULL;
	     oarc_b = oarc_b->next){
	    if ((arc_b_id = find_arcset_id(&(net_ali->arcset_b),oarc_b->arc,
					   b)) < 0){
		fprintf(stderr,"Error: find_arcset_id for Net B failed\n");
		exit(1);
	    }
	    if (db > 5) {
		printf("%s    Checking substitution arcs: (%d,%d)\n",buf,
		       arc_a_id,arc_b_id);
		printf("%s        A:",buf); print_arc(oarc_a->arc,0);
		printf("%s        B:",buf); print_arc(oarc_b->arc,0);
	    }
	    if (net_ali->cell[arc_a_id][arc_b_id].min_d <= cost &&
		((d=wwd(arc_a->data,arc_b->data,net_a->arc_func.equal)) + 
		 net_ali->cell[arc_a_id][arc_b_id].min_d) <= cost){
		if (db > 5){ 
		    printf("%s            Possible PATH\n",buf);
		    printf("%s            *****  Substitute %s and %s\n",buf,
			   ((WORD*)(arc_a->data))->value,
			   ((WORD*)(arc_b->data))->value);
		}
		*rlev += 2;
		PATH_append(path,arc_a->data,arc_b->data,(d==0)?P_CORR:P_SUB);
		find_minimal_paths(net_ali,arc_a_id,arc_b_id,rlev,wwd,net_a,path,count);
		PATH_remove(path);
		*rlev -= 2;
	    }
	}	
    }
    la = a;	lb = b;
    a = net_ali->cell[la][lb].back_a;
    b = net_ali->cell[la][lb].back_b;
}
