#include "sctk.h"

static void expand_alternates(ARC *arc, void *p);
static void change_node_name(NODE *node, void *p);
static void reset_net_pointer(NODE *node, NETWORK *net);
static void reset_flag2_node_r(NODE *node, int bits);
void dump_flag2(NODE *node);
static void set_node_flag1_to_inarc_count(NODE *node, void *ptr);
static void set_node_flag1_to_outarc_count(NODE *node, void *ptr);
void recursive_connect(NETWORK *net, int connect_factor, void *(*append)(void *, void *), int (*test)(void *, void *), void *test_data, ARC *base_arc, ARC *cur_arc, void *value);
void find_null_arcs(ARC *arc, void *ptr);

#define MAXSTRING 10000

int Network_add_arc_to_head(NETWORK *net, void *str){
    char *proc="Network_add_arc_to_head";

    if (db >= 1) printf("Entering %s:\n",proc);
    if (db >= 2) {
	printf("Net: %p",net);
	printf("Item:   "); net->arc_func.print(str);
    }
    return (NETWORK_insert_arc_after_node(net, net->start_node, str));
}

int Network_add_arc_to_tail(NETWORK *net, void *str){
    char *proc="Network_add_arc_to_tail";

    if (db >= 1) printf("Entering %s:\n",proc);
    if (db >= 2) {
	printf("Net: %p\n",net);
	printf("Item:   "); net->arc_func.print(str);
    }
    if (net->stop_node == (NODE *)0)
	return (NETWORK_insert_arc_after_node(net, net->start_node, str));
    return (NETWORK_insert_arc_before_node(net, net->stop_node, str));
    
}

int Network_add_net_to_tail(NETWORK *net, NETWORK *mnet){
    char *proc="Network_add_net_to_tail";
    ARC *nullarc;

    if (db >= 1) printf("Entering %s:\n",proc);

    /* first add a null arc to the end of the network */
    if (Network_add_arc_to_tail(net,(void *)0) > 0){
	fprintf(stderr,"Error: %s failed to add a null arc\n",proc);
	return(1);
    }

    nullarc = net->stop_node->in_arcs->arc;
    /* Add in the new network */
    if (Network_merge_network(nullarc->from_node,
			      net->stop_node,mnet) > 0){
	fprintf(stderr,"Error: %s failed to merge the network\n",proc);
	return(1);
    }

    /* Delete the Null Arc */
    if (Network_delete_arc(nullarc) > 0) {
	fprintf(stderr,"Error: %s failed to delete null arc\n",proc);
	return(1);
    }
    return(0);
}


int Network_add_net_to_head(NETWORK *net, NETWORK *mnet){
    char *proc="Network_add_net_to_tail";
    ARC *nullarc;

    if (db >= 1) printf("Entering %s:\n",proc);

    /* first add a null arc to the end of the network */
    if (Network_add_arc_to_head(net,(void *)0) > 0){
	fprintf(stderr,"Error: %s failed to add a null arc\n",proc);
	return(1);
    }

    nullarc = net->start_node->out_arcs->arc;
    /* Add in the new network */
    if (Network_merge_network(nullarc->from_node,
			      nullarc->to_node,mnet) > 0){
	fprintf(stderr,"Error: %s failed to merge the network\n",proc);
	return(1);
    }

    /* Delete the Null Arc */
    if (Network_delete_arc(nullarc) > 0) {
	fprintf(stderr,"Error: %s failed to delete null arc\n",proc);
	return(1);
    }
    return(0);
}

void Network_traverse(NETWORK *net, void (*node_op)(NODE *, void *), void *node_data, void (*arc_op)(ARC *, void *), void *arc_data, int mode){
    char *proc = "Network_traverse";
    static int traversal_id_set = 0, num_trav=0;
    int cur_state = 0, flip_state;
    int max_trav=(sizeof(int) * 8) - 1, i;
    int cur_id = (-1);
    int flag_bit;
    LList *Fringe;
    NODE *pushnode;

    LL_init(&Fringe);

    if (db >= 1) printf("Entering %s:\n",proc);
    if (db >= 2) {
	printf("Net: %p\n",net);
	printf("Mode: %x\n",mode); 
	printf("Static Variables:  traversal_id_set=%8x, num_trav=%d\n",
	       traversal_id_set,num_trav);
    }

    if (net == NULL){
	fprintf(scfp,"%s: Null Network Pointer Passed\n",proc);
	return;
    }
    
    /* Set the default flags */
    if ((mode & NT_CA_For) == 0 && (mode & NT_CA_Back) == 0)
	mode += NT_CA_For;
    if ((mode & NT_For) == 0 && (mode & NT_Back) == 0)
	mode += NT_For;
    if ((mode & NT_Breadth) == 0 && (mode & NT_Depth) == 0
	&& (mode & NT_Inorder) == 0)
	mode += NT_Depth;

    if ((mode & NT_Inorder) != 0){
	int odb=db;
	/* perform the inorder traversal, first set flag1 to the
	   in_arc count */
	db=0;
	if (db >= 5) printf("Recursive call to %s, for Inorder traverse\n",
			    proc);
	if ((mode & NT_For) != 0)
	    Network_traverse(net,set_node_flag1_to_inarc_count,0,0,0,0);
	else
	    Network_traverse(net,set_node_flag1_to_outarc_count,0,0,0,0);
	db = odb;
    }

    /* find a traverse id */
    for (i = 0; i<max_trav; i++)
	if (((1 << i) & traversal_id_set) == 0){
	    cur_id = i;
	    break;
	}
    if (cur_id < 0){
	fprintf(stderr,"%s: Error, too many traversals > %d\n",proc,max_trav);
	exit(1);
    }
    flag_bit = (1 << cur_id);
    traversal_id_set += flag_bit;

    cur_state = flag_bit & net->start_node->flag2;
    flip_state = cur_state ^ flag_bit;

    num_trav ++;
    if (db >= 10) {
	printf("TRAVERSAL STARTED: %d ACTIVE\n",num_trav);
	printf("cur_id=%d  flag_bit=%x  traversal_id_set=%x  cur=%x flip=%x\n",
	       cur_id,flag_bit,traversal_id_set, cur_state,flip_state);
    }

    if ((mode & NT_Verbose) != 0){
	printf("Traverse Network\n");
	printf(" Network name: '%s'\n",net->name);
	printf(" highest_nnode_name = %d\n",net->highest_nnode_name);
	printf(" Node Count = %d\n",net->node_count);
	printf(" Arc Count = %d\n\n",net->arc_count);
	printf(" Mode = %x\n\n",mode);
    }
    
    /* Do the traversal */
    if ((mode & NT_For) != 0){
	LL_put_front(&Fringe,net->start_node);
	net->start_node->flag2 ^= flag_bit;   /* set the visited flag */
    } else {
	LL_put_front(&Fringe,net->stop_node);
	net->stop_node->flag2 ^= flag_bit;   /* set the visited flag */
    }

    while (! LL_empty(Fringe)){
	ARC_LIST_ATOM *p, *pn;
	NODE *node;
	/* pop off the next node */
	if (! LL_get_first(&Fringe,(void *)&node)){
	    fprintf(stderr,"%s: LL_get_first failed\n",proc);
	    exit(1);
	}
	if (db >= 10) {	printf("POPPED NODE: "); print_node(node,0); }

	if (node_op != NULL) node_op(node,node_data);

	/* push the pointed to nodes onto the LLIST */
	for (p = (((mode & NT_For) != 0) ? node->out_arcs :node->in_arcs);
	     p != NULL; p = p->next){
	    pushnode = ((mode & NT_For) != 0) ? p->arc->to_node :
		                               p->arc->from_node;

	    /* only push the node if it's not been pushed before */
	    if (db >= 10) {
		printf("About to push Cs %x",cur_state);
		print_node(pushnode,0);
	    }

	    /* If Inorder, decriment the count of incoming arcs */
	    if ((mode & NT_Inorder) != 0)
		if (--pushnode->flag1 > 0){
		    if (db >= 10)
			printf("********* Node still has incoming arcs\n");
		    continue;
		}

	    if ((cur_state > 0  && (pushnode->flag2 & flag_bit) == 0) ||
		(cur_state == 0 && (pushnode->flag2 & flag_bit) !=  0)){
		if (db >= 10) printf("********* Node already visited \n");
		continue;
	    }
	    pushnode->flag2 ^= flag_bit;   /* set the visited flag */
	    if (db >= 10) printf("Mod_flag2 %x\n",pushnode->flag2); 

	    if ((mode & NT_Breadth) != 0 || (mode & NT_Inorder) != 0){
		LL_put_tail(&Fringe,(void *)pushnode);
	    } else {
		LL_put_front(&Fringe,(void *)pushnode);
	    }
	}

	/* travel the arcs */
	if (node->out_arcs != NULL && (mode & NT_CA_For) != 0 && arc_op !=NULL)
	    for (p = node->out_arcs; p != NULL; p = pn){
		pn = p->next;
		arc_op(p->arc,arc_data);
	    }
	if (node->in_arcs != NULL && (mode & NT_CA_Back) != 0 && arc_op !=NULL)
	    for (p = node->in_arcs; p != NULL; p = pn){
		pn = p->next;
		arc_op(p->arc,arc_data);
	    }

    }	

    traversal_id_set -= (1 << cur_id);
    num_trav --;
}

int Network_merge_network(NODE *from_node, NODE *to_node, NETWORK *from_net){
    char *proc = "Network_merge_network", newnum[10];
    ARC_LIST_ATOM *oa, *p, *p_next;
    int err, *perr = &err, ret;

    /* do this by cheaply by just re-linking the network */
    if (db >= 1) printf("Entering %s:\n",proc);
    if (db >= 2) {
	printf("From Node: "); print_node(from_node,0);
	printf("To Node:   "); print_node(to_node,0);
	Network_traverse(from_net, print_node, 0, print_arc, 0, 0);
    }
    from_node->net->highest_nnode_name ++;
    sprintf(newnum,"M%d",from_node->net->highest_nnode_name);
    Network_traverse(from_net, change_node_name, newnum, NULL, 0, 0);
    
    /* Now that the out_arcs in the from_node have been set, change the */
    /* 'net' pointer of the arcs to correct network */

    /* res the node count based on the new network */
    from_node->net->node_count += 
	(from_net->node_count - 2) < 0 ? 0 : from_net->node_count - 2;
    from_node->net->arc_count += from_net->arc_count;

    reset_net_pointer(from_net->start_node, from_node->net);

    /* link in the start */
    for (oa = from_net->start_node->out_arcs; oa != NULL; oa = oa->next){
	if (db >= 5){
	    printf("Linking in arcs\n"); print_arc(oa->arc,0);
	}
	from_node->out_arcs =add_to_arc_list(from_node->out_arcs,oa->arc,perr);
	if (*perr > 0){
	    printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	    ret = 1; goto RETURN;
	}
	/* change the arcs from pointer to the new node */	
	oa->arc->from_node = from_node;
    }

    /* link in the stop */
    for (oa = from_net->stop_node->in_arcs; oa != NULL; oa = oa->next){
	if (db >= 5) {
	    printf("Linking out arcs\n"); print_arc(oa->arc,0);
	}
	to_node->in_arcs =add_to_arc_list(to_node->in_arcs,oa->arc,perr);
	if (*perr > 0){
	    printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	    ret = 1; goto RETURN;
	}
	/* change the arcs from pointer to the new node */	
	oa->arc->to_node = to_node;
    }

    /* De-allocate the from network */
    for (p = from_net->start_node->out_arcs; p != NULL; p = p_next){
	p_next = p->next;
	from_net->start_node->out_arcs =
	    del_from_arc_list(&(from_net->start_node->out_arcs),p->arc,perr);
    }
    for (p = from_net->stop_node->in_arcs; p != NULL; p = p_next){
	p_next = p->next;
	from_net->stop_node->in_arcs =
	    del_from_arc_list(&(from_net->stop_node->in_arcs),p->arc,perr);
    }
    kill_node2(from_net->stop_node);
    from_net->node_count = 1;
    from_net->stop_node = (NODE *)0;
    Network_destroy(from_net);

    ret=0;
  RETURN:
    if (db >= 1) printf("Exiting %s:\n",proc);
    return ret;
}

int Network_destroy(NETWORK *net){
    char *proc = "Network_destroy";

    if (db >= 1) printf("Entering: %s\n",proc);
    if (db >= 2) printf("Net: %p\n",net);

    while (net->start_node->out_arcs != (ARC_LIST_ATOM *)NULL)
	Network_delete_arc(net->start_node->out_arcs->arc);
    kill_node2(net->start_node);
    net->node_count++;
    free ((void*)net->name);
    free ((void*)net);
    if (db >= 1) printf("Leaving: %s\n",proc);

    return(1);
}


int Network_delete_arc(ARC *arc){
    char *proc = "Network_delete_arc";
    NODE *from_node, *to_node;
    NETWORK *net;
    int err, *perr=&err;
    ARC_LIST_ATOM *p;

    if (db >= 1) printf("Entering: %s\n",proc);
    if (db >= 2) {
	printf("    %s:\n    Arc: ",proc);
	arc->net->arc_func.print(arc->data);
	printf("    From_NODE: "); print_node(arc->from_node,stdout);
	printf("    To_NODE:   "); print_node(arc->to_node,stdout);
    }
    
    if (arc == (ARC *)NULL)
	return 1;

    net = arc->net;
    from_node = arc->from_node;
    to_node = arc->to_node;
    
    /***** remove this arc from the from and to nodes!!!! *****/
    from_node->out_arcs = del_from_arc_list(&(from_node->out_arcs),arc,perr);
    to_node->in_arcs = del_from_arc_list(&(to_node->in_arcs),arc,perr);
    arc->net->arc_func.destroy(arc->data);
    kill_arc(arc);
    net->arc_count--;

#ifdef nneded
    /* SPECIAL CASE:   
       The arc is the ONLY arc in the network. . . This means the arc network 
       must be returned to it's initial state only once. 
       */
    if ((to_node->in_arcs == (ARC_LIST_ATOM *)NULL) && 
	(from_node->out_arcs == (ARC_LIST_ATOM *)NULL) && 
	(strcmp(from_node->name,"START") != 0) &&
	(strcmp(to_node->name,"STOP") != 0)){


		/* killing the STOP NODE */
		kill_node2(to_node);
		net->node_count++;
		to_node = (NODE *)0;
		net->stop_node = (NODE *)0;

        return(0);
    }
#endif


    /* Delete from the "to" direction first */
    if (to_node->in_arcs == (ARC_LIST_ATOM *)NULL) {
	/* this node needs deleted then linked into the previous node */
        if (db >= 10) printf("The To NODE has no in_arcs, re-linking\n");
	/* node isn't the stop node */
	if (strcmp(to_node->name,"STOP") == 0) {
	    if (db >= 10) printf(" Special case for the STOP NODE\n");
	    if (strcmp(from_node->name,"START") != 0) {
		/* delete the from_node, replacing it with the to_node */
		/* copy the from_node's in_arcs to to_node's in_arcs */
		for (p = from_node->in_arcs; p != NULL; p = p->next){
		    to_node->in_arcs = add_to_arc_list(to_node->in_arcs,
						       p->arc,perr);
		    p->arc->to_node = to_node;
		}
		/* delete the from_node */
		kill_node2(from_node);
		net->node_count++;
		from_node = (NODE *)0;
	    } else {
		/* killing the STOP NODE,  */
		kill_node2(to_node);
		net->node_count++;
		to_node = (NODE *)0;
		net->stop_node = (NODE *)0;
	    }
	} else {
	    /* add that to_node's out_arcs to change the from_node's out_arcs */
	    for (p = to_node->out_arcs; p != NULL; p = p->next){
		from_node->out_arcs = add_to_arc_list(from_node->out_arcs,
						      p->arc,perr);
		p->arc->from_node = from_node;
	    }
	    kill_node2(to_node);
	    to_node = (NODE *)0;
	    net->node_count++;
	}
    }
    /* Delete in the "from" direction next */
    /*   The check "from_node != 0" signifies the re-linking is finished */
    if (from_node != (NODE *)0 && 
	from_node->out_arcs == (ARC_LIST_ATOM *)NULL) {
	/* this node needs deleted then linked into the previous node */
        if (db >= 10) printf("    Re-linking arc from the from_node, to the"
			     " to_node, and deleting from_node\n");
	if (strcmp(from_node->name,"START") == 0){
	    if (db >= 10) printf("    Returning network to init state\n");
	    /* check to make sure the stop node wasn't already killed */
	    if (to_node != (NODE *)0){
	        kill_node2(to_node);
		to_node = net->stop_node = (NODE *)0;	    
	    }
	} else {
	    /* relink the in_arcs from from_node to to_node's in_arcs */
	    for (p = from_node->in_arcs; p != NULL; p = p->next){
	        if (db >= 10) {
		    printf("    Re-linking arc: "); print_arc(p->arc,stdout);
		}
		to_node->in_arcs = add_to_arc_list(to_node->in_arcs,
						   p->arc,perr);
		p->arc->to_node = to_node;
	    }
	    /* kill the from node */
	    kill_node2(from_node);
	    net->node_count++;
	    from_node = (NODE *)0;
	} 
    }

    return(0);
}


NETWORK *Network_create_from_TEXT(TEXT *text, char *name, void (*aprn)(void *), int (*aequal)(void *, void *),  void (*adestroy)(void *), int (*is_null_alt)(void *), int (*is_opt_del)(void *), void *(*copy)(void *), void *(*make_empty)(void *), int (*use_count)(void *, int)){
    char *proc="Network_create_from_TEXT";
    NETWORK *net;
    WORD *tword = NULL_WORD;          /* temp storage getting new words */
    TEXT *ctext = text,               /* current position is text string */
         token[MAXSTRING];                  /* temp storage for tokens */

    if (db >= 1) printf("Entering: %s\n",proc);
    if (db >= 2) printf("    function args: text='%s'\n",text);
	
    /* Stage 1: parse the string, separating it into singleton words */
    /*          and alternations */

    net = Network_init(aprn,aequal,adestroy,is_null_alt,
		       is_opt_del, copy,make_empty,use_count,name);

    while (! end_of_TEXT(*ctext)) {
	if (find_next_TEXT_token(&ctext,token,MAXSTRING)) {	
	    if (db > 5) printf("    Token: '%s', ctext: %s\n",token,ctext);
	    /* create new word structure */
	    
	    tword = new_WORD_parseText(token, -1, 0.0, 0.0, 0.0, 0, 0, -1.0);
	    /* append to the word list */
	    if (Network_add_arc_to_tail(net,(void *)tword) > 0){
		fprintf(stderr,"Error: Network_add_arc_to_tail failed in %s\n",
			proc);
		return(NULL_NETWORK);
	    }
	}
    }
    Network_traverse(net,NULL,0,expand_alternates,0,0);

    return(net);
}


extern NETWORK *Network_init_from_net(NETWORK *net, char *name){
  return Network_init(net->arc_func.print, 
		      net->arc_func.equal, 
		      net->arc_func.destroy, 
		      net->arc_func.is_null_alt,
		      net->arc_func.is_opt_del, 
		      net->arc_func.copy,
		      net->arc_func.make_empty,
		      net->arc_func.use_count,
		      name);
}

extern NETWORK *Network_init(void (*arc_data_prn)(void *), int (*arcs_equal)(void *, void *), void arc_data_destroy(void *), int (*is_null_alt)(void *), int (*is_opt_del)(void *), void *(*copy)(void *), void *(*make_empty)(void *), int (*use_count)(void *, int), char *name){
    char *proc="Network_init";
    NETWORK *net;
    int err, *perr = &err;

    if (db >= 1) printf("Entering: %s\n",proc);
    if (db >= 2) printf("Net name: %s\n",name);

    /* initialization stuff */
    alloc_singarr(net,1,NETWORK);
    net->name = (char *)TEXT_strdup((TEXT *)name);
    net->node_count = 1;  /* for the initial start node */
    net->arc_count = 0;
    net->arc_func.print = arc_data_prn;
    net->arc_func.destroy = arc_data_destroy;
    net->arc_func.equal = arcs_equal;
    net->arc_func.is_null_alt = is_null_alt;
    net->arc_func.is_opt_del = is_opt_del;
    net->arc_func.copy = copy;
    net->arc_func.make_empty = make_empty;
    net->arc_func.use_count = use_count;

    net->arc_func.print_name       =  "";
    net->arc_func.destroy_name     =  "";
    net->arc_func.equal_name       =  "";
    net->arc_func.is_null_alt_name =  "";
    net->arc_func.is_opt_del_name  =  "";
    net->arc_func.copy_name        =  "";
    net->arc_func.make_empty_name  =  "";
    net->arc_func.use_count_name   =  "";

    /* create the start Node */
    net->highest_nnode_name = 0;
    net->start_node = make_node("START",net,(ARC*)NULL,(ARC*)NULL,
				&(net->highest_nnode_name),perr);
    if (*perr > 0)
	{printf("%s:*ERR: make_node(START) returns %d\n",proc,*perr);
	 goto RETURN;
     }
    net->start_node->start_state = T;
    net->start_node->stop_state  = F;
    net->stop_node = (NODE *)0;
  RETURN:
    return(net);
}

/*************************************************/
/*  Dumps a node of a network.                   */
/*************************************************/
void print_node(NODE *node, void *p){
    if (node != NULL){
      /* printf(" Node name: %9s  addr: %x",node->name,(int)node); */
        printf(" Node name: %9s  ",node->name);
	printf(" is_start='%s', is_stop='%s', flag1='%d', flag2='%08x'\n",
	       bool_print(node->start_state),bool_print(node->stop_state),
	       node->flag1,node->flag2);
    }    
}

/*************************************************/
/*  Dumps an arc of a network.                   */
/*************************************************/
void print_arc(ARC *arc, void *p)          
{
    if (arc != NULL){
	printf("  Arc From: ");
	if (arc->from_node != NULL) printf("%9s",arc->from_node->name);
	else         
	    printf("**NULL**");
	printf("  To: ");     
	if (arc->to_node != NULL)   printf("%9s",arc->to_node->name);
	else                        printf(" **NULL**");
	printf("  addr: %p  weight: %d  ",arc,arc->weight); 
	printf("  data addr: %p ",arc->data); 
	arc->net->arc_func.print((void *)arc->data);
    }
    return;
} /* end of function "print_arc" */


/*************************************************/
/* define Network_fully_connect as a subcase of  */
/* Network_fully_connect_cond                    */
/*************************************************/
int always_true(void *data, void *elem);
int always_true(void *data, void *elem){ return 1; }

int Network_fully_connect(NETWORK *net, int connect_factor, void *(*append)(void *, void *)){
    return(Network_fully_connect_cond(net, connect_factor, append,
				      always_true, (void *)0));
}

void recursive_connect(NETWORK *net, int connect_factor, void *(*append)(void *, void *), int (*test)(void *, void *), void *test_data, ARC *base_arc, ARC *cur_arc, void *value)
{
    ARC_LIST_ATOM *p;
    void *new_data;
    int pass;

    if (connect_factor == 0) return;  /* end recursion */
    for (p=cur_arc->to_node->out_arcs; p != NULL; p = p->next){
	new_data = append(value,p->arc->data);
	if ((pass = test(test_data,new_data)))
	    NETWORK_insert_arc_between_node(net,base_arc->from_node,
					    p->arc->to_node,
					    new_data);
	recursive_connect(net,connect_factor-1,append,test,test_data,base_arc,
			  p->arc,new_data);
	if (! pass) net->arc_func.destroy(new_data);
    }
}

int Network_fully_connect_cond(NETWORK *net, int connect_factor, void *(*append)(void *, void *), int (*test)(void *, void *), void *test_data){
    char *proc = "Network_fully_connect_cond";
    ARCSET arcset;
    int aa;

   
    if (db >= 1) printf("Entering %s:\n",proc);
    if (db >= 2)
	printf("Net: %p  Connect_factor: %d",net,connect_factor);

    alloc_ARCSET(&arcset,net->arc_count+1);

    Network_traverse(net,0,0,add_to_arcset,(void *)&(arcset),
		     NT_CA_For + NT_Breadth);

    if (db >= 5){
	printf("%s: Arcset\n",proc);
	for (aa=0; aa<arcset.num; aa++){
	    printf("  %2d: ",aa);
	    print_arc(arcset.arcs[aa],0);
	}
    }
    for (aa=0; aa<arcset.num; aa++) {
	recursive_connect(net, connect_factor-1, append, test, test_data,
			  arcset.arcs[aa],arcset.arcs[aa],
			  arcset.arcs[aa]->data);
    }
	
    free_ARCSET(&arcset);
    return(0);
}

NETWORK *Network_create_from_WTOKE(WTOKE_STR1 *wt,int start,int end, char *name, void (*aprn)(void *), int (*aequal)(void *, void *),  void (*adestroy)(void *), int (*is_null_alt)(void *), int (*is_opt_del)(void *), void *(*copy)(void *), void *(*make_empty)(void *), int (*use_count)(void*, int), int left_to_right){
    char *proc="Network_create_from_WTOKE";
    NETWORK *net;
    WORD *tword = NULL_WORD;          /* temp storage getting new words */
    int i;
    if (db >= 1) printf("Entering: %s\n",proc);
    if (db >= 2) printf("    words from %d to %d\n",start,end);
	
    /* Stage 1: parse the string, separating it into singleton words */
    /*          and alternations */

    net = Network_init(aprn,aequal,adestroy,is_null_alt,
		       is_opt_del,copy,make_empty,use_count,name);
    
    for (i=start; (i<=wt->n)&&(i<=end); i++){
	if ((!wt->word[i].ignore) && wt->word[i].alternate) {
	    NETWORK *alt_net, *alt_net_out=NULL_NETWORK;
	    int first_alt=1, acnt;

	    /* skip the ALT_BEGIN */
	    for (; (i<=wt->n) && (i<=end) &&
		 (TEXT_strcasecmp((TEXT*)"<ALT_BEGIN>",wt->word[i].sp) == 0);
		 i++)
		;
	    do {
		alt_net = Network_init(aprn,aequal,adestroy,is_null_alt,
				       is_opt_del,copy,make_empty,use_count,
				       name); 
		acnt=0;
		while ((i<=wt->n) && (i<=end) &&
		       (TEXT_strCcasecmp((TEXT*)"<ALT",
					 wt->word[i].sp,4) != 0)) {
		  /* create new word structure */
		  tword = new_WORD_parseText(wt->word[i].sp, -1,
				   wt->word[i].t1,
				   wt->word[i].t1 + wt->word[i].dur,
				   (wt->has_conf)? wt->word[i].confidence : -1,	
				   0, 0, -1.0);
		  if (left_to_right){
		    /* append to the word list */		    
		    if (Network_add_arc_to_tail(alt_net,(void *)tword) > 0){
		      fprintf(stderr,"Error: Network_add_arc_to_tail ");
		      fprintf(stderr,"failed in %s\n",proc);
		      return(NULL_NETWORK);
		    }
		  } else {
		    /* append to the word list */		    
		    if (Network_add_arc_to_head(alt_net,(void *)tword) > 0){
		      fprintf(stderr,"Error: Network_add_arc_to_head ");
		      fprintf(stderr,"failed in %s\n",proc);
		      return(NULL_NETWORK);
		    }
		  }
		  i++;
		  acnt++;
		}		
		if (acnt == 0){ /* add a NULL work into the transcript */
		    /* create new word structure */
		  tword = new_WORD((TEXT *)"@", -1, 0.0, 0.0, 
				   (wt->has_conf)? wt->word[i].confidence : -1,
				   (TEXT *)NULL, (TEXT *)NULL,
				   0, 0, -1.0);

		    /* append to the word list */
		    if (Network_add_arc_to_tail(alt_net,(void *)tword) > 0){
			fprintf(stderr,"Error: Network_add_arc_to_tail ");
			fprintf(stderr,"failed in %s\n",proc);
			return(NULL_NETWORK);
		    }
		    i++;
		}
		if (first_alt)
		    alt_net_out = alt_net;
		else {
		    /* merge the graphs together */
		    Network_merge_network(alt_net_out->start_node,
					  alt_net_out->stop_node,alt_net);
		}
	        first_alt = 0;
		if (TEXT_strcasecmp((TEXT*)"<ALT>",wt->word[i].sp) == 0 &&
		    TEXT_strcasecmp((TEXT*)"<ALT_END>",wt->word[i+1].sp) == 0)
		    ;
		else if (TEXT_strcasecmp((TEXT*)"<ALT_END>",wt->word[i].sp)!=0)
		    i++;
	    } while ((i<=wt->n) && (i<=end) &&
		     (TEXT_strcasecmp((TEXT*)"<ALT_END>",wt->word[i].sp)!= 0));

	    if (alt_net_out == (NETWORK *)0){
		fprintf(stderr,"Error: %s failed to produce a network for "
			"an alternation.  Possible file format error.\n",proc);
		return(NULL_NETWORK);
	    }		

	    if (left_to_right){
	      if (Network_add_net_to_tail(net,alt_net_out) > 0){
		fprintf(stderr,"Error: Network_add_net_to_tail failed in %s\n",
			proc);
		return(NULL_NETWORK);
	      }
	    } else {
	      if (Network_add_net_to_head(net,alt_net_out) > 0){
		fprintf(stderr,"Error: Network_add_net_to_head failed in %s\n",
			proc);
		return(NULL_NETWORK);
	      }
	    }

	} else if (!wt->word[i].ignore) {
	    if (db > 5) printf("    Adding: '%s'\n",wt->word[i].sp);
	    	
	    /* create new word structure */
	    tword = new_WORD_parseText(wt->word[i].sp, -1,
			     wt->word[i].t1, wt->word[i].t1 + wt->word[i].dur,
			     (wt->has_conf) ? wt->word[i].confidence : -1,
			     0, 0, -1.0);
	    if (left_to_right){
	      /* append to the word list */
	      if (Network_add_arc_to_tail(net,(void *)tword) > 0){
		fprintf(stderr,"Error: Network_add_arc_to_tail failed in %s\n",
			proc);
		return(NULL_NETWORK);
	      }
	    } else {
	      /* prepend to the word list */
	      if (Network_add_arc_to_head(net,(void *)tword) > 0){
		fprintf(stderr,"Error: Network_add_arc_to_tail failed in %s\n",
			proc);
		return(NULL_NETWORK);
	      }
	    }
	} else {
	    if (db > 5) printf("    Ignored: '%s'\n",wt->word[i].sp);
	}
    }
    return(net);
}

/************************************************************************/
/* File internal functions, not intended for outside use                */
/************************************************************************/

int NETWORK_insert_arc_after_node(NETWORK *net, NODE *last_node, void *str){
    char *proc = "NETWORK_insert_arc_after_node";
    ARC *arcx;
    ARC_LIST_ATOM *oa;
    int err, *perr = &err, ret;
    NODE *new_node;
    
    /* alloc an arc, set */
    arcx = make_arc(str,last_node,(NODE*)NULL,perr);
    arcx->net = net;

    /* alloc a node, and set the input to the node to be the new arc */
    new_node = make_node((net->stop_node == (NODE *)0) ? "STOP" : (char*)NULL,
			 net,arcx,(ARC *)0, &(net->highest_nnode_name),perr);
    new_node->flag2 = net->start_node->flag2;

    if (*perr > 0){
	printf("%s:*ERR: make_node() returns %d\n",proc,*perr);
	ret = 1; goto RETURN;
    }

    /* set the start and stop flags */
    new_node->stop_state = new_node->start_state = F;
    if (net->stop_node == (NODE *)0){
	net->stop_node = new_node;
	net->stop_node->stop_state  = T;
    }
    net->node_count++;
    net->arc_count++;

    /* link the arc to the new node */
    arcx->to_node = new_node;

    /* use the last node's out arcs as this nodes out_arc's */
    /* making sure to change the arc's from pointer to the new node */
    for (oa = last_node->out_arcs; oa != NULL; oa = oa->next){
	new_node->out_arcs = add_to_arc_list(new_node->out_arcs,oa->arc,perr);
	/* change the arcs from pointer to the new node */	
	oa->arc->from_node = new_node;
	if (*perr > 0){
	    printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	    ret = 1; goto RETURN;
	}
    }
    { /* a cheat, delete all the arcs */
	ARC_LIST_ATOM *p1, *p1_next;
	for (p1 = last_node->out_arcs;  p1 != NULL; p1 = p1_next){
	    p1_next = p1->next;
	    free((void*)p1);
	}
	last_node->out_arcs = NULL;
    }
    /* link the last node's out arc to the new arc */
    last_node->out_arcs = add_to_arc_list(last_node->out_arcs,arcx,perr);    

    ret = 0;
  RETURN:
    return ret;
}

int NETWORK_insert_arc_before_node(NETWORK *net, NODE *this_node, void *str){
    char *proc = "NETWORK_insert_arc_before_node";
    ARC *arcx;
    ARC_LIST_ATOM *oa;
    int err, *perr = &err, ret;
    NODE *new_node;

    /* alloc an arc, set the arc to point to this_node*/
    arcx = make_arc(str,(NODE*)NULL,this_node,perr);
    arcx->net = net;

    /* alloc a node, and set the input to the node to be the new arc */
    new_node = make_node((char*)NULL,net,(ARC *)0,arcx,
			 &(net->highest_nnode_name),perr);
    new_node->flag2 = net->start_node->flag2;

    if (*perr > 0){
	printf("%s:*ERR: make_node() returns %d\n",proc,*perr);
	ret = 1; goto RETURN;
    }
    net->node_count++;
    net->arc_count++;

    /* link the arc to the new node */
    arcx->from_node = new_node;

    /* use the this_node's in arcs as the new nodes in_arc's */
    /* making sure to change the arc's to pointer to the new node */
    for (oa = this_node->in_arcs; oa != NULL; oa = oa->next){
	new_node->in_arcs = add_to_arc_list(new_node->in_arcs,oa->arc,perr);
	/* change the arcs from pointer to the new node */	
	oa->arc->to_node = new_node;
	if (*perr > 0){
	    printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	    ret = 1; goto RETURN;
	}
    }
    { /* a cheat, delete all the arcs */
	ARC_LIST_ATOM *p1, *p1_next;
	for (p1 = this_node->in_arcs;  p1 != NULL; p1 = p1_next){
	    p1_next = p1->next;
	    free((void*)p1);
	}
	this_node->in_arcs = NULL;
    }
    /* link the last node's out arc to the new arc */
    this_node->in_arcs = add_to_arc_list(this_node->in_arcs,arcx,perr);    
    
    ret = 0;
  RETURN:
    return ret;
}

int NETWORK_insert_arc_between_node(NETWORK *net, NODE *from_node, NODE *to_node, void *str){ 
   char *proc = "NETWORK_insert_arc_between_node";
    ARC *arcx;
    int err, *perr = &err, ret;

    /* alloc an arc, set the arc to point to this_node*/
    arcx = make_arc(str,(NODE*)NULL,from_node,perr);
    arcx->net = net;

    net->arc_count++;

    /* link the arc to the new node */
    arcx->from_node = from_node;
    arcx->to_node = to_node;

    arcx->from_node->out_arcs = add_to_arc_list(arcx->from_node->out_arcs,arcx,perr);
    if (*perr > 0){
	printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	ret = 1; goto RETURN;
    }

    arcx->to_node->in_arcs = add_to_arc_list(arcx->to_node->in_arcs,arcx,perr);
    if (*perr > 0){
	printf("%s:*ERR: add_to_arc_list() returns %d\n",proc,*perr);
	ret = 1; goto RETURN;
    }
    
    ret = 0;
  RETURN:
    return ret;
}

static void change_node_name(NODE *node, void *ptr){
    char *t, buff[40];
    if (node != NULL){
	t = node->name;
	sprintf(buff,"%s%s",(char *)ptr,t);
	node->name = (char *)TEXT_strdup((TEXT *)buff);
	free(t);
    }    
}

static void expand_alternates(ARC *arc, void *ptr){
    char *proc = "expand_alternates";
    WORD *tw = (WORD *)(arc->data);
    TEXT *ctext, *p, token[MAXSTRING], *p1;
    NETWORK *subnet, *subnet2=NULL_NETWORK;
    int first=1;
    char buf[20];
    static int alt=1;

   /* if the string has alternates */
    if (TEXT_strchr(tw->value,'{') != NULL){
	if (db >= 5) {
	    printf("%s: alternates found in arc\n",proc);
	    arc->net->arc_func.print(arc->data);
	}
	/* copy the text, removing the beginning and ending braces */
	p = ctext = TEXT_strdup(tw->value+1);
	if ((p1 = TEXT_strrchr(p,'}')) != NULL)
	    *p1 = NULL_TEXT;
	
	while (! end_of_TEXT(*p)) {
	    if (find_next_TEXT_alternation(&p,token,MAXSTRING)) {
		if (db >= 5)printf("    Token: '%s'\n",token);
		sprintf(buf,"Alt-net %d",alt);
		subnet = Network_create_from_TEXT(token, buf,
						  arc->net->arc_func.print,
						  arc->net->arc_func.equal,
						  arc->net->arc_func.destroy,
						  arc->net->arc_func.is_null_alt,
						  arc->net->arc_func.is_opt_del,
						  arc->net->arc_func.copy,
						  arc->net->arc_func.make_empty,
						  arc->net->arc_func.use_count);
		alt ++;
		if (first)
		    subnet2 = subnet;
		else {
		    /* merge the graphs together */
		    if (db >= 10) {
			printf("%s: merge net 1\n",proc);
			Network_traverse(subnet,print_node,0,print_arc,0,
					 NT_CA_For+NT_CA_Back);
			
			printf("%s: merge net 2\n",proc);
			Network_traverse(subnet2,print_node,0,print_arc,0,
					 NT_CA_For+NT_CA_Back);
		    }
		    Network_merge_network(subnet2->start_node,
					  subnet2->stop_node,subnet);
		    if (db >= 10) {
			printf("%s: Resulting net\n",proc);
			Network_traverse(subnet2,print_node,0,print_arc,0,
					 NT_CA_For+NT_CA_Back);
		    }
		}
		first=0;
	    }
	}
	TEXT_free(ctext);

	if (db >= 10){
	    printf("Network After Merge\n");
	    Network_traverse(subnet2,print_node,0,print_arc,0,0);
	}
	Network_merge_network(arc->from_node,arc->to_node,subnet2);
  	Network_delete_arc(arc);
    }
}

static void reset_flag2_node_r(NODE *node, int bits)            
 /*******************************************************************/
 /*   Turns off flag2 of NODE *node and all nodes following it,     */
 /* recursively.                                                    */
 /*******************************************************************/
{
    char *proc = "reset_flag2_node_r";
    ARC_LIST_ATOM *p;
    db_enter_msg(proc,1); /* debug only */
    if (node != NULL) {
	if ((node->flag2 & bits) > 0)
	    node->flag2 -= bits;
	if (node->out_arcs != NULL){
	    for (p = node->out_arcs; p != NULL; p = p->next)
		reset_flag2_node_r(p->arc->to_node,bits);
	}
    }
    db_leave_msg(proc,1); /* debug only */
    return;
} /* end of function "reset_flag2_node_r" */

static void reset_net_pointer(NODE *node, NETWORK *net)            
 /*******************************************************************/
 /*   resets the network pointer in each arc recursively.           */
 /*******************************************************************/
{
    char *proc = "reset_net_pointer";
    ARC_LIST_ATOM *p;
    db_enter_msg(proc,1); /* debug only */
    if (node != NULL) {
	if (node->out_arcs != NULL){
	    node->net = net;
	    node->flag2 = net->start_node->flag2; /* reset flag 2 */
	    for (p = node->out_arcs; p != NULL; p = p->next){
		p->arc->net = net;
		reset_net_pointer(p->arc->to_node,net);
	    }
	}
    }
    return;
} /* end of function "reset_net_pointer" */

/* A recursive, exhaustive network search, printing out flag2's contents */
void dump_flag2(NODE *node) { 
    ARC_LIST_ATOM *p;
    if (node != NULL){
	printf("node '%s' flag2 = '%08x'\n",node->name,node->flag2);
	if (node->out_arcs != NULL)
	    {for (p = node->out_arcs; p != NULL; p = p->next)
		 dump_flag2(p->arc->to_node);
	 }  }
    return;
} 

/* sets flag1 of the node to the number if incoming arcs */
static void set_node_flag1_to_inarc_count(NODE *node, void *ptr){
    int c = 0;
    ARC_LIST_ATOM *p;
    if (node != NULL){
	for (p = node->in_arcs; p != NULL; p = p->next)
	    c++;
	node->flag1 = c;
    }    
}

/* sets flag1 of the node to the number if outgoing arcs */
static void set_node_flag1_to_outarc_count(NODE *node, void *ptr){
    int c = 0;
    ARC_LIST_ATOM *p;
    if (node != NULL){
	for (p = node->out_arcs; p != NULL; p = p->next)
	    c++;
	node->flag1 = c;
    }    
}

/******************************************************************/
/*  Warning the network copy function uses global variables */
static NETWORK *out_net = (NETWORK *)0;
static NODE ***node_map;
static int n_node_map = 0;
static int insert_node_in_node_map(NODE *n1, NODE *n2, int add);
static void nt_copy_arc(ARC *arc, void *p);

static int insert_node_in_node_map(NODE *n1, NODE *n2, int add){
    int n;
    for (n=0; n<n_node_map; n++)
        if (node_map[n][0] == n1) 
	    return(n);
    if (!add) 
        return(-1);

    if (db >= 10){
        printf("   Adding Node name: %9s  addr: %p    ",n1->name,n1);
	printf("   Adding Node name: %9s  addr: %p\n",n2->name,n2);
    }
    node_map[n_node_map][0] = n1;
    node_map[n_node_map][1] = n2;
    return(n_node_map++);
}


void find_null_arcs(ARC *arc, void *ptr){
    ARC_LIST_ATOM **arcs = (ARC_LIST_ATOM **)ptr;
    int perr;

    if (arc != NULL && arc->data == (void *)0){
	*arcs = add_to_arc_list(*arcs, arc, &perr);
	if (perr != 0){
	    fprintf(stderr,"Error: add_to_arc_list in find_null_arcs "
		    "failed");
	    exit(1);
	}
    }
}
      
static void nt_copy_arc(ARC *arc, void *p)          
{
    char *proc = "nt_copy_arc";
    void *new_data;
    ARC *new_arc = (ARC *)0;
    ARC_LIST_ATOM *parc;
    NETWORK *snet;

    if (db >= 10) printf("============================================\n");
    if (arc != NULL){

	/* push the start nodes onto the list */
	if (arc->from_node == arc->net->start_node){
	    if (db >= 10) printf("Setting out the out_net\n");
	    insert_node_in_node_map(arc->from_node,out_net->start_node,1);
	    if (out_net->stop_node == (NODE*)0){
	        /* prime the pump on the network */
	        if (db >= 10) printf("Prime oup the network\n");
	        Network_add_arc_to_tail(out_net, new_data = 
					arc->net->arc_func.copy(arc->data));
		Network_add_arc_to_tail(out_net, (void *)0);
		if (db >= 10) printf("Searching for arcs\n");
		for (parc=out_net->start_node->out_arcs;
		     parc != (ARC_LIST_ATOM *)0; parc = parc->next)
		    if (parc->arc->data == new_data)
		        new_arc = parc->arc;
		if (new_arc == (ARC *)0){
		    fprintf(stderr,"Error: Internal error to '%s'\n",proc);
		    exit(1);
		}
		if (db >= 10) printf("Completed search\n");
		insert_node_in_node_map(arc->to_node,new_arc->to_node,1);
		if (db >= 10) {
		    printf("-------------- OUT_NET\n");
		    Network_traverse(out_net,print_node,0,print_arc,0,
				     NT_CA_For+NT_Verbose);
		    printf("-------------- OUT_NET\n");
		}
		return;
	    }
	} 
	if (insert_node_in_node_map(arc->to_node,0,0) == -1){
	    /* the node has not been seen before, create a new network
	       and, with the arc and a null arc, then merge the network
	       into the output network */
	    if (db >= 10) {
	        printf("Adding new arc\n");
		printf("-------------- OUT_NET\n");
		Network_traverse(out_net,print_node,0,print_arc,0,
				 NT_CA_For+NT_Verbose);
		printf("-------------- OUT_NET\n");
	    }
	    if ((snet = Network_init(arc->net->arc_func.print,
				      arc->net->arc_func.equal,
				      arc->net->arc_func.destroy,
				      arc->net->arc_func.is_null_alt,
				      arc->net->arc_func.is_opt_del,
				      arc->net->arc_func.copy,
				      arc->net->arc_func.make_empty,
				      arc->net->arc_func.use_count,
				      "tmp net")) == NULL_NETWORK){
	        fprintf(scfp,"Internal error in %s\n",proc);
		exit(1);
	    }
	    Network_add_arc_to_tail(snet, new_data = 
				    arc->net->arc_func.copy(arc->data));
	    Network_add_arc_to_tail(snet, (void *)0);
	    /* find the new arc in the network */
	    if (db >= 10) printf("Searching for arcs\n");
	    for (parc=snet->start_node->out_arcs;
		 parc != (ARC_LIST_ATOM *)0; parc = parc->next)
	        if (parc->arc->data == new_data)
		    new_arc = parc->arc;
	    if (new_arc == (ARC *)0){
	        fprintf(stderr,"Error: Internal error to '%s'\n",proc);
		exit(1);
	    }
	    if (db >= 10) {
	        printf("-------------- SNET\n");
		Network_traverse(snet,print_node,0,print_arc,0,
				 NT_CA_For+NT_Verbose);
		printf("-------------- SNET\n");
		printf("After Traversal\n");
	    }

	    Network_merge_network(
		   node_map[insert_node_in_node_map(arc->from_node,0,0)][1],
		   out_net->stop_node,snet);
	    if (db >= 10) printf("After Merge network\n");
	    insert_node_in_node_map(arc->to_node,new_arc->to_node,1);
	    if (db >= 10) printf("After add node in node map\n");
	} else {
	    NETWORK_insert_arc_between_node(out_net, 
		node_map[insert_node_in_node_map(arc->from_node,0,0)][1],
		node_map[insert_node_in_node_map(arc->to_node,0,0)][1],
					    arc->net->arc_func.copy(arc->data));
	}
	if (db >= 10) {
	    printf("---------------- resulting network\n");
	    Network_traverse(out_net,0,0,print_arc,0,NT_CA_For+NT_Verbose);
	    printf("----------------- resulting network\n");
	}
    }
    return;
} /* end of function "nt_copy_arc" */

NETWORK *Network_copy(NETWORK *in_net){
    char *proc = "Network_copy";

    if (db > 0) printf("Entering: %s\n",proc);

    n_node_map = 0;
    alloc_2dimZ(node_map,in_net->node_count + 1,2,NODE *,(NODE *)0);

    /* initialize the structure */
    if ((out_net = Network_init(in_net->arc_func.print,
			   in_net->arc_func.equal,
			   in_net->arc_func.destroy,
			   in_net->arc_func.is_null_alt,
			   in_net->arc_func.is_opt_del,
			   in_net->arc_func.copy,
			   in_net->arc_func.make_empty,
			   in_net->arc_func.use_count,
			   rsprintf("Copy of %s",in_net->name))) == (NETWORK *)0)
       return((NETWORK *)0);

    if (db >= 10) printf("%s: Beginning copy \n",proc);
    Network_traverse(in_net,0,0,nt_copy_arc,0,NT_CA_For);

    if (db >= 10) printf("%s: deleting arcs with NULL's \n",proc);
    Network_delete_null_arcs(out_net);

    free_2dimarr(node_map,in_net->node_count + 1,NODE *);
    return(out_net);
}


/**********************************************/
/*       SGML dump callbacks                  */

static void add_to_node_list(NODE *node, void *ptr);
static void dump_net_tag(NETWORK *net, FILE *fp);
static void dump_node_tag(NODE *node, FILE *fp);
static void dump_arc_tag(ARC *arc, FILE *fp);
static int lookup_node(NODE **list, int cnt, NODE *node);
static NODE **node_list;
static int max_node = 0, num_node = 0;

static void dump_net_tag(NETWORK *net, FILE *fp){
  fprintf(fp,"<NET version=\"1.0\" name=\"%s\" lastnode=\"%d\" "
	  " Dtype=\"WORD\" ",
	  net->name, 
	  net->highest_nnode_name);
  fprintf(fp,"print=\"%s\" destroy=\"%s\" equal=\"%s\" "
	  "is_null_alt=\"%s\" is_opt_del=\"%s\" copy=\"%s\" mkempty=\"%s\" use_cnt=\"%s\">\n",
	  net->arc_func.print_name,
	  net->arc_func.destroy_name,
	  net->arc_func.equal_name,
	  net->arc_func.is_null_alt_name,
	  net->arc_func.is_opt_del_name,
	  net->arc_func.copy_name,
	  net->arc_func.use_count_name,
	  net->arc_func.make_empty_name);
}

static void dump_node_tag(NODE *node, FILE *fp){
  fprintf(fp,"<NODE name=\"%s\" id=\"%d\">\n", node->name, node->flag1);
}

static void dump_arc_tag(ARC *arc, FILE *fp){
  fprintf(fp,"<ARC weight=\"%d\" to=\"%d\" from=\"%d\">\n",
	  arc->weight, lookup_node(node_list, num_node, arc->to_node),
	  lookup_node(node_list, num_node, arc->from_node));  
  sgml_dump_WORD((WORD *)arc->data, fp);
  fprintf(fp,"</ARC>\n");
}

static int lookup_node(NODE **list, int cnt, NODE *node){
  int nn;

  for (nn=0; nn<cnt; nn++)
    if (list[nn] == node) 
      return nn;
  return(-1);
}
  
static void add_to_node_list(NODE *node, void *ptr){
  if (lookup_node(node_list, num_node, node) != -1)
      return;

  if (num_node == max_node)
    expand_1dim(node_list,num_node,max_node,2,NODE *, 1);
  node_list[num_node] = node;
  node->flag1 = num_node;
  num_node++;
}

void Network_sgml_dump(NETWORK *net, FILE *fp){
    char *proc = "Network_sgml_dump";
    int n;

    if (db > 0) printf("Entering: %s\n",proc);

    /* initialize variables */
    num_node = 0;
    max_node = net->node_count + 10;
    alloc_singZ(node_list,max_node,NODE *,(NODE *)0);
    Network_traverse(net,add_to_node_list,0,0,0,0);

    dump_net_tag(net, fp);

    fprintf(fp,"<NODES start_id=\"%d\" stop_id=\"%d\">\n",
	    net->start_node->flag1, net->stop_node->flag1);
    for (n=0; n<num_node; n++)
      dump_node_tag(node_list[n],fp);      
    fprintf(fp,"</NODES>\n");

    fprintf(fp,"<ARCS>\n");
    for (n=0; n<num_node; n++){
      ARC_LIST_ATOM *oa;
      for (oa = node_list[n]->out_arcs; oa != NULL; oa = oa->next)
	dump_arc_tag(oa->arc,fp);
    }
    fprintf(fp,"</ARCS>\n");

    fprintf(fp,"</NET>\n");
    
    /* erase variables */
    free_singarr(node_list, NODE *);
}


void Network_delete_null_arcs(NETWORK *out_net){
    char *proc="Network_delete_null_arcs";
    ARC_LIST_ATOM *arcs = (ARC_LIST_ATOM *)0;
    int perr;

    /* do this a different way: 
       1: build a list of null_arcs
       2: foreach null arc, delete it */
    
    Network_traverse(out_net,0,0,find_null_arcs,&arcs,0);

    while (arcs != (ARC_LIST_ATOM *)0){
	ARC *sarc = arcs->arc;
	arcs = del_from_arc_list(&arcs,sarc,&perr);
	if (perr != 0){
	    fprintf(stderr,"Error: del_from_arc_list failed in %s",
		    proc);
	    exit(1);
	}
	Network_delete_arc(sarc);
    }
}


