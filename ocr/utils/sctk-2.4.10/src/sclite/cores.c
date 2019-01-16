
#include "sctk.h"

void print_CORES(void *p){
  int i;
  CORES *cs = (CORES *)p;

  printf("\n    Correspondence Set:\n");
  for (i=0; i<cs->nword; i++){
    printf((i == 0) ? "        Words: " : "               ");
    print_WORD_wt((void *)(cs->words[i]));
  }
  printf("        Max_conf= %f  min_conf=%f  avg_conf=%f  avg_time=%f  avg_dur=%f  bias_nword= %f\n",
	 cs->max_conf, cs->min_conf, cs->avg_conf, cs->avg_time, cs->avg_dur, cs->bias_nword);
}

int equal_CORES(void *p1, void *p2){
  return 0;
}

void *get_CORES(void){
  CORES *tcs;
  alloc_singarr(tcs,1,CORES);
  tcs->use = 1;
  tcs->sum_conf = 0.0;
  tcs->max_conf = -1.0;
  tcs->min_conf = -1.0;
  tcs->avg_conf = -1.0;
  tcs->avg_time = -1.0;
  tcs->avg_dur = -1.0;
  tcs->nword = 0;
  tcs->bias_nword = 0.0;
  return((void *)tcs);
}

void release_CORES(void *p){
  int i;
  CORES *cs = (CORES *)p;
  
  cs->use --;
  if (cs->use > 0)    return;
  for (i=0; i<cs->nword; i++)
    release_WORD((void *)cs->words[i]);
  free_singarr(cs, CORES);  
}

int null_alt_CORES(void *p){
  return 0;
}

int opt_del_CORES (void *p){
  return 0;
}

void *copy_CORES(void *p){
  int i;
  CORES *tcs, *cs = (CORES *)p;

  tcs = get_CORES();
  for (i=0; i<cs->nword; i++)
    tcs->words[i] = (WORD *)copy_WORD((void *)cs->words[i]);
  tcs->max_conf = cs->max_conf;
  tcs->min_conf = cs->min_conf;
  tcs->avg_conf = cs->avg_conf;
  tcs->bias_nword = cs->bias_nword;
  tcs->nword = cs->nword;
  return(tcs);
}

void *make_empty_CORES(void *p){
  CORES *tcs;

  tcs = get_CORES();
  return(tcs);
}

int use_count_CORES(void *p, int i){
  ((CORES *)p)->use += i;
  return(((CORES *)p)->use);
}

/****************************************************************************/
/****************************************************************************/
static void add_to_node_list(NODE *node, void *ptr);
static int lookup_node(NODE **list, int cnt, NODE *node);
static NODE **node_list;
static int max_node = 0, num_node = 0;
NETWORK *Network_WORD_to_CORES(NETWORK *net);

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

NETWORK *Network_WORD_to_CORES(NETWORK *net){
    char *proc = "Network_WORD_to_CORES";
    int n;
    NETWORK *csnet;

    if (db > 0) printf("Entering: %s\n",proc);

    /* initialize variables */
    num_node = 0;
    max_node = net->node_count + 10;
    alloc_singZ(node_list,max_node,NODE *,(NODE *)0);
    Network_traverse(net,add_to_node_list,0,0,0,0);

    if ((csnet = Network_init(print_CORES, equal_CORES, release_CORES,
			      null_alt_CORES, opt_del_CORES, copy_CORES,
			      make_empty_CORES,use_count_CORES,
			      rsprintf("CORES net of %s",net->name))) == 0)
      return((NETWORK *)0);
    
    for (n=0; n<num_node; n++){
      CORES *cs[50];
      int ncs, cur_cs, pre_existing;
      ARC_LIST_ATOM *oa;

      ncs = 0;
      if (node_list[n]->out_arcs != NULL){
	/* build a correspondence set */
	for (oa = node_list[n]->out_arcs; oa != NULL; oa = oa->next){
	  /* lood for a pre-existing word in a correspondence set */
	  pre_existing = 0;
	  for (cur_cs = 0; cur_cs < ncs; cur_cs++)
	    if (equal_WORD2(cs[cur_cs]->words[0], oa->arc->data) == 0){
	      cs[cur_cs]->words[cs[cur_cs]->nword] =
		(WORD *)copy_WORD(oa->arc->data);
	      cs[cur_cs]->nword++;
	      pre_existing = 1;
	    }
	  if (!pre_existing){
	    cs[ncs] = (CORES *)get_CORES();
	    cs[ncs]->words[cs[ncs]->nword] = (WORD *)copy_WORD(oa->arc->data);
	    cs[ncs]->nword++;
	    ncs++;	    
	  }
	}      
      }
      if (ncs > 0){
	int x;
	/* add the first corresp set to the tail of the network */
	if (Network_add_arc_to_tail(csnet,(void *)cs[0]) > 0){
	  fprintf(stderr,"Error: Network_add_arc_to_tail failed in %s\n",
		  proc);
	  return((NETWORK *)0);
	}
	for (x = 1; x<ncs ; x++)
	  /* merge the other corresp set into the same place */
	  if (NETWORK_insert_arc_between_node(csnet,
				   csnet->stop_node->in_arcs->arc->from_node,
					      csnet->stop_node,
					      cs[x]) > 0)
	    return((NETWORK*)0);
      }
    }
    return(csnet);
}
