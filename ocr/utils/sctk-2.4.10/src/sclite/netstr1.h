/* file netstr1.h                         */
/* structure #1 for representing networks */
/* composed of nodes and arcs.            */

#ifndef _NETWORK_DEF_
#define _NETWORK_DEF_

#define MAX_NODES 100
#define MAX_ARCS 100

struct ARC_FUNCTIONS
  { void (*print)(void *);        char *print_name;
    void (*destroy)(void *);      char *destroy_name;
    int (*equal)(void *, void *); char *equal_name;
    int (*is_null_alt)(void *);   char *is_null_alt_name;
    int (*is_opt_del)(void *);    char *is_opt_del_name;
    void *(*copy)(void *);        char *copy_name;
    void *(*make_empty)(void *);  char *make_empty_name;
    int (*use_count)(void *, int); char *use_count_name;
  };
typedef struct ARC_FUNCTIONS ARC_FUNC;

struct ARC1
{void *data;
 int weight;
 struct NETWORK1 *net;
 struct NODE1 *from_node, *to_node;
};
typedef struct ARC1 ARC;

struct ARC_LIST_ATOM1
  {ARC *arc; /* ocontents of this atom */
   struct ARC_LIST_ATOM1 *next, *prev;
  };
typedef struct ARC_LIST_ATOM1 ARC_LIST_ATOM;

struct NODE1
  {char *name;
   ARC_LIST_ATOM *in_arcs, *out_arcs;
   struct NETWORK1 *net;
   boolean start_state, stop_state, flag1, flag2;
  };
typedef struct NODE1 NODE;


 struct NETWORK1
   {char *name;
    int highest_nnode_name;
    int node_count;
    int arc_count;
    ARC_FUNC arc_func;
    NODE *start_node;
    NODE *stop_node;
   };
 typedef struct NETWORK1 NETWORK;

enum edits {DEL=1,SUB,INS}; 

typedef struct CELL_struct {
    float min_d;         /* the current minimum distance */
    short back_a,        /* array r index to CELL with min transition */
          back_b;        /* array h index to CELL with min transition */
} CELL;

typedef struct ARCSET_struct {
    int max;             /* current size of the data structure */
    int num;             /* current size of the networks being aligned */
    ARC **arcs;          /* list of ARC pointers to A NET */
} ARCSET;

typedef struct NET_ALIGN_struct {
    ARCSET arcset_a,     /* structure or arc list information */   
           arcset_b; 
    CELL **cell;         /* 2-Dim table, containing Cell structures */
} NET_ALIGN;

#endif

#define NULL_NETWORK (NETWORK *)0

extern void           dump_network(NETWORK *net);
extern void           dump_network_arcs(NETWORK *net);
extern void           free_network(NETWORK *net);
extern NETWORK       *init_network2(char *name, char *s0, int *perr);
extern void           literalize_network(NETWORK *net, NODE *node, int *perr);
extern void           dump_node(NODE *);
extern void	      dump_arc(ARC *arc);
extern NODE          *make_node(char *name, NETWORK *net, ARC *from_arc, ARC *to_arc, int *highest_nnode_name, int *perr);
extern ARC_LIST_ATOM *add_to_arc_list(ARC_LIST_ATOM *list_atom, ARC *arc, int *perr);
extern boolean        arcs_equal(ARC *arc1, ARC *arc2);
extern ARC           *make_arc(void *data, NODE *from_node, NODE *to_node, int *perr);
extern void           deflag2_node_r(NODE *node);
extern ARC_LIST_ATOM *del_from_arc_list(ARC_LIST_ATOM **plist, ARC *arc, int *perr);
extern void           kill_arc(ARC *arc1);
extern void           kill_node2(NODE *node1);

/* an attempt to define an abstract Data type for Networks */
/* The following functions are required to manipulate the data within 
   the network :

   void arc_data_prn(void *)   -> accepts a void structure ptr, and prints out
                                  the values stored in the network.
   void arc_data_free(void *) ->  free's the memory associated with data struct

*/

#define NT_For        0x0001   /* traverse from the START Node, Default */
#define NT_Back       0x0002
#define NT_Breadth    0x0004
#define NT_Depth      0x0008
#define NT_Inorder    0x0010
#define NT_CA_For     0x0100   /* Call on foward arcs from a node, Default */
#define NT_CA_Back    0x0200
#define NT_Verbose    0x8000

extern NETWORK *Network_init_from_net(NETWORK *net, char *name);
extern NETWORK *Network_init(void (*arc_data_prn)(void *), int (*arcs_equal)(void *, void *), void arc_data_destroy(void *), int (*is_null_alt)(void *), int (*is_opt_del)(void *), void *(*copy)(void *), void *(*make_empty)(void *), int (*use_count)(void *, int), char* desc);
extern NETWORK *Network_copy(NETWORK *in_net);
extern int      Network_destroy(NETWORK *net);
extern int      Network_add_arc_to_head(NETWORK *net, void *str);
extern int      Network_add_arc_to_tail(NETWORK *net, void *str);
extern int      Network_delete_arc(ARC *arc);
extern void     Network_dump(NETWORK *net);
extern void     Network_sgml_dump(NETWORK *in_net, FILE *fp);
extern void     Network_traverse(NETWORK *net, void (*node_op)(NODE *, void *), void *node_data, void (*arc_op)(ARC *, void *), void *arc_data, int mode);
extern int      Network_merge_network(NODE *from_node, NODE *to_node, NETWORK *from_net);
extern void     print_node(NODE *node, void *);
extern void     print_arc(ARC *arc, void *);
extern void     delete_null_arcs(ARC *arc, void *p);
extern int      Network_fully_connect(NETWORK *net, int connect_factor, void *(*append)(void *, void *));
extern int      Network_dpalign(NETWORK *ref,NETWORK *hyp, float wwd(void *,void *, int (*cmp)(void *, void *)), PATH **outpath, int include_nulls);
extern int Network_dpalign_n_networks(NETWORK **in_nets, int n_nets, float wwd(void *, void *, int (*cmp)(void *, void *)), NETWORK **out_net, void *null_alt);
extern void     cleanup_NET_ALIGN(void);
extern int      Network_add_net_to_tail(NETWORK *net, NETWORK *mnet);
extern int      Network_add_net_to_head(NETWORK *net, NETWORK *mnet);
extern NETWORK *Network_create_from_WTOKE(WTOKE_STR1 *wt,int start,int end, char *name, void (*aprn)(void *), int (*aequal)(void *, void *),  void (*adestroy)(void *), int (*is_null_alt)(void *), int (*is_opt_del)(void *), void *(*copy)(void *), void *(*make_empty)(void *), int (*use_count)(void *, int), int left_to_right);
extern NETWORK *Network_create_from_TEXT(TEXT *text, char *name, void (*arc_data_prn)(void *), int (*arcs_equal)(void *, void *), void (*arc_data_destroy)(void *), int (*is_null_alt)(void *), int (*is_opt_del)(void *), void *(*copy)(void *), void *(*make_empty)(void *), int (*use_count)(void *, int));
int Network_fully_connect_cond(NETWORK *net, int connect_factor, void *(*append)(void *, void *), int (*test)(void *, void *), void *test_data);
int NETWORK_insert_arc_between_node(NETWORK *, NODE *, NODE *, void *);
int NETWORK_insert_arc_after_node(NETWORK *, NODE *, void *);
int NETWORK_insert_arc_before_node(NETWORK *, NODE *, void *);

NET_ALIGN *alloc_NET_ALIGN(NET_ALIGN *, int nref, int nhyp);
void alloc_ARCSET(ARCSET *arcset, int n);
void free_ARCSET(ARCSET *arcset);
    
#define NET_ALIGN_NULL  (NET_ALIGN *)0
#define CELL_NULL       (CELL *)0

void add_to_arcset(ARC *arc, void *ptr);
int find_arcset_id(ARCSET *arcset, ARC *arc, int from);

void dump_NET_ALIGN(NET_ALIGN *net_ali,FILE *fp);
void Network_delete_null_arcs(NETWORK *out_net);
void mfalign_ctm_files(char **hypname, int nhyps, int time_align, int case_sense, int feedback, void (*callback)(NETWORK *, char *, char *),double silence_dur);
int find_common_silence(WTOKE_STR1 **ctms, int nctm, int *conv_end, int *sil_end, double silence_dur);    
void locate_next_file_channel(WTOKE_STR1 **ctms, int nctms, FILE **files, char **hypname, int *eofs, int *conv_end, int case_sense, int feedback);

