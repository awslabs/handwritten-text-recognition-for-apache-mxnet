/* file: corresp.c
   Desc: Utility to locate common paths to multiple SCORES structures.
   Date: April 25, 1997
   */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct score_group_correspondence_struct{
  int *grp_ptr;
  int num_path;
  int **path_ptr;
} SC_COR_GRP;

typedef struct score_correspondence_struct{
  SCORES **scor;
  int nscor;
  int max_grp;
  int num_grp;
  int max_path;
  SC_COR_GRP **grp;
  
} SC_CORRESPONDENCE;

#ifdef __cplusplus
}
#endif

void locate_matched_data(SCORES *scor[], int nscor, SC_CORRESPONDENCE **corresp);
void find_matched_grp(SC_CORRESPONDENCE *corresp);
void find_matched_paths(SC_CORRESPONDENCE *corresp);
SC_CORRESPONDENCE *alloc_SC_CORRESPONDENCE(SCORES *scor[], int nsc);
void dump_SC_CORRESPONDENCE(SC_CORRESPONDENCE *corresp, FILE *fp);
void dump_paths_of_SC_CORRESPONDENCE(SC_CORRESPONDENCE *corresp, int maxlen, FILE *fp, int score_diff);
void PATH_multi_print(SCORES **scor, PATH **path_set, int npath, int maxlen, FILE *fp, int *refWord, int *refErrWord, AUTO_LEX *alex);

