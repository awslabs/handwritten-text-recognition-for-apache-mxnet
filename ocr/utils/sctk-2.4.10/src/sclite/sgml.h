
#define MAX_TAGS 10
#define MAX_ATTRIB 30

typedef struct sgml_label_struct{
    TEXT *name;
    int attrib_num;
    TEXT *attrib_name[MAX_ATTRIB];
    TEXT *attrib_value[MAX_ATTRIB];
} SGML_LABEL;

typedef struct sgml_tags{
    int num_tags;
    SGML_LABEL tags[MAX_TAGS];
} SGML;

void init_SGML(SGML *sg);
void dump_SGML_tag(SGML *sg, int n, FILE *fp);
int add_SGML_tag(SGML *sg, TEXT *str);
TEXT *delete_SGML_tag(SGML *sg, TEXT *str);
TEXT *get_SGML_attrib(SGML *sg, int tn, TEXT *sname);

