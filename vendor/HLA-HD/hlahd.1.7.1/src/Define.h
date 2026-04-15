struct List_Set{
 int lcnt;
 string *exfile[2];
 string *infile[2];
 string *exname;
 string *inname;
 string *faefile;
 string *faifile;
 int *intidl;
 int *intidr;
 int *exidl;
 int *exidr;
 int *lpos,*rpos;
 int *used;
};

struct GENE_LIST{
 string name;
 string einame;
 int freq;
 GENE_LIST *next;
 int id;
};

struct FASTQ_HASH{
  string id;
  string sline;
  double w[2];
  bool paired;
  bool ipaired;
  bool edge[2];
  int max[2];
  int *freq[2];
  int ts[2];
  bool inhit;
  bool *inid[2];
  bool ehit[2];
  string code[2];
  FASTQ_HASH *next;
  int minhit[2];
  GENE_LIST *gl[2];
};

struct FASTQ_CHAIN{
  FASTQ_HASH *stmp;
  FASTQ_CHAIN *next;
  bool hit;
  int *ilpos;
  int *irpos;
};


struct AL_2D{
  string aname;
  int cnt;
  AL_2D *next;
};

struct MPED_FQ{
  FASTQ_HASH *fq;
  int s,e;
  bool paired,opaired[2];
  bool edge;
  int NS,NE;
  MPED_FQ *pair;
  MPED_FQ *next;
};

struct IE_PAIR_MPED_FQ{
  FASTQ_HASH *fq;
  MPED_FQ *mp[2];
  int ex[2];
  int lcnt[2];
  bool hit[3];
  IE_PAIR_MPED_FQ *next;
};

struct HLA_HASH{
  string id;
  string aname;
  string einame;
  HLA_HASH *next;
  HLA_HASH **samee,**samei;
  int pcnt;
  bool plot;
  int *len,*ilen;  
  int *es,*ee,*is,*ie;
  bool *hit,*edge,*ihit;
  bool **hitcnt,**hitcntp,**ihitcnt,**ihitcntp;
  int *nmcnt,*nmcntp;
  int *inmcnt,*inmcntp;
  int *rsum;
  int *irsum;
  double *exvcnt,*exv2cnt;
  double *invcnt;
  int *nlen,*inlen;
  int *mm,*mmp;
  int *imm,*immp,*imiss;
  int tmpmm,tmpmmp,tmpnm,tmpnmp;
  int tmpv,tmplen,tmprsum;
  double tmpvcnt,tmpv2cnt;
  MPED_FQ **mp[2],**mptmp[2];
  MPED_FQ **imp[2],**imptmp[2];
  char **code,**icode;
  string *esticode;
  int *estD,*estI;
  int *estis,*estie; 
};

struct ALIST{
  bool all;
  HLA_HASH *hla;
  ALIST *next;
};

struct RANK_TREE{
  HLA_HASH **hrank;
  HLA_HASH **hlabest;
  int hithla;
  int *sameid;
  int sameidcnt;
  int nextcnt;
  int *used;
  int *finl;
  int depth;
  double **diffcnt,**diffcnt2;  
  int *mm,*pmm;
  int *rsum;
  int bestcnt;
  double *readcnt,*readcnt2,*readcnt3;
  int **bestpair;
  int paircnt;
  int *rankid[2];
  int ranksum;
  ALIST *alist[2];
  int alistcnt[2];
  double covth;
  bool fin;
  int bestucnt;
  RANK_TREE **next;
  bool comp;
};
