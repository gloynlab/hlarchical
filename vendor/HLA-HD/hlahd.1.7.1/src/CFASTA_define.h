const int CL_HLA_NUM=9;
string CL_HLA[CL_HLA_NUM] = {"A","B","C","DRB1","DQA1","DQB1","DPA1","DPB1","DRA"};
int CL_HLA_EX[CL_HLA_NUM] = {8,7,8,6,4,6,4,5,4};

struct NUCL{
  string nuc;
  NUCL *next;
};

struct PROPERTY;

struct GENELC{
  string g;
  string odir;
  int ex;
  int cnt;
  int npcnt;
  int egcnt;
  int mglen;
  int mnlen;
  int *melen;
  GENELC *next;
  ofstream outl,outg,outgn,outn,outu5,outu3,outall,outgnutr;
  ofstream outog,outogn;
  ofstream *oute,*outi;
  ofstream *outien,*outin,*outienl,*outienr; //28-Aug-2014
  ofstream *outen;
  PROPERTY **sameex,**samein,**sameutr;
};

struct PROPERTY{
  string g;
  string name;
  string al;
  bool par;
  bool pse,unuse;
  bool eg;
  int a[4];  
  int excnt;
  int length;
  int *es,*ee;
  int *eid;
  string trans;
  NUCL *ntop;
  PROPERTY *next;
  char *seq;
  GENELC *glc;
};


