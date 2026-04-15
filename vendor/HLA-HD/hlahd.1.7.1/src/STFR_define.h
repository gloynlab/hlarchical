struct FASTQ_HASH{
  string id;
  bool hit,win;
  bool gonly;
  bool *best;
  int bhit;
  int s;
  int tag;
  int hcnt[4];
  int *gcnt;
  string G;
  bool ehit[2];
  FASTQ_HASH *next;
};

struct GLIST{
  string name;
  int id;
  int acnt;
  GLIST *next;
};
