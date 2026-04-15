#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

string lname;
string oname="result.txt";
string gname="";
string loname="";
bool paired=true;
int EMTH = 0;
int INMTH = 2;

int MIN_MM_COV = 1;
int hsize = 4;
int hsize2 = 4;
int *hpos;

int SORTMAX = 500;
int BESTMAX = 500;
int PLOT_MAX = 3;
int CORRECT_PMAX = 10;

double MIN_WEIGHT = 0.01;
double covth = 0.25;
double homoth = 4.0;
double fuzzyth = 0.9;
double fuzzyth2 = 0.7;
double fuzzyth3 = 0.03;
double pairth = 3.0;
double exthomoth = 20.0;
double mmco = 0.0;
double mhth = 0.25;

int MINMATCH = 50;
int MINLENGTH = 100;

bool pairmatch = false;
bool plotmreads = false;
bool plotbestlist = false;
string plrname = "";
string tallele="";

bool convert_read = false;
string cvname;

bool intron_estimate = false;

int FIRST_AMB_MAX = 10;
int AMB_REDUCE_CNT = 0;

#include "Tools.h"
#include "Define.h"
#include "Read.h"
#include "Count.h"
#include "Rank.h"
#include "Sort.h"
#include "Plot.h"
#include "Estimate.h"
#include "Select.h"
#include "Mcompare.h"
#include "Reselect.h"
#include "Tree.h"

class hla_estimation{

public:
  
  hla_estimation(){
    int i,ii,j,k,l,s,n,n2,ts;
    ifstream inl;
    string word,word2,word3,words[10],sline;
    int hitcnt,hitcntp,hitcntall;
    bool hit;
    int hcnt,hcnt2;
    
    hcnt = 1;
    for(i=0;i<hsize;i++){
      hcnt *= 10;
    } 
    hcnt2 = 1;
    for(i=0;i<hsize2;i++){
      hcnt2 *= 10;
    }
    hpos = new int[hsize2];
    
    FASTQ_HASH **sh,*stmp;
    sh = new FASTQ_HASH*[hcnt2];
    for(i=0;i<hcnt2;i++){
      sh[i] = new FASTQ_HASH();
      sh[i]->next = NULL;
    }  

    //2-Dec-2012
    FASTQ_HASH **ss;
    ss = new FASTQ_HASH*[hcnt2];
    for(i=0;i<hcnt2;i++){
      ss[i] = new FASTQ_HASH();
      ss[i]->next = NULL;
    }   
    
    List_Set *lset = new List_Set;
    int lcnt = Read_List_File(lset);

    clock_t start, end;
      
    int hlacnt = 0;
    HLA_HASH **hh,*htmp,*htmp2;    
    hh = new HLA_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      hh[i] = new HLA_HASH();
      hh[i]->next = NULL;
    }

    cout << "Search hash positions:" << endl;
    start = clock();
    Search_Hash_Position(lcnt,lset);
    end = clock();
    
    cout << "Read_Exon:" << endl; 
    start = clock();          
    hlacnt = Read_Exon_Map(sh,hh,lcnt,hcnt,hcnt2,lset); 
    end = clock();
    cout << "Read_Exon:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;

    start = clock();
    Read_All_Map(sh,hcnt,hcnt2);
    end = clock();
    cout << "Read_All:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;
 
    int **lmode;
    int *maxlm;
    Get_Exon_Mode(hh,hlacnt,hcnt,lcnt,lmode,maxlm);
      
    start = clock();
    Count_Map_Exon(hh,lcnt,hcnt,lset);
    //Count_Map_Intron(hh,lcnt,hcnt);
    end = clock();
    cout << "Count map:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;
    
    RANK_TREE *rtree = new RANK_TREE();
    rtree->hrank = new HLA_HASH*[hlacnt];
    rtree->sameid = new int[hlacnt];   
    for(i=0;i<2;i++){
      rtree->alist[i] = new ALIST();
      rtree->alist[i]->all = true;    
      rtree->alistcnt[i] = hlacnt;
    }
    rtree->used = new int[lcnt];
    rtree->finl = new int[lcnt];
    rtree->depth = 1;  
    for(i=0;i<lcnt;i++){
      rtree->used[i] = lset->used[i];
      rtree->finl[i] = lset->used[i];
    }
    rtree->hlabest = new HLA_HASH*[BESTMAX];
    rtree->diffcnt = new double*[BESTMAX];
    rtree->diffcnt2 = new double*[BESTMAX];
    rtree->mm = new int[BESTMAX];
    rtree->pmm = new int[BESTMAX];
    rtree->rsum = new int[BESTMAX];
    rtree->bestcnt = 0;
    for(i=0;i<BESTMAX;i++){
      rtree->diffcnt[i] = new double[BESTMAX];
      rtree->diffcnt2[i] = new double[BESTMAX];
    }
    rtree->covth = covth;
    rtree->next=NULL;
    rtree->fin = false;
    rtree->comp = true;

    start = clock();
    Create_Rank_Tree(rtree,hh,lcnt,hcnt,hlacnt,lmode,maxlm,lset);
    end = clock();
    cout << "Create_Rank_Tree:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;
    Check_Tree(rtree,hh,lset,lcnt,hcnt,hlacnt);

  }
};

int main(int narg, char** args){
  int i;
  string *sargs;
  
  int argscnt = 0;
  bool wrong_args = false;
  sargs = new string[narg];

  for(i=0;i<narg-1;i++){
    sargs[i] = string(args[i+1]);
  }
  
  for(i=0;i<narg-1;i++){   
    if(sargs[i] == "-o"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      oname = sargs[i];
    }
    else if(sargs[i] == "-m"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      BESTMAX = atoi(sargs[i].c_str());
      if(BESTMAX < 1)
	BESTMAX = 1;
    }
    else if(sargs[i] == "-w"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      MIN_WEIGHT = atof(sargs[i].c_str());
      if(MIN_WEIGHT < 0)
	MIN_WEIGHT = 0;
    }
    else if(sargs[i] == "--cv"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      covth = atof(sargs[i].c_str());
      if(covth > 1.0)
	covth = 1.0;
      if(covth < 0)
	covth = 0.0;
    }
    else if(sargs[i] == "--hth"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      homoth = atof(sargs[i].c_str());    
      if(homoth < 0)
	homoth = 0.0;
    }
    else if(sargs[i] == "--mm"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      mmco = atof(sargs[i].c_str());    
      if(mmco < 0)
	mmco = 0.0;
    }
    else if(sargs[i] == "--ml"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      MINMATCH = atoi(sargs[i].c_str());    
      if(MINMATCH < 1)
	MINMATCH = 1;
    }
    else if(sargs[i] == "-L"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      MINLENGTH = atoi(sargs[i].c_str());    
      if(MINLENGTH < 1)
	MINLENGTH = 1;
    }
    else if(sargs[i] == "--mhth"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      mhth = atof(sargs[i].c_str());    
      if(mhth < 0)
	mhth = 0.0;
    }
    else if(sargs[i] == "--fz1"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      fuzzyth = atof(sargs[i].c_str());    
      if(fuzzyth < 0)
	fuzzyth = 0.0;
    }
    else if(sargs[i] == "--fz2"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      fuzzyth2 = atof(sargs[i].c_str());    
      if(fuzzyth2 < 0)
	fuzzyth2 = 0.0;
    }
    else if(sargs[i] == "--fz3"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }      
      fuzzyth3 = atof(sargs[i].c_str());    
      if(fuzzyth3 < 0)
	fuzzyth3 = 0.0;
    }
    else if(sargs[i] == "--pread"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      plotmreads = true;
      plrname = sargs[i];
    }
    else if(sargs[i] == "--emiss"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      EMTH = atoi(sargs[i].c_str());   
    }
    else if(sargs[i] == "--target"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      tallele = sargs[i];
    }
    else if(sargs[i] == "--imiss"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      INMTH = atoi(sargs[i].c_str());   
    }
    else if(sargs[i] == "--estI"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      if(sargs[i]=="T") 
	intron_estimate = true;
      else if(sargs[i]=="F") 
	intron_estimate = false;
    }
    else{
      if(argscnt == 0){
	gname = sargs[i];
      }
      else if(argscnt == 1){
	lname = sargs[i];
      }
      else if(argscnt == 2){
	loname = sargs[i];
      }
      argscnt++;  
    }
  }
  
  if(wrong_args || argscnt != 3){
    cout << "Usage:" << endl;
    cout << args[0] << " [options] gene exon_intron_maplist maplist" << endl;
    cout << "\t-o\t<string> [ output default: result.txt ]" << endl;
    cout << "\t-m\t<int> [ maximum of estimated rank default: 500 ]" << endl;
    cout << "\t-L\t<double> [ minimum read length default: 500 ]" << endl; 
    cout << "\t-w\t<int> [ minimum weight default: 0.01 ]" << endl;
   cout << "\t--cv\t<double> [ coverage threshold [0,1] default: 0.25 ]" << endl;
    cout << "\t--hth\t<double> [ homo/hetero threshold (>= 0) default: 4.0 ]" << endl;
    cout << "\t--mm\t<double> [ mismatch coefficient (>= 0) default: 0.0 ]" << endl;
    cout << "\t--ml\t<double> [ minimum match length default: 50 ]" << endl;
    //cout << "\t--fz1\t<double> " << endl;
    //cout << "\t--fz2\t<double> " << endl;
    //cout << "\t--fz3\t<double> " << endl;
    cout << "\t--imiss\t<int> [ maximum missmatch for intron default: 2 ]" << endl;
    cout << "\t--emiss\t<int> [ maximum missmatch for exon default: 0 ]" << endl;
    cout << "\t--mhth\t<int> [ multiple gene hit (>= 0.0) default: 0.25 ]" << endl; 
    cout << "\t--pread\t<string> [ plot reads to <string> default: false ]" << endl;       
    //cout << "\t--target\t<string> " << endl;
    cout << "\t--estI\tT/F [estimate intron default: false ]" << endl;
    exit(0); 
  }
  cout << "Start hla_est (version 1.7.0)" << endl;
  hla_estimation *c = new hla_estimation();
}
