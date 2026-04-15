#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "STFR_Tools.h"
#include "STFR_define.h"

string fnames,fnameo;
int hsize = 6;

class Get_Diff_Fasta{

public:
  
  Get_Diff_Fasta(){
    int i,j,k,l,s,n,n2;
    ifstream ins,ino;
    string word,sline;
    ins.open(fnames.c_str(),ios::in);
    if(!ins){
      cout << "Couldn't open target file." << endl;
      exit(-1);
    }
    ino.open(fnameo.c_str(),ios::in);
    if(!ino){
      cout << "Couldn't open  file." << endl;
      exit(-1);
    }
    
    bool hit;
    int hcnt,hid;
    char sw;
    hcnt = 1;
    for(i=0;i<hsize;i++){
      hcnt *= 10;
    } 

    
    FASTQ_HASH **sh,*stmp;
    sh = new FASTQ_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      sh[i] = new FASTQ_HASH();
      sh[i]->next = NULL;
    }  
  

    int *ls = new int[hsize];
    ls[0] = 1;
    for(i=1;i<hsize;i++)
      ls[i] = 10*ls[i-1];


    while(getline(ino,sline)){
      n = 0;
      word = GetWordsp(n,sline);
      j = word.size();
      k = 0;
      i = 0;
      hid = 0;
      while(1){
	if(j-i-1 < 0)
	  break;
	sw = word[j-i-1];
	if ( sw < '0' || sw > '9' ) {
	  i++;
	  continue;
	} else {	
	  s = (int)(sw - '0');
	}
	hid += ls[k]*s;
	k++;
	if(k == hsize)
	  break;
	i++;
      }

      hit = false;
      stmp = sh[hid];
      while(stmp->next){
	stmp = stmp->next;
	if(stmp->id==word){
	  hit = true;
	  break;
	}
      }
      if(!hit){
	stmp->next = new FASTQ_HASH();
	stmp = stmp->next;
	stmp->next = NULL;
	stmp->id = word;
      }      
      for(i=0;i<3;i++)
	getline(ino,sline);
    }
    
    while(getline(ins,sline)){
      n = 0;
      word = GetWordsp(n,sline);
      j = word.size();
      k = 0;
      i = 0;
      hid = 0;
      while(1){
	if(j-i-1 < 0)
	  break;
	sw = word[j-i-1];
	if ( sw < '0' || sw > '9' ) {
	  i++;
	  continue;
	} else {	
	  s = (int)(sw - '0');
	}
	hid += ls[k]*s;
	k++;
	if(k == hsize)
	  break;
	i++;
      }

      hit = false;
      stmp = sh[hid];
      while(stmp->next){
	stmp = stmp->next;
	if(stmp->id==word){
	  hit = true;
	  break;
	}
      }
      if(!hit){
	cout << sline << endl;
	for(i=0;i<3;i++){
	  getline(ins,sline);
	  cout << sline << endl;
	}
      }
      else{
	for(i=0;i<3;i++)
	  getline(ins,sline);
      }
    }    
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
    if(argscnt == 0){
      fnames = sargs[i];
    }
    else if(argscnt == 1){
      fnameo = sargs[i];
    } 
    argscnt++;    
  }

  if (argscnt != 2){
    cout << "Usage:" << endl;
    cout << args[0] << " fastq_target fastq_hit" << endl;  
    exit(0);
  }

  Get_Diff_Fasta *c = new Get_Diff_Fasta();
}
