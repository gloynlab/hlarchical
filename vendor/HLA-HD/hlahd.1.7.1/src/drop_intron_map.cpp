/*******************************************************************************
 *  sam_to_fasta_reverse, written by Shuji Kawaguchi, ph.D.                    *
 *  Center for Genomic Medicine, Graduate School of Medicine, Kyoto University *
 *  Copyright (C) 2015-2017 All Rights Reserved.                               *
********************************************************************************/

#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "STFR_Tools.h"
#include "STFR_define.h"

string ename[2],iname[2],fname[2],oname;
int hsize = 7;
int ssize = 0;

class Drop_Intron_Map{

public:
  
  Drop_Intron_Map(){
    int i,j,k,l,s,n,n2;
    ifstream exi[2],ini[2],fi[2];

    char sw;
    string word,word2,sline,slines[4];
    FASTQ_HASH **shf,*stmp;
    int hid,mid;
    int hitcnt;
    bool hit;
    int hcnt;
    hcnt = 1;
    for(i=0;i<hsize;i++){
      hcnt *= 10;
    } 
       
    for(i=0;i<2;i++){
      exi[i].open(ename[i].c_str(),ios::in);
      if(!exi[i]){
	cout << "Couldn't open " << ename[i] << endl;
	exit(-1);
      }
    }    
    for(i=0;i<2;i++){
      ini[i].open(iname[i].c_str(),ios::in);
      if(!ini[i]){
	cout << "Couldn't open " << iname[i] << endl;
	exit(-1);
      }
    }
    for(i=0;i<2;i++){
      fi[i].open(fname[i].c_str(),ios::in);
      if(!fi[i]){
	cout << "Couldn't open " << fname[i] << endl;
	exit(-1);
      }
    }
  
    shf = new FASTQ_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      shf[i] = new FASTQ_HASH();
      shf[i]->next = NULL;
    }  

    int *ls = new int[hsize];
    ls[0] = 1;
    for(i=1;i<hsize;i++)
      ls[i] = 10*ls[i-1];
    
    for(l=0;l<2;l++){
      while(getline(exi[l],sline)){
	if(sline[0] == '@')
	  continue;
	n = 0;
	word = GetWordtab(n,sline);
	hid = 0;
	word = word.substr(ssize,word.size()-ssize);
	j = word.size();
	k = 0;
	i = 0;
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
	
	mid = atoi(GetWordtab(n,sline).c_str());
	
	word2 = GetWordtab(n,sline);
	if(word2 == "*"){
	  continue;
	}

	stmp = shf[hid];
	
	hit = false;
	while(stmp->next){
	  stmp = stmp->next;
	  if(stmp->id == word){
	    hit = true;
	    break;
	  }
	}
	if(!hit){
	  stmp->next = new FASTQ_HASH();
	  stmp = stmp->next;
	  stmp->next = NULL;
	  stmp->id = word;
	  for(i=0;i<2;i++){
	    stmp->ehit[i] = false;
	  }
	}
	stmp->ehit[l] = true;
      }
    }
    for(l=0;l<2;l++)
      exi[l].close();
  
    for(l=0;l<2;l++){
      while(getline(ini[l],sline)){
	if(sline[0] == '@')
	  continue;
	n = 0;
	word = GetWordtab(n,sline);
	hid = 0;
	word = word.substr(ssize,word.size()-ssize);
	j = word.size();
	k = 0;
	i = 0;
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
	
	mid = atoi(GetWordtab(n,sline).c_str());
	
	word2 = GetWordtab(n,sline);
	if(word2 == "*"){
	  continue;
	}

	hit = false;
	do{
	  word2 = GetWordtab(n,sline);
	  if(word2.size() > 5){
	    if(word2.substr(0,5)=="MD:Z:"){
	      hit = true;
	      break;
	    }
	  }
	}while(n < sline.size());

	if(!hit)
	  continue;
	
	word2 = word2.substr(5,word2.size()-5);

	if((word2[0]=='0' && word2[1]=='N') || word2[word2.size()-1]=='0' && word2[word2.size()-2]=='N'){ 
	  stmp = shf[hid];	
	  hit = false;
	  while(stmp->next){
	    stmp = stmp->next;
	    if(stmp->id == word){
	      hit = true;
	      break;
	    }
	  }
	  if(!hit){
	    stmp->next = new FASTQ_HASH();
	    stmp = stmp->next;
	    stmp->next = NULL;
	    stmp->id = word;
	    for(i=0;i<2;i++){
	      stmp->ehit[i] = false;
	    }
	  }	  
	  stmp->ehit[l] = true;
	}
      }
    }
    for(l=0;l<2;l++)
      ini[l].close();

    string outname;
    ofstream out[2];
    
    outname = oname + ".R1.fastq";   
    out[0].open(outname.c_str(),ios::out);
    outname = oname + ".R2.fastq";   
    out[1].open(outname.c_str(),ios::out);
    
    for(l=0;l<2;l++){
      while(getline(fi[l],slines[0])){
	for(i=0;i<3;i++)
	  getline(fi[l],slines[i+1]);	
	n = 1;
	word = GetWordsp(n,slines[0]);
	hid = 0;
	word = word.substr(ssize,word.size()-ssize);
	j = word.size();
	k = 0;
	i = 0;
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

	stmp = shf[hid];	
	hit = false;
	while(stmp->next){
	  stmp = stmp->next;
	  if(stmp->id == word){
	    hit = true;
	    break;
	  }
	}
	if(!hit){
	  continue;
	}
	  
	if(stmp->ehit[0] || stmp->ehit[1]){
	  for(i=0;i<4;i++){
	    out[l] << slines[i] << endl;
	  }
	}
	
      }
    }
    for(l=0;l<2;l++){
      fi[l].close();
    }

    for(l=0;l<2;l++){
      out[l].close();
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
      ename[0] = sargs[i];
    }else if(argscnt == 1){
      ename[1] = sargs[i];
    }else if(argscnt == 2){
      iname[0] = sargs[i];
    }else if(argscnt == 3){
      iname[1] = sargs[i];
    }else if(argscnt == 4){
      fname[0] = sargs[i];
    }else if(argscnt == 5){
      fname[1] = sargs[i];
    }else if(argscnt == 6){
      oname = sargs[i];
    }
    argscnt++;    
  }

  
  if (argscnt != 7){
    cout << "Usage:" << endl;
    cout << args[0] << " sam_file_exon_R1 sam_file_exon_R2 sam_file_intron_R1 sam_file_intron_R2 fastq_file_R1 fastq_file_R2 output" << endl;  
    exit(0);
  }

  Drop_Intron_Map *c = new Drop_Intron_Map();
}
