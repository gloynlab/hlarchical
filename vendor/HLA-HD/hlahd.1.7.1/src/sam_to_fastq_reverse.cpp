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

bool paired = false;
string sname,fname,rname,oname,tmpname;
bool gex = false;
bool gbest = false;
bool fuzzy = true;
bool unmap = false;
bool nohit = false;
bool PM = false;
string lname ="";
string gname = "";

int hsize = 7;
int ssize = 0;
int esize = 0;
int MINLEN = 0;

class Sam_To_Fastq_Reverse{

public:
  
  Sam_To_Fastq_Reverse(){
    int i,j,k,l,s,n,n2;
    ifstream ins,inf,inr,inl;
    ins.open(sname.c_str(),ios::in);
    if(!ins){
      cout << "Couldn't open sam file." << endl;
      exit(-1);
    }
    inf.open(fname.c_str(),ios::in);
    if(!inf){
      cout << "Couldn't open fastq file." << endl;
      exit(-1);
    }
    if(paired){
      inr.open(rname.c_str(),ios::in);
      if(!inr){
	cout << "Couldn't open fastq (paired) file." << endl;
	exit(-1);
      }
    }
 

    char sw;
    string word,word2,word3,words[10],sline;
    FASTQ_HASH **shf,**shr,*stmp;
    int hid,mid,mid2;
    int hitcnt;
    bool hit;
    int hcnt;
    hcnt = 1;
    for(i=0;i<hsize;i++){
      hcnt *= 10;
    } 
   
    shf = new FASTQ_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      shf[i] = new FASTQ_HASH();
      shf[i]->next = NULL;
    }  
    if(paired){
      shr = new FASTQ_HASH*[hcnt];
      for(i=0;i<hcnt;i++){
	shr[i] = new FASTQ_HASH();	
	shr[i]->next = NULL;    
      }
    }

    int *ls = new int[hsize];
    ls[0] = 1;
    for(i=1;i<hsize;i++)
      ls[i] = 10*ls[i-1];

    long int ficnt = 0;

    
    while(getline(ins,sline)){
      n = 0;
      word = GetWordtab(n,sline);
      if(word[0] == '@')
	continue;
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
      mid2 = mid;

      word2 = GetWordtab(n,sline);
      if(nohit){
	if(word2 != "*"){
	  continue;
	}
      }

      n2 = 0;
      word2 = GetWordcol(n2,word2);
      for(i=0;i<2;i++)
	GetWordtab(n,sline);

      if(!unmap){
	if(nohit){
	  if(GetWordtab(n,sline) != "*")
	    continue;
	}
	else if(GetWordtab(n,sline) == "*")
	  continue;
      }

      if(MINLEN > 0){
	if(!unmap){
	  for(i=0;i<3;i++)
	    GetWordtab(n,sline);
	}
	else
	  for(i=0;i<4;i++)
	    GetWordtab(n,sline);
	word3 = GetWordtab(n,sline);
	if(word3.size() < MINLEN)
	  continue;
      }

      if(mid2 >= 512)
	mid2 -= 512;
      if(mid2 >= 256)
	mid2 -= 256;
      if(mid2 >= 128){
	if(!paired){
	  cout << "This sam file is paired end file." << endl;
	  exit(0);
	}
	stmp = shr[hid];
      }
      else{
	stmp = shf[hid];
      }

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
	if(gex){
	  stmp->bhit = 1000000;
	  stmp->win = false;
	  stmp->gonly = false;
	  for(i=0;i<2;i++)
	    stmp->hcnt[i] = 0;
	}	
      }

      if(gex){	 
	hit = false;
	do{
	  word = GetWordtab(n,sline);
	  if(word.substr(0,2) == "NM"){
	    n2 = 0;
	    for(i=0;i<2;i++)
	      GetWordcol(n2,word);
	    hitcnt = atoi(GetWordtab(n2,word).c_str());
	    hit = true;
	    break;
	  }	      
	}while(n < sline.size());
	if(!hit)
	  continue;

	if(PM){
	  if(hitcnt == 0)
	    stmp->win = true;
	  continue;	  
	}

	if(word2 == gname){
	  stmp->hit = true;
	  stmp->hcnt[0]++;
	}
	else{
	  stmp->hcnt[1]++;
	}

	if(hitcnt < stmp->bhit){
	  stmp->bhit = hitcnt;
	  if(word2 == gname){
	    stmp->win = true;
	    stmp->gonly = true;
	  }
	  else{
	    stmp->win = false;
	    stmp->gonly = false;
	  }
	}
	else if(hitcnt == stmp->bhit){
	  if(word2 == gname)
	    stmp->win = true;
	  else
	    stmp->gonly = false;
	}
      }  
      ficnt++;
      if(ficnt % 1000000 == 0)
	cout << ficnt << " lines finished." << endl;
    }

    string outname;
    if(paired)
      outname = oname + ".R1.fastq";
    else
      outname = oname + ".fastq";
    ofstream out;
    out.open(outname.c_str(),ios::out);
    
    while(getline(inf,sline)){
      if(sline[0] != '@')
	continue;
      n = 1;
      word = GetWordsp(n,sline);
      hid = 0;
      word = word.substr(ssize,word.size()-ssize-esize);
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

      hit = false;
      stmp = shf[hid];
      while(stmp->next){
	stmp = stmp->next;
	if(stmp->id == word){
	  if(PM){
	    if(!stmp->win)
	      continue;
	  }
	  else if(gex){
	    if(!stmp->hit)
	      continue;
	    if(gbest){
	      if(!stmp->gonly)
		continue;
	      //if(stmp->hcnt[0] <= stmp->hcnt[1])
	      //continue;
	    }
	  }
	  out << sline << endl;
	  for(i=0;i<3;i++){
	    getline(inf,sline);
	    out << sline << endl;	    
	  }	 
	  hit = true;
	  break;
	}      
      }
      if(!hit && paired){
	stmp = shr[hid];
	while(stmp->next){
	  stmp = stmp->next;
	  if(stmp->id == word){
	    if(PM){
	      if(!stmp->win)
		continue;
	    }
	    else if(gex){
	      if(!stmp->hit)
		continue;
	      if(gbest){
		if(!stmp->gonly)
		  continue;
		//if(stmp->hcnt[0] <= stmp->hcnt[1])
		//continue;
	      }
	    }
	    out << sline << endl;
	    for(i=0;i<3;i++){
	      getline(inf,sline);
	      out << sline << endl;	    
	    }	 
	    break;
	  }
	}
      }
    }
    out.close();
    
    if(paired){
      outname = oname + ".R2.fastq";
      out.open(outname.c_str(),ios::out);
      while(getline(inr,sline)){
	if(sline[0] != '@')
	  continue;
	n = 1;
	word = GetWordsp(n,sline);
	word = word.substr(ssize,word.size()-ssize-esize);
	hid = 0;

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

	hit = false;
	stmp = shr[hid];
	while(stmp->next){
	  stmp = stmp->next;
	  if(stmp->id == word){	
	    if(PM){
	      if(!stmp->win)
		continue;
	    }
	    else if(gex){
	      if(!stmp->hit)
		continue;
	      if(gbest){
		if(!stmp->gonly)
		  continue;
		//if(stmp->hcnt[0] <= stmp->hcnt[1])
		// continue;
	      }
	    }
	    out << sline << endl;
	    for(i=0;i<3;i++){
	      getline(inr,sline);
	      out << sline << endl;
	    }	 
	    hit = true;
	    break;
	  }
	}
	if(!hit){
	  stmp = shf[hid];
	  while(stmp->next){
	    stmp = stmp->next;
	    if(stmp->id == word){
	      if(PM){
		if(!stmp->win)
		  continue;
	      }
	      else if(gex){
		if(!stmp->hit)
		  continue;	  
		if(gbest){
		  if(!stmp->gonly)
		    continue;
		  //if(stmp->hcnt[0] <= stmp->hcnt[1])
		  // continue;
		}
	      }
	      out << sline << endl;
	      for(i=0;i<3;i++){
		getline(inr,sline);
		out << sline << endl;	    
	      }	 
	      break;
	    }
	  }
	}
      }
      out.close();      
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
    if(sargs[i] == "-g"){
      i++;
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      gex = true;
      gname = sargs[i];
    }
    else if(sargs[i] == "--best"){     
      gbest = true;
    }
    else if(sargs[i] == "--list"){     
      i++;     
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      fuzzy = false;
      lname = sargs[i];
    }
    else if(sargs[i] == "--unmap"){        
      unmap = true;
    }
    else if(sargs[i] == "--nohit"){        
      nohit = true;
      MINLEN = 0;
    }
    else if(sargs[i] == "--PM"){  
      gex = true;
      PM = true;
    }
    else if(sargs[i] == "-H"){     
      i++;     
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      hsize = atoi(sargs[i].c_str());
      if(hsize < 2)
	hsize = 2;
      if(hsize > 10)
	hsize = 10;
    }
    else if(sargs[i] == "-S"){     
      i++;     
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      ssize = atoi(sargs[i].c_str());
      if(ssize < 0)
	ssize = 0;  
    }
    else if(sargs[i] == "-E"){     
      i++;     
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      esize = atoi(sargs[i].c_str());
      if(esize < 0)
	esize = 0;  
    }
    else if(sargs[i] == "-L"){     
      i++;     
      if(i==narg-1){
	wrong_args = true;
	break;
      }
      MINLEN = atoi(sargs[i].c_str());
      if(MINLEN < 0)
	MINLEN = 0;  
    }
    else{
      if(argscnt == 0){
	sname = sargs[i];
      }
      else if(argscnt == 1){
	fname = sargs[i];
      }
      else if(argscnt == 2){
	oname = sargs[i];
      }
      else if(argscnt == 3){
	tmpname = sargs[i];
	rname = oname;
	oname = tmpname;
	paired = true;
      }
      argscnt++;
    }
  }

  if(gbest || !fuzzy)
    if(!gex){
      cout << "Gene name is required for best mode." << endl;
      exit(0);
    }

  if (wrong_args || (argscnt != 3 && argscnt != 4)){
    cout << "Usage:" << endl;
    cout << args[0] << " [options] sam_file fastq_file fastq_file2(if paired) output" << endl;
    cout << "\t-g\t<string> [ specific gene name default: null ]" << endl;
    cout << "\t-L\t<int> [ minimum output length default: 0 ]" << endl;
    cout << "\t--best [ extract best only default: false ]" << endl;
    cout << "\t--unmap\t<string> [ extract unmap default: false ]" << endl;
    cout << "\t--nohit\t<string> [ extract unmap only default: false ]" << endl;
    cout << "\t--PM\t<string> [ perfect match only default: false ]" << endl;
    cout << "\t-H\t<int> [ Hash size default: 7 (Min2,Max10) ]" << endl;
    cout << "\t-S\t<int> [ same ID size default: 0 ]" << endl;
    cout << "\t-E\t<int> [ cut end size default: 0 ]" << endl;
    exit(0);
  }

  Sam_To_Fastq_Reverse *c = new Sam_To_Fastq_Reverse();
}
