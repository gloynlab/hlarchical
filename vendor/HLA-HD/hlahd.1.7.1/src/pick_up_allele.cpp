#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "Tools.h"

string gname,rname,fname="";

bool compress=true;
int cpd=3;

class Pick_up_allele{

public:
  
  Pick_up_allele(){
    int i,j,k,l,s,n,n2;
    ifstream in,in2;
    string word,word2,word3,sline;
    int pcnt=0;
    int amgcnt=0;
    int redcnt=0;
    string *pairs[2];
    string *amgs[2];
    string tmps;

    struct FREQ{
      long int cnt;
      string aname;
      FREQ *next;
    };

    in.open(rname.c_str(),ios::in);
    if(in){
      while(getline(in,sline)){
	n = 0;
	word = GetWordtab(n,sline);
	if(word == "No candidate."){
	  pcnt = 1;
	  amgcnt = 1;
	  for(i=0;i<2;i++){
	    pairs[i] = new string[pcnt];
	  }
	  pairs[0][0] = "NULL";
	}
	else if(word == "#Best allele pair"){
	  word = GetWordtab(n,sline);
	  pcnt = atoi(word.c_str());	  
	  for(i=0;i<2;i++){
	    pairs[i] = new string[pcnt];
	  }
	  for(i=0;i<pcnt;i++){
	    getline(in,sline);
	    n = 0;
	    for(j=0;j<2;j++){
	      word = GetWordtab(n,sline);
	      if(word.size() > 0){
		pairs[j][i] = word;
	      }
	      else{
		pairs[j][i] = "-";
	      }
	    }
	  }
	}
      }
    }
    else{
      cout << "Couldn't read result file." << endl;
      exit(0);
    }

    double maxp;
    string **comps[2];
    int *ccnt[2];
    bool hit;
    if(pairs[0][0] == "NULL"){   
      cout << gname << "\tNot typed\tNot typed" << endl;
    }
    else{
      for(i=0;i<2;i++){
	comps[i] = new string*[pcnt];
	ccnt[i] = new int[pcnt];
      }
      for(i=0;i<pcnt;i++){
	for(j=0;j<2;j++){
	  if(pairs[j][i] == "-"){
	    comps[j][i] = new string[1];
	    comps[j][i][0] = "-";
	    ccnt[j][i] = 1;
	  }
	  n = 0;
	  k = 0;	  
	  do{
	    GetWordcam(n,pairs[j][i]);
	    k++;
	  }while(n < pairs[j][i].size());
	  comps[j][i] = new string[k];
	  ccnt[j][i] = 0;
	  n = 0;
	  for(l=0;l<k;l++){
	    comps[j][i][l] = "";
	    word = GetWordcam(n,pairs[j][i]);
	    n2 = 0;
	    for(s=0;s<cpd;s++){
	      GetWordcol(n2,word);
	    }
	    if(n2 < word.size()){
	      word2 = word.substr(0,n2-1);
	    }
	    else
	      word2 = word;
	    hit = false;
	    for(s=0;s<l;s++){
	      if(word2 == comps[j][i][s]){
		hit = true;
		break;
	      }
	    }
	    if(!hit){
	      comps[j][i][ccnt[j][i]] = word2;
	      ccnt[j][i]++;
	    }
	  }
	}
      }

      int psum=0;
      for(i=0;i<pcnt;i++){	
	psum += ccnt[0][i]*ccnt[1][i];
      }    
      string *pair[2];
      for(i=0;i<2;i++)
	pair[i] = new string[psum];
      k=0;
      for(i=0;i<pcnt;i++){
	for(j=0;j<ccnt[0][i];j++){
	  for(l=0;l<ccnt[1][i];l++){
	    pair[0][k] = comps[0][i][j];
	    pair[1][k] = comps[1][i][l];
	    k++;
	  }
	}
      }

      n = 0;
      string g = GetWordast(n,pair[0][0]);
      g = g.substr(4,g.size()-4);

      if(fname.size() > 0){
	in2.open(fname.c_str(),ios::in);
	if(in2){
	  FREQ *ftmp,*ftop;
	  ftop = new FREQ();
	  ftop->next = NULL;
	  ftmp = ftop;
	  while(getline(in2,sline)){
	    n = 0;
	    word = GetWordtab(n,sline);
	    n2 = 0;
	    word3 = GetWordast(n2,word);
	    if(word3 != g)
	      continue;
	    n2 = 0;
	    for(s=0;s<cpd;s++){
	      GetWordcol(n2,word);
	    }
	    if(n2 < word.size()){
	      word2 = word.substr(0,n2-1);
	    }
	    else
	      word2 = word;
	    hit = false;
	    ftmp = ftop;
	    while(ftmp->next){
	      ftmp = ftmp->next;
	      if(word2 == ftmp->aname){
		hit = true;
		break;
	      }
	    }
	    if(!hit){
	      ftmp->next = new FREQ();
	      ftmp = ftmp->next;
	      ftmp->next = NULL;
	      ftmp->aname = word2;
	      ftmp->cnt = atol(GetWordtab(n,sline).c_str());
	    }
	    else{
	      ftmp->cnt += atol(GetWordtab(n,sline).c_str());
	    }
	  }

	  long int *fcnt = new long int[psum];
	  long int maxf = -1;
	  string tmps;
	  long int csum[2];
	  for(i=0;i<psum;i++){
	    csum[0] = 0; csum[1] = 0;
	    for(j=0;j<2;j++){
	      if(pair[j][i] == "-"){
		csum[j] = 1;
		continue;
	      }
	      tmps = pair[j][i].substr(4,pair[j][i].size()-4);
	      ftmp = ftop;
	      while(ftmp->next){
		ftmp = ftmp->next;
		if(ftmp->aname.size() <= tmps.size()){
		  if(tmps.substr(0,ftmp->aname.size()) == ftmp->aname){
		    csum[j] += ftmp->cnt;
		  }
		}
	      }	  	  
	      if(csum[j] == 0)
		csum[j] = 1;
	    }
	  
	    fcnt[i] = csum[0]*csum[1];
	    if(fcnt[i] > maxf)
	      maxf = fcnt[i];
	  }
	
	  for(i=0;i<psum-1;i++){
	    for(j=i+1;j<psum;j++){
	      if(fcnt[i] < fcnt[j]){
		for(k=0;k<2;k++){
		  tmps = pair[k][i];
		  pair[k][i] = pair[k][j];
		  pair[k][j] = tmps;
		}
		l = fcnt[i];
		fcnt[i] = fcnt[j];
		fcnt[j] = l;	      
	      }
	    }
	  }
	  for(i=0;i<psum;i++)
	    if(fcnt[i] < maxf)
	      break;
	  psum = i;
	}
      }    
      
      cout << gname;
      for(i=0;i<psum;i++){
	  cout << "\t" << pair[0][i] << "\t" << pair[1][i];
      }
      cout << endl; 
    }
    in.close();
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
    if(sargs[i] == "-f"){
      i++;
      if(i==narg-1){
        wrong_args = true;
        break;
      }      
      fname = sargs[i];
    }
    else{
      if(argscnt == 0){
	gname = sargs[i];
      }
      else if(argscnt == 1){
	rname = sargs[i];
      }
      argscnt++;      
    }
  }
  
  if(wrong_args || (argscnt != 2)){
    cout << "Usage:" << endl;
    cout << args[0] << " [-f freq_file] gname resultfile" << endl;
    exit(0);
  }

  Pick_up_allele *c = new Pick_up_allele();
}
