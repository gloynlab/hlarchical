#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include <string.h>
#include <stdlib.h>
#include <math.h>

string iname;
int lsize;
bool alt = false;

#include "Tools.h"

class Cut_line{
public:
  
  Cut_line(){
    int i,j,k;
    ifstream in;
    string sline;
    string outname;
    ofstream *out;

    in.open(iname.c_str(),ios::in);
    if(!in){
      cout << "Couldn't open file." << endl;
    }  
    out = new ofstream[lsize];
    for(i=0;i<lsize;i++){
      outname = iname + "." + deci_to_st(i);
      out[i].open(outname.c_str(),ios::out);
    }
    i = 0;
    while(getline(in,sline)){
      out[i] << sline << endl;
      i++;
      if(i==lsize)
	i = 0;
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
    if(sargs[i] == "-a"){
      alt = true;
    }
    else{
      if(argscnt == 0){
	iname = sargs[i];
      }
      else if(argscnt == 1){
	lsize = atoi(sargs[i].c_str());
	if(lsize < 0){
	  wrong_args = true;
	  break;
	}
      }    
      argscnt++;    
    }
  }
  
  if(wrong_args || argscnt != 2){
    cout << "Usage:" << endl;
    cout << args[0] << " [option] input number" << endl;
    //cout << "-a :cut alternately." << endl;
    exit(0);
  }
  
  Cut_line *c = new Cut_line();
}

