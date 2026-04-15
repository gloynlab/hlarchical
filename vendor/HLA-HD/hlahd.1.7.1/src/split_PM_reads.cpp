#include <iostream>
#include <fstream>
#include <iomanip>
using namespace std;

#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

string ename[2],iname[2],sname,idname;
string RDIR="./";
string ODIR="./";
string FDIR="";

#include "Tools.h"
#include "Define.h"

int MSIZE = 100;
int NSIZE = 100;
int MINMATCH = 50;

class Split_PM_Reads{
  
public:  
  Split_PM_Reads(){
    int i,j,k,n,n2;
    string word,word2,sline;
    ifstream in;

    in.open(sname.c_str(),ios::in);
    if(!in){
      cout << "Error:couldn't open split file." << endl;
      exit(0);
    }
    int gcnt = 0;
    while(getline(in,sline)){
      gcnt++;
    }
    in.close();
    
    string *gname;
    int *excnt;
    bool **fuse;
     
    gname = new string[gcnt];
    excnt = new int[gcnt];
    fuse = new bool*[gcnt];

    in.open(sname.c_str(),ios::in);
    int hitcnt;   
    for(i=0;i<gcnt;i++){
      getline(in,sline);
      n = 0;
      word = GetWordtab(n,sline);
      gname[i] = word;
      word = GetWordtab(n,sline);
      excnt[i] = atoi(word.c_str());      
      fuse[i] = new bool[excnt[i]];
      word = GetWordtab(n,sline);
      if(word.size() != excnt[i]){
	cout << "Error:exon size and use parameter is not match." << endl;
	exit(0);
      }
      hitcnt = 0;
      for(j=0;j<excnt[i];j++){
	if(word[j] == '0')
	  fuse[i][j] = false;
	else if(word[j] == '1'){
	  fuse[i][j] = true;
	  hitcnt++;
	}
	else{
	  cout << "Error:wrong character " << word[j] << endl;
	  exit(0);
	}	
      }
      if(hitcnt == 0){
	cout << "Error:use parameter of at least one exon must be one." << endl;
	exit(0);
      }      
    }
    in.close();
   
    string cmd;
    string outdir = ODIR + "/" + idname + "/";
    cmd = "if [ -e " + outdir + " ]; then : ; else mkdir " + outdir + " ; fi"; 
    system(cmd.c_str());
    cmd = "if [ -e " + outdir + "exon/ ]; then : ; else mkdir " + outdir + "exon/ ; fi"; 
    system(cmd.c_str());
    cmd = "if [ -e " + outdir + "intron/ ]; then : ; else mkdir " + outdir + "intron/ ; fi";
    system(cmd.c_str());
    cmd = "if [ -e " + outdir + "maplist/ ]; then : ; else mkdir " + outdir + "maplist/ ; fi"; 
    system(cmd.c_str());
    cmd = "if [ -e " + outdir + "result/ ]; then : ; else mkdir " + outdir + "result/ ; fi"; 
    system(cmd.c_str());
    cmd = "if [ -e " + outdir + "log/ ]; then : ; else mkdir " + outdir + "log/ ; fi"; 
    system(cmd.c_str());
    
    ofstream **oute[2],**outi[2];
    string outname;
    for(i=0;i<2;i++){
      oute[i] = new ofstream*[gcnt];
      outi[i] = new ofstream*[gcnt];
      for(j=0;j<gcnt;j++){
	oute[i][j] = new ofstream[excnt[j]];
      }
      for(j=0;j<gcnt;j++){
	outi[i][j] = new ofstream[excnt[j]-1];
      }
    }
    
    for(i=0;i<gcnt;i++){
      for(j=0;j<excnt[i];j++){
	for(k=0;k<2;k++){
	  outname = outdir + "exon/" + idname + "_R" + deci_to_st(k+1)
	    + "." + gname[i] + "_exon" + deci_to_st(j+1) + ".sam";
	  oute[k][i][j].open(outname.c_str(),ios::out);
	}
      }      
      for(j=0;j<excnt[i]-1;j++){
	for(k=0;k<2;k++){
	  outname = outdir + "intron/" + idname + "_R" + deci_to_st(k+1)
	    + "." + gname[i] + "_intron" + deci_to_st(j+1) + ".sam";
	  outi[k][i][j].open(outname.c_str(),ios::out);
	}
      } 
    }
    
    for(k=0;k<2;k++){
      in.open(ename[k].c_str(),ios::in);
      while(getline(in,sline)){
	n = 0;
	if(sline[0] == '@'){
	  word = GetWordtab(n,sline);
	  if(word == "@HD"){
	    for(i=0;i<gcnt;i++){
	      for(j=0;j<excnt[i];j++)
		oute[k][i][j] << sline << endl;
	    }
	  }
	  else if(word == "@SQ"){
	    word = GetWordtab(n,sline);
	    n2 = 0;
	    for(i=0;i<2;i++)
	      word2 = GetWordcol(n2,word);
	    for(i=0;i<gcnt;i++){
	      if(word2 == gname[i]){
		word2 = GetWordUbar(n2,word);	
		if(word2.substr(0,4) == "Exon"){
		  j = atoi(word2.substr(4,word2.size()-4).c_str());
		  if(j >= 1 && j <= excnt[i]){
		    oute[k][i][j-1] << sline << endl;
		  }
		}
	      }
	    }
	  }
	  else if(word == "@PG"){
	    for(i=0;i<gcnt;i++){
	      for(j=0;j<excnt[i];j++)
		oute[k][i][j] << sline << endl;
	    }
	  }
	}
	else{
	  for(i=0;i<3;i++)
	    word = GetWordtab(n,sline);
	  n2 = 0;
	  word2 = GetWordcol(n2,word);
	  for(i=0;i<gcnt;i++){
	    if(word2 == gname[i]){
	      word2 = GetWordUbar(n2,word);
	      if(word2.substr(0,4) == "Exon"){
		j = atoi(word2.substr(4,word2.size()-4).c_str());
		if(j >= 1 && j <= excnt[i]){
		  oute[k][i][j-1] << sline << endl;
		}
	      }
	    }
	  }
	}
      }     
      in.close();
    }
    
    for(k=0;k<2;k++){
      in.open(iname[k].c_str(),ios::in);
      while(getline(in,sline)){
	n = 0;
	if(sline[0] == '@'){
	  word = GetWordtab(n,sline);
	  if(word == "@HD"){
	    for(i=0;i<gcnt;i++){
	      for(j=0;j<excnt[i]-1;j++)
		outi[k][i][j] << sline << endl;
	    }
	  }
	  else if(word == "@SQ"){
	    word = GetWordtab(n,sline);
	    n2 = 0;
	    for(i=0;i<2;i++)
	      word2 = GetWordcol(n2,word);
	    for(i=0;i<gcnt;i++){
	      if(word2 == gname[i]){
		word2 = GetWordUbar(n2,word);	
		if(word2.substr(0,6) == "Intron"){
		  j = atoi(word2.substr(6,word2.size()-6).c_str());
		  if(j >= 1 && j <= excnt[i]-1){
		    outi[k][i][j-1] << sline << endl;
		  }
		}
		break;
	      }
	    }
	  }
	  else if(word == "@PG"){
	    for(i=0;i<gcnt;i++){
	      for(j=0;j<excnt[i]-1;j++)
		outi[k][i][j] << sline << endl;
	    }
	  }
	}
	else{
	  for(i=0;i<3;i++)
	    word = GetWordtab(n,sline);
	  n2 = 0;
	  word2 = GetWordcol(n2,word);
	  for(i=0;i<gcnt;i++){
	    if(word2 == gname[i]){
	      word2 = GetWordUbar(n2,word);	
	      if(word2.substr(0,6) == "Intron"){
		j = atoi(word2.substr(6,word2.size()-6).c_str());
		if(j >= 1 && j <= excnt[i]-1){
		  outi[k][i][j-1] << sline << endl;
		}
	      }
	    }
	  }	  
	}
      }     
      in.close();
    }
   
    ofstream outall;
    outname = outdir + "maplist/maplist.txt";
    outall.open(outname.c_str(),ios::out);
    outall << "convert\t" << RDIR << "/convert.list" << endl;
    outall << "exon\t1\t" << ename[0] << endl;
    outall << "exon\t2\t" << ename[1] << endl;
    outall << "intron\t1\t" << iname[0] << endl;
    outall << "intron\t2\t" << iname[1] << endl;
    outall.close();

    ofstream *outm;
    outm = new ofstream[gcnt];
    for(i=0;i<gcnt;i++){     
      outname = outdir + "maplist/maplist_" + gname[i] + ".txt";
      outm[i].open(outname.c_str(),ios::out);
    }      
    for(i=0;i<gcnt;i++){
      outm[i] << "convert\t" << RDIR << "/convert.list" << endl;
      for(j=0;j<excnt[i];j++){
	outm[i] << "exon\texon" << j+1 << "\t" << fuse[i][j] << endl;
      }
      for(j=0;j<excnt[i]-1;j++){
	outm[i] << "intron\tintron" << j+1 << "\texon" << j+1 << "\texon" << j+2 << endl;
      }
      for(j=0;j<excnt[i];j++){
	outm[i] << "fasta\texon" << j+1 << "\t" << RDIR << "/" << gname[i] << "/"
		<< gname[i] << "_exon" << j+1 << "_N" << NSIZE << ".fasta" << endl;
      }
      for(j=0;j<excnt[i]-1;j++){
	outm[i] << "fasta\tintron" << j+1 << "\t" << NSIZE << "\t" << NSIZE << "\t" 
		<< RDIR << "/" << gname[i] << "/"
		<< gname[i] << "_intron" << j+1 << "_N" << NSIZE << ".fasta" << endl;
      }
      for(k=0;k<2;k++){
	for(j=0;j<excnt[i];j++){
	  outm[i] << "exon" << j+1 << "\t" << k+1 << "\t" << outdir << "exon/" << idname << "_R" << k+1
		  << "." << gname[i] << "_exon" << j+1 << ".sam" << endl;
	}
      }
      for(k=0;k<2;k++){
	for(j=0;j<excnt[i]-1;j++){
	  outm[i] << "intron" << j+1 << "\t" << k+1 << "\t" << outdir << "intron/" << idname 
		  << "_R" << k+1 << "." << gname[i] << "_intron" << j+1 << ".sam" << endl;
	}
      }
    }

    ofstream outest;
    outname = outdir + "estimation.sh";
    outest.open(outname.c_str(),ios::out);
    for(i=0;i<gcnt;i++){
      outest << "hla_est -L " << MSIZE << " --ml " << MINMATCH << " -m 100 --hth 4.0 ";   
      outest  << gname[i] << " " << outdir << "maplist/maplist_" 
	      << gname[i] << ".txt " << outdir << "maplist/maplist.txt -o " << outdir << "result/" << idname << "_" << gname[i]
	   << ".est.txt --pread " << outdir << "result/" << idname << "_" << gname[i]
	   << ".read.txt > " << outdir << "log/" << idname << "_" << gname[i] << ".log" << endl; 
    }
    outest.close();
         
    outname = outdir + "pickup.sh";
    outest.open(outname.c_str(),ios::out);
    for(i=0;i<gcnt;i++){
      outest << "pick_up_allele " << gname[i] << " ";
      outest << outdir << "result/" << idname << "_" << gname[i]
	     << ".est.txt";
      if(FDIR.size() > 0){
	outest << " -f " << FDIR << "/" << gname[i] << "_count.txt";
      }
      if(i==0)
	outest << " > " << outdir << "result/" << idname << "_final.result.txt" << endl;
      else
	outest << " >> " << outdir << "result/" << idname << "_final.result.txt" << endl;
    }
    outest.close();

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
    if(sargs[i] == "-t"){
      i++;
      if(i==narg-1){
        wrong_args = true;
        break;
      }      
      MSIZE = atoi(sargs[i].c_str());    
      if(MSIZE < 0)
        MSIZE = 0;
    }
    else if(sargs[i] == "-m"){
      i++;
      if(i==narg-1){
        wrong_args = true;
        break;
      }      
      MINMATCH = atoi(sargs[i].c_str());    
      if(MINMATCH < 0)
        MINMATCH = 0;
    }
    else if(sargs[i] == "-f"){
      i++;
      if(i==narg-1){
        wrong_args = true;
        break;
      }      
      FDIR = sargs[i];
    }
    else{
      if(argscnt == 0){
	ename[0] = sargs[i];
      }
      else if(argscnt == 1){
	ename[1] = sargs[i];
      }
      else if(argscnt == 2){
	iname[0] = sargs[i];
      }
      else if(argscnt == 3){
	iname[1] = sargs[i];
      }
      else if(argscnt == 4){
	sname = sargs[i];
      }
      else if(argscnt == 5){
	idname = sargs[i];
      }
      else if(argscnt == 6){
	RDIR = sargs[i];
	if(RDIR[RDIR.size()-1] == '/')
	  RDIR = RDIR.substr(0,RDIR.size()-1);
      }
      else if(argscnt == 7){
	NSIZE = atoi(sargs[i].c_str());
      }
      else if(argscnt == 8){
	ODIR = sargs[i];
	if(ODIR[ODIR.size()-1] == '/')
	  ODIR = ODIR.substr(0,ODIR.size()-1);    
      }
   
      argscnt++;      
    }
  }
  
  if(wrong_args || argscnt != 9){
    cout << "Usage:" << endl;    
    cout << args[0] << " [-t <int> :min_tag_size] [-m <int> :min_match_size] [-f <string> :frequency directory] exonmap_F exonmap_R intronmap_F intronmap_R splitfile idname refdir NSIZE outputdir " << endl;
    exit(0);
  }
  Split_PM_Reads *c = new Split_PM_Reads();
}
