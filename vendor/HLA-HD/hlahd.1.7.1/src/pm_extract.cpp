/*******************************************************************************
 *  pm_extract, written by Shuji Kawaguchi, ph.D.                              *
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
#include <time.h>

int hsize = 6;
int MPTH = 0;
int NMTH = 0;
double OC=1.0;
string mname;
#include "Tools.h"
#include "Define.h"

class Extract_Forward_PM_Reads{

public:
  
  Extract_Forward_PM_Reads(){
    int i,ii,j,k,k2,l,s,n,n2,ts;
    int fr;  
    int XN,XM,XO,XG;
    bool N0,NE,NS;  
    ifstream in;
    string idword,word,word2,word3,word4,words[10],sline,reads;
    int hcnt;
    char sw;
    int hid,mid,mid2,hitcnt;
    bool hit;
    int D,I;
    int ns,ne;
    int start;
    string MD;

    hcnt = 1;
    for(i=0;i<hsize;i++){
      hcnt *= 10;
    } 

    int *ls = new int[hsize];
    ls[0] = 1;
    for(i=1;i<hsize;i++)
      ls[i] = 10*ls[i-1];

    in.open(mname.c_str(),ios::in);
    if(!in){
      cout << "Couldn't open sam file." << endl;
      exit(0);
    } 
    
    FASTQ_HASH **sh,*stmp;
    sh = new FASTQ_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      sh[i] = new FASTQ_HASH();
      sh[i]->next = NULL;
    }   

    int misssum,contmis,nowi;
    long int lcnt = 0;
    while(getline(in,sline)){
      n = 0;
      idword = GetWordtab(n,sline);
      if(idword[0] == '@'){
	continue;	    
      }

      lcnt++;
  
      hid = 0;
      j = idword.size();
      k = 0;
      i = 0;
      while(1){
	if(j-i-1 < 0)
	  break;
	sw = idword[j-i-1];
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
	
      stmp = sh[hid];
    
      word2 = GetWordtab(n,sline);  
      word4 = word2;
  
      start = atoi(GetWordtab(n,sline).c_str());
      GetWordtab(n,sline);
      word3 = GetWordtab(n,sline);
      if(word3 == "*")
	continue;
	 
      for(i=0;i<3;i++)
	GetWordtab(n,sline);
      reads = GetWordtab(n,sline);

      hit = false;
      NS = false;
      NE = false;
      XN=0;
      
      do{
	word = GetWordtab(n,sline);
	if(word.substr(0,2) == "XN"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  XN = atoi(GetWordtab(n2,word).c_str());
	}	  
	else if(word.substr(0,2) == "NM"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  hitcnt = atoi(GetWordtab(n2,word).c_str());
	}
	else if(word.substr(0,2) == "MD"){
	  if(word[5] == '0' && word[6] == 'N')
	    NS = true;
	  if(word[word.size()-1] == '0' && word[word.size()-2] == 'N')
	    NE = true;
	  MD = word.substr(5,word.size()-5);	  
	}
      }while(n < sline.size());
       
      hit = false;
      while(stmp->next){
	stmp = stmp->next;
	if(stmp->id == idword){
	  hit = true;
	  break;
	}
      }

      if(!hit){
	stmp->next = new FASTQ_HASH();
	stmp = stmp->next;
	stmp->id = idword;
	stmp->max[0] = 0;
	stmp->next = NULL;
	stmp->minhit[0] = NMTH;
      }

      if(hitcnt - XN < stmp->minhit[0]){
	stmp->minhit[0] = hitcnt - XN;
      }
      
      ts = reads.size();
      if(stmp->max[0]>=ts)
	continue;
 
      if(hitcnt == 0 || XN + MPTH >= hitcnt){
	stmp->max[0] = ts;
	continue;
      }

      ns = 0;
      ne = 0;
      if(XN > 0){
	if(NS && NE){	
	  for(i=0;i+1<MD.size();i+=2){
	    if(MD[i] == '0' && MD[i+1] == 'N')
	      ns++;
	    else
	      break;
	  }
	  for(i=MD.size()-1;i-1>=0;i-=2){
	    if(MD[i-1] == 'N' && MD[i] == '0')
	      ne++;
	    else
	      break;
	  }
	}
	else if(NS){
	  ns = XN;
	}
	else if(NE){
	  ne = XN;	  
	}
      }

      I=0;
      D=0;
      n = 0;
      do{
	word2 = GetWordMID(n,word3);
	if(word3[n-1]=='D'){
	  D += atoi(word2.c_str());
	}
	else if(word3[n-1]=='I'){
	  I += atoi(word2.c_str());
	}
      }while(n < word3.size());
      
      
      int *MIDs;
      char *creads[2];
      MIDs = new int[ts+D];
      for(i=0;i<2;i++)
	creads[i] = new char[ts+D];
      
      n = 0;
      i = 0;
      do{
	word2 = GetWordMID(n,word3);
	if(word3[n-1]=='D'){
	  for(j=0;j<atoi(word2.c_str());j++){
	    MIDs[i] = 2;
	    i++;
	  }
	}
	else if(word3[n-1]=='I'){
	  for(j=0;j<atoi(word2.c_str());j++){
	    MIDs[i] = 1;
	    i++;
	  }
	}
	else{
	  for(j=0;j<atoi(word2.c_str());j++){
	    MIDs[i] = 0;
	    i++;
	  }
	}
      }while(n < word3.size());

      i=0;
      for(j=0;j<ts;j++){	
	while(MIDs[i]==2){
	  creads[0][i]='D';
	  i++;
	  if(i==ts+D)
	    break;
	}
	creads[0][i]=reads[j];
	i++;
      }
      
      
      i=0;
      for(j=0;j<ns;j++){
	while(MIDs[i]==1){
	  creads[1][i]='I';
	  i++;	  
	}
	creads[1][i] = 'N';
	i++;
      }

      for(j=2*ns;j<MD.size()-2*ne;j++){
	if(MD[j] > '0' && MD[j] <= '9'){
	  k=j;
	  while(j+1 < MD.size()-2*ne){
	    if(MD[j+1] < '0' || MD[j+1] > '9')
	      break;
	    j++;
	  }
	  s=atoi(MD.substr(k,j-k+1).c_str());
	  for(k=0;k<s;k++){
	    while(MIDs[i]==1){
	      creads[1][i]='I';
	      i++;	  
	    }
	    creads[1][i]=creads[0][i];
	    i++;
	  }
	}
	else if(MD[j] == '0'){
	  continue;
	}
	else if(MD[j] == '^'){
	  creads[1][i]=MD[j+1];
	  j++;
	  i++;
	}
	else{
	  while(MIDs[i]==1){
	    creads[1][i]='I';
	    i++;	  
	  }
	  creads[1][i]=MD[j];
	  i++;
	}
      }
      for(j=0;j<ne;j++){
	while(MIDs[i]==1){
	  creads[1][i]='I';
	  i++;	  
	}
	creads[1][i] = 'N';
	i++;
      }      
        
      misssum = 0;
      contmis = 0;
      hitcnt =0;
      nowi = 0;
      for(i=0;i<ts+D;i++){
	if(creads[1][i] == 'N'){
	  nowi=i;
	  contmis = 0;
	  continue;
	}
	else if(creads[0][i] != creads[1][i]){
	  misssum++;
	  if(misssum == NMTH+1){
	    misssum--;
	    break;
	  }
	  if(misssum >= MPTH+1 && (double)(hitcnt+misssum)/(ts-(ns-ne)) > OC){
	    misssum--;
	    break;
	  }
	  contmis++;
	}
	else{
	  hitcnt++;
	  contmis = 0;
	  nowi=i;	  
	}
      }

      if((double)(hitcnt+misssum)/(ts-(ns-ne)) >= OC){
	if(misssum-contmis < stmp->minhit[0]){
	  stmp->minhit[0] = misssum-contmis;
	  stmp->max[0] = hitcnt+misssum;
	}
	else if(misssum-contmis == stmp->minhit[0]){
	  if(hitcnt > stmp->max[0]){
	    stmp->max[0] = hitcnt+misssum;
	  }
	}
      }

      misssum = 0;
      contmis = 0;
      hitcnt =0;
      nowi = 0;
      for(i=ts+D-1;i>=0;i--){
	if(creads[1][i] == 'N'){
	  nowi=i;
	  contmis = 0;
	  continue;
	}
	else if(creads[0][i] != creads[1][i]){
	  misssum++;
	  if(misssum == NMTH+1){
	    misssum--;
	    break;
	  }
	  if(misssum >= MPTH+1 && (double)(hitcnt+misssum)/(ts-(ns-ne)) > OC){
	    misssum--;
	    break;
	  }
	  contmis++;
	}
	else{
	  hitcnt++;
	  nowi=i;
	  contmis = 0;
	}
      }

      if((double)(hitcnt+misssum)/(ts-(ns-ne)) >= OC){
	if(misssum-contmis < stmp->minhit[0]){
	  stmp->minhit[0] = misssum-contmis;
	  stmp->max[0] = hitcnt + misssum;
	}
	else if(misssum-contmis == stmp->minhit[0]){
	  if(hitcnt > stmp->max[0]){
	    stmp->max[0] = hitcnt + misssum;
	  }
	}
      }      

      delete[] MIDs;
      for(i=0;i<2;i++)
	delete[] creads[i];
      
    }                    
    in.close(); 
      

    bool plot = false;
    int pstart;
    string pword,pword2;
    int pD,pI;
    int kI,kD;
   
    in.open(mname.c_str(),ios::in);    
    while(getline(in,sline)){
      n = 0;
      idword = GetWordtab(n,sline);
      if(idword[0] == '@'){
	cout << sline << endl;
	continue;	    
      }

      hid = 0;
      j = idword.size();
      k = 0;
      i = 0;
      while(1){
	if(j-i-1 < 0)
	  break;
	sw = idword[j-i-1];
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
	
      stmp = sh[hid];
    
      word2 = GetWordtab(n,sline); 
      word4 = word2;
      start = atoi(GetWordtab(n,sline).c_str());
      GetWordtab(n,sline);
      word3 = GetWordtab(n,sline);
      if(word3 == "*")
	continue;
	 
      for(i=0;i<3;i++)
	GetWordtab(n,sline);
      reads = GetWordtab(n,sline);

      hit = false;
      NS = false;
      NE = false;
      XN=0;
      
      do{
	word = GetWordtab(n,sline);
	if(word.substr(0,2) == "XN"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  XN = atoi(GetWordtab(n2,word).c_str());
	}	  
	else if(word.substr(0,2) == "NM"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  hitcnt = atoi(GetWordtab(n2,word).c_str());
	}
	else if(word.substr(0,2) == "MD"){
	  if(word[5] == '0' && word[6] == 'N')
	    NS = true;
	  if(word[word.size()-1] == '0' && word[word.size()-2] == 'N')
	    NE = true;
	  MD = word.substr(5,word.size()-5);	  
	}
      }while(n < sline.size());
       
      hit = false;
      while(stmp->next){
	stmp = stmp->next;
	if(stmp->id == idword){
	  hit = true;
	  break;
	}
      }
      
      if(!hit){
	stmp->next = new FASTQ_HASH();
	stmp = stmp->next;
	stmp->id = idword;
	stmp->max[0] = 0;
	stmp->next = NULL;
      }
      
      if(stmp->max[0]==0)
	continue;

      ts = reads.size();
      if(hitcnt-XN <= MPTH){
	cout << sline << endl;
	continue;
      }
      
      //if(stmp->max[0]==ts)
      //continue;

      ns = 0;
      ne = 0;
      if(XN > 0){
	if(NS && NE){	
	  for(i=0;i+1<MD.size();i+=2){
	    if(MD[i] == '0' && MD[i+1] == 'N')
	      ns++;
	    else
	      break;
	  }
	  for(i=MD.size()-1;i-1>=0;i-=2){
	    if(MD[i-1] == 'N' && MD[i] == '0')
	      ne++;
	    else
	      break;
	  }
	}
	else if(NS){
	  ns = XN;
	}
	else if(NE){
	  ne = XN;	  
	}
      }

      I=0;
      D=0;
      n = 0;
      do{
	word2 = GetWordMID(n,word3);
	if(word3[n-1]=='D'){
	  D += atoi(word2.c_str());
	}
	else if(word3[n-1]=='I'){
	  I += atoi(word2.c_str());
	}
      }while(n < word3.size());
      
      int *MIDs;
      char *creads[2];
      MIDs = new int[ts+D];
      for(i=0;i<2;i++)
	creads[i] = new char[ts+D];
      
      //cout << sline << endl;
      n = 0;
      i = 0;
      do{
	word2 = GetWordMID(n,word3);
	if(word3[n-1]=='D'){
	  for(j=0;j<atoi(word2.c_str());j++){
	    MIDs[i] = 2;
	    i++;
	  }
	}
	else if(word3[n-1]=='I'){
	  for(j=0;j<atoi(word2.c_str());j++){
	    MIDs[i] = 1;
	    i++;
	  }
	}
	else{
	  for(j=0;j<atoi(word2.c_str());j++){
	    MIDs[i] = 0;
	    i++;
	  }
	}
      }while(n < word3.size());

      i=0;
      for(j=0;j<ts;j++){	
	while(MIDs[i]==2){
	  creads[0][i]='D';
	  i++;
	  if(i==ts+D)
	    break;
	}
	creads[0][i]=reads[j];
	i++;
      }
      
      i=0;
      for(j=0;j<ns;j++){
	while(MIDs[i]==1){
	  creads[1][i]='I';
	  i++;	  
	}
	creads[1][i] = 'N';
	i++;
      }
      for(j=2*ns;j<MD.size()-2*ne;j++){
	if(MD[j] > '0' && MD[j] <= '9'){
	  k=j;
	  while(j+1 < MD.size()-2*ne){
	    if(MD[j+1] < '0' || MD[j+1] > '9')
	      break;
	    j++;
	  }
	  s=atoi(MD.substr(k,j-k+1).c_str());
	  for(k=0;k<s;k++){
	    while(MIDs[i]==1){
	      creads[1][i]='I';
	      i++;	  
	    }
	    creads[1][i]=creads[0][i];
	    i++;
	  }
	}
	else if(MD[j] == '0'){
	  continue;
	}
	else if(MD[j] == '^'){
	  creads[1][i]=MD[j+1];
	  j++;
	  i++;
	}
	else{
	  while(MIDs[i]==1){
	    creads[1][i]='I';
	    i++;	  
	  }
	  creads[1][i]=MD[j];
	  i++;
	}

      }
      for(j=0;j<ne;j++){
	while(MIDs[i]==1){
	  creads[1][i]='I';
	  i++;	  
	}
	creads[1][i] = 'N';
	i++;
      }
       
      misssum = 0;
      contmis = 0;
      hitcnt =0;
      nowi = 0;
      for(i=0;i<ts+D;i++){
	if(creads[1][i] == 'N'){
	  nowi=i;
	  contmis=0;
	  continue;
	}
	else if(creads[0][i] != creads[1][i]){
	  misssum++;
	  if(misssum == NMTH+1){
	    misssum--;
	    break;
	  }
	  if(misssum >= MPTH+1 && (double)(hitcnt+misssum)/(ts-(ns-ne)) > OC){
	    misssum--;
	    break;
	  }
	  contmis++;
	}
	else{
	  hitcnt++;
	  contmis=0;
	  nowi=i;	  
	}
      }

      plot = false;
      if((double)(hitcnt+misssum)/(ts-(ns-ne)) >= OC){
	if(hitcnt+misssum == stmp->max[0] && misssum-contmis == stmp->minhit[0]){
	  NS = true;
	  pstart = start;
	  pD = 0; pI = 0;
	  XN = 0;
	  XO = 0;
	  XG = 0;
	  misssum = 0;
	  pword = ""; pword2 = "";
	  k = 0; k2 = 0; kI = 0; kD = 0;
	  for(i=0;i<=nowi;i++){
	    if(creads[1][i] == 'N'){
	      if(k2 > 0){
		pword2 += deci_to_st(k2);
		k2 = 0;
	      }	     
	      if(NS) 
		pword2 += "0N";
	      else
		pword2 += "N0";
	      XN++;
	      k++;
	    }
	    else{
	      if(NS)
		NS = false;
	      if(creads[0][i] != creads[1][i]){
		if(creads[0][i] == 'D'){
		  if(k > 0){
		    pword += deci_to_st(k) + "M";
		    k = 0;
		  }
		  else if(kI > 0){
		    pword += deci_to_st(kI) + "I";
		    XO++;
		    XG += kI;
		    kI = 0;
		  }
		  if(k2 > 0){
		    pword2 += deci_to_st(k2);
		    k2 = 0;
		  }
		  kD++;
		  pword2 += "^";
		  pword2 += creads[1][i];
		  misssum++;
		}
		else if(creads[1][i] == 'I'){
		  if(k > 0){
		    pword += deci_to_st(k) + "M";
		    k = 0;
		  }
		  else if(kD > 0){
		    pword += deci_to_st(kD) + "D";
		    XO++;
		    XG += kD;
		    kD = 0;
		  }
		  kI++;
		  misssum++;
		}
		else{
		  if(kI > 0){
		    pword += deci_to_st(kI) + "I";
		    XO++;
		    XG += kI;
		    kI = 0;
		  }
		  else if(kD > 0){
		    pword += deci_to_st(kD) + "D";
		    XO++;
		    XG += kD;
		    kD = 0;
		  }
		  pword2 += deci_to_st(k2);
		  k2 = 0;
		  pword2 += creads[1][i];
		  k++;
		  misssum++;
		}
	      }
	      else{
		if(kI > 0){
		  pword += deci_to_st(kI) + "I";
		  XO++;
		  XG += kI;
		  kI = 0;
		}
		else if(kD > 0){
		  pword += deci_to_st(kD) + "D";
		  XO++;
		  XG += kD;
		  kD = 0;
		}
		k++;
		k2++;
	      }
	    }
	  }
	  if(k > 0){
	    pword += deci_to_st(k) + "M";
	  }
	  else if(kI > 0){
	    pword += deci_to_st(kI) + "I";
	    XO++;
	    XG += kI;
	  }
	  else if(kD > 0){
	    pword += deci_to_st(kD) + "D";
	    XO++;
	    XG += kD;
	  }

	  if(k2 > 0){
	    pword2 += deci_to_st(k2);	    
	  }
	  plot = true;

	  //cout << "F:" << endl;
	  //cout << sline << endl << endl;
	  n = 0;
	  for(i=0;i<3;i++){
	    if(i==0)
	      cout << GetWordtab(n,sline);
	    else
	      cout << "\t" <<  GetWordtab(n,sline);
	  }	    
	  GetWordtab(n,sline);
	  cout << "\t" << pstart;
	  cout << "\t" << GetWordtab(n,sline);
	  cout << "\t" << pword;
	  GetWordtab(n,sline);
	  for(i=0;i<3;i++){	  
	    cout << "\t" <<  GetWordtab(n,sline);
	  }
	  cout << "\t";
	  for(i=0;i<=nowi;i++){
	    if(creads[0][i] != 'D'){
	      cout << creads[0][i];
	    }
	  }
	  GetWordtab(n,sline);
	  word = GetWordtab(n,sline);
	  cout << "\t";
	  j = 0;
	  for(i=0;i<=nowi;i++){
	    if(creads[0][i] == 'D'){
	      continue;
	    }
	    cout << word[j];
	    j++;
	  }
	  do{
	    word = GetWordtab(n,sline);
	    cout << "\t";
	    if(word[0] == 'X' && word[1] == 'N'){
	      cout << "XN:i:" << XN;
	    }
	    else if(word[0] == 'N' && word[1] == 'M'){
	      cout << "NM:i:" << XN+misssum;
	    }
	    else if(word[0] == 'X' && word[1] == 'O'){
	      cout << "XO:i:" << XO;
	    }
	    else if(word[0] == 'X' && word[1] == 'M'){
	      cout << "XM:i:" << XN+misssum-XG;
	    }
	    else if(word[0] == 'X' && word[1] == 'G'){
	      cout << "XM:i:" << XG;
	    }
	    else if(word[0] == 'M' && word[1] == 'D'){
	      cout << "MD:Z:" << pword2;
	    }
	    else{
	      cout << word;
	    }
	  }while(n < sline.size());
	  cout << endl;
	}
      }
              
      if(!plot){
	misssum = 0;
	contmis = 0;
	hitcnt =0;
	nowi = ts+D-1;
	for(i=ts+D-1;i>=0;i--){
	  if(creads[1][i] == 'N'){
	    nowi=i;
	    contmis = 0;
	    continue;
	  }
	  else if(creads[0][i] != creads[1][i]){
	    misssum++;	
	    if(misssum == NMTH+1){
	      misssum--;
	      break;
	    }
	    if(misssum >= MPTH+1 && (double)(hitcnt+misssum)/(ts-(ns-ne)) > OC){
	      misssum--;
	      break;
	    }
	    contmis++;
	  }
	  else{
	    hitcnt++;
	    nowi=i;
	    contmis=0;
	  }
	}

	if((double)(hitcnt+misssum)/(ts-(ns-ne)) >= OC){
	  if(hitcnt+misssum == stmp->max[0] && misssum-contmis == stmp->minhit[0]){
	    NS = true;
	    kD = 0; kI = 0; //20200707 bug fix
	    for(i=0;i<nowi;i++){
	      if(creads[1][i] == 'I')
		kI++;
	    }
	    pstart = start+nowi-kI;
	    pD = 0; pI = 0;
	    XN = 0;
	    XO = 0;
	    XG = 0;
	    misssum = 0;
	    pword = ""; pword2 = "";
	    k = 0; k2 = 0; kI = 0; kD = 0;
	    for(i=nowi;i<ts+D;i++){
	      if(creads[1][i] == 'N'){
		if(k2 > 0){
		  pword2 += deci_to_st(k2);
		  k2 = 0;
		}	     
		if(NS) 
		  pword2 += "0N";
		else
		  pword2 += "N0";
		XN++;
		k++;
	      }
	      else{
		if(NS)
		  NS = false;
		if(creads[0][i] != creads[1][i]){
		  if(creads[0][i] == 'D'){
		    if(k > 0){
		      pword += deci_to_st(k) + "M";
		      k = 0;
		    }
		    else if(kI > 0){
		      pword += deci_to_st(kI) + "I";
		      XO++;
		      XG += kI;
		      kI = 0;
		    }
		    if(k2 > 0){
		      pword2 += deci_to_st(k2);
		      k2 = 0;
		    }
		    kD++;
		    pword2 += "^";
		    pword2 += creads[1][i];
		    misssum++;
		  }
		  else if(creads[1][i] == 'I'){
		    if(k > 0){
		      pword += deci_to_st(k) + "M";
		      k = 0;
		    }
		    else if(kD > 0){
		      pword += deci_to_st(kD) + "D";
		      XO++;
		      XG += kD;
		      kD = 0;
		    }
		    kI++;
		    misssum++;
		  }
		  else{
		    if(kI > 0){
		      pword += deci_to_st(kI) + "I";
		      XO++;
		      XG += kI;
		      kI = 0;
		    }
		    else if(kD > 0){
		      pword += deci_to_st(kD) + "D";
		      XO++;
		      XG += kD;
		      kD = 0;
		    }		    
		    pword2 += deci_to_st(k2);
		    k2 = 0;
		    pword2 += creads[1][i];
		    k++;
		    misssum++;
		  }
		}
		else{
		  if(kI > 0){
		    pword += deci_to_st(kI) + "I";
		    XO++;
		    XG += kI;
		    kI = 0;
		  }
		  else if(kD > 0){
		    pword += deci_to_st(kD) + "D";
		    XO++;
		    XG += kD;
		    kD = 0;
		  }
		  k++;
		  k2++;
		}
	      }
	    }
	    if(k > 0){
	      pword += deci_to_st(k) + "M";
	    }
	    else if(kI > 0){
	      pword += deci_to_st(kI) + "I";
	      XO++;
	      XG += kI;
	    }
	    else if(kD > 0){
	      pword += deci_to_st(kD) + "D";
	      XO++;
	      XG += kD;
	    }

	    if(k2 > 0){
	      pword2 += deci_to_st(k2);	    
	    }
	    plot = true;

	    //cout << "R:" << endl;
	    //cout << sline << endl << endl;
	    n = 0;
	    for(i=0;i<3;i++){
	      if(i==0)
		cout << GetWordtab(n,sline);
	      else
		cout << "\t" <<  GetWordtab(n,sline);
	    }	   
	    GetWordtab(n,sline);
	    cout << "\t" << pstart;
	    cout << "\t" << GetWordtab(n,sline);
	    cout << "\t" << pword;
	    GetWordtab(n,sline);

	    for(i=0;i<3;i++){	  
	      cout << "\t" <<  GetWordtab(n,sline);
	    }
	    cout << "\t";
	    for(i=nowi;i<ts+D;i++){
	      if(creads[0][i] != 'D'){
		cout << creads[0][i];
	      }
	    }
	    GetWordtab(n,sline);
	    word = GetWordtab(n,sline);
	    cout << "\t";
	    j = 0;
	    for(i=0;i<nowi;i++){
	      if(creads[0][i] == 'D'){
		continue;
	      }
	      j++;
	    }
	    for(i=nowi;i<ts+D;i++){
	      if(creads[0][i] == 'D'){
		continue;
	      }
	      cout << word[j];
	      j++;
	    }

	    do{
	      word = GetWordtab(n,sline);
	      cout << "\t";
	      if(word[0] == 'X' && word[1] == 'N'){
		cout << "XN:i:" << XN;
	      }
	      else if(word[0] == 'N' && word[1] == 'M'){
		cout << "NM:i:" << XN+misssum;
	      }
	      else if(word[0] == 'X' && word[1] == 'O'){
		cout << "XO:i:" << XO;
	      }
	      else if(word[0] == 'X' && word[1] == 'M'){
		cout << "XM:i:" << XN+misssum-XG;
	      }
	      else if(word[0] == 'X' && word[1] == 'G'){
		cout << "XM:i:" << XG;
	      }
	      else if(word[0] == 'M' && word[1] == 'D'){
		cout << "MD:Z:" << pword2;
	      }
	      else{
		cout << word;
	      }
	    }while(n < sline.size());
	    cout << endl;
	  }
	}      
      }
      
      delete[] MIDs;
      for(i=0;i<2;i++)
	delete[] creads[i];
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
    if(sargs[i] == "--MP"){
      i++;
      if(i==narg-1){
        wrong_args = true;
        break;
      }
      MPTH = atoi(sargs[i].c_str());
    }
    else if(sargs[i] == "--NM"){
      i++;
      if(i==narg-1){
        wrong_args = true;
        break;
      }
      NMTH = atoi(sargs[i].c_str());
    }
    else{
      if(argscnt == 0){
	mname = sargs[i];
      }
      else if(argscnt == 1){
	OC = atof(sargs[i].c_str());
	if(OC <= 0 || OC > 1.0){
	  wrong_args = false;	  
	}
      }
      argscnt++;    
    }
  }
  
  if(wrong_args || argscnt != 2){
    cout << "Usage:" << endl;
    cout << args[0] << "[options] mapname occupation(0,1]" << endl;
    cout << "--MP <int> [ Minimum using threshold default: 0 ] " << endl; 
    cout << "--NM <int> [ Mismatch threshold default: 0 ] " << endl; 
    exit(0);
  }

  Extract_Forward_PM_Reads *c = new Extract_Forward_PM_Reads();
}
