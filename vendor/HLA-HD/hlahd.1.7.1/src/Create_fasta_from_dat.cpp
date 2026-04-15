/*******************************************************************************
 *  Create_fasta_from_dat, written by Shuji Kawaguchi, ph.D.                   *
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

#include "CFASTA_Tools.h"
#include "CFASTA_define.h"

const int MAX_EXON = 1024;
int N_BASE = 30;

bool useother = false;
string otname;

class Create_FASTA_F_DAT{

public:
  
  Create_FASTA_F_DAT(char* dname, char* oname, char* Nc){
    int i,j,k,k2,k3,k4,l,n,n2;    
    ifstream in(dname,ios::in),in2;    
    string word,word2,words[10],sline;
    PROPERTY *ptop,*ptmp;
    int exsum;
    ptop = new PROPERTY();
    ptop->next = NULL;
    ptmp = ptop;

    N_BASE = atoi(Nc);
    if(N_BASE < 0)
      N_BASE = 0;

    bool exon,gene;
    if(in){
      while(getline(in,sline)){
	n = 0;
	word = GetWordsp(n,sline);
	if(word == "ID"){
	  word = GetWordsp(n,sline);
	  if(word[word.size()-1] == ';')
	    word = word.substr(0,word.size()-1);
	  cout << word << " ";	  
	  ptmp->next = new PROPERTY();
	  ptmp = ptmp->next;
	  ptmp->es = new int[MAX_EXON];
	  ptmp->ee = new int[MAX_EXON];
	  ptmp->eid = new int[MAX_EXON];
	  ptmp->name = word;
	  for(i=0;i<MAX_EXON;i++){
	    ptmp->es[i] = -1;
	    ptmp->ee[i] = -1;
	  }
	  ptmp->excnt = 0;
	  ptmp->next = NULL;	 
	  ptmp->par = false;
	  ptmp->pse = false;
	  ptmp->unuse = false;
	  ptmp->eg = false;
	  ptmp->glc = NULL;
	  exsum = 0;
	  gene = false;
	  exon = false;
	  while(getline(in,sline)){
	    n = 0;
	    word = GetWordsp(n,sline);	   
	    if(word == "//"){
	      ptmp->pse = true;
	      break;
	    }
	    if(word == "DE"){
	      ptmp->al = GetWordcam(n,sline);
	      cout << ptmp->al << " ";
	    }
	    else if(word == "FT"){
	      GetWordsp(n,sline);
	      for(i=0;i<2;i++)
		GetWorddot(n,sline);
	      ptmp->length = atoi(GetWordsp(n,sline).c_str());
	      cout << ptmp->length << " ";
	      while(getline(in,sline)){
		n = 0;
		word = GetWordsp(n,sline);
		if(word == "SQ")
		  break;
		word = GetWordsp(n,sline);
		if(word == "exon"){
		  if(ptmp->excnt == MAX_EXON){
		    cout << "Number of exons overs " << MAX_EXON << "." << endl;
		    exit(-1);
		  }
		  ptmp->es[ptmp->excnt] = atoi(GetWorddot(n,sline).c_str());
		  GetWorddot(n,sline).c_str();
		  ptmp->ee[ptmp->excnt] = atoi(GetWordsp(n,sline).c_str());
		  if(ptmp->excnt == 0)
		    cout << ptmp->es[ptmp->excnt] << ".." << ptmp->ee[ptmp->excnt];
		  else
		    cout << "," << ptmp->es[ptmp->excnt] << ".." << ptmp->ee[ptmp->excnt];
		  getline(in,sline);
		  n = 0;
		  GetWorddq(n,sline);
		  ptmp->eid[ptmp->excnt] = atoi(GetWorddq(n,sline).c_str())-1;
		  cout << ":" << ptmp->eid[ptmp->excnt];
		  exsum += ptmp->ee[ptmp->excnt] - ptmp->es[ptmp->excnt] + 1;
		  ptmp->excnt++;
		  exon = true;
		  gene = false;
		}
		else if(word == "/pseudo"){
		  if(gene){
		    ptmp->pse = true;
		    cout << "Pseudo ";
		  }
		}
		else if(word == "/partial"){
		  ptmp->par = true;
		  cout << "Partial ";
		}
		else if(word.substr(0,7)=="/allele"){
		  n = 0;
		  n2 = 0;
		  GetWordbar(n,word);
		  word2 = GetWordast(n2,word);
		  if(n2 > n)
		    ptmp->g = GetWordast(n,word);
		  else{
		    n = 0;
		    GetWorddq(n,word);
		    ptmp->g = GetWordast(n,word);
		  }
		  cout << ptmp->g;
		  for(i=0;i<4;i++){
		    ptmp->a[i] = atoi(GetWordcol(n,word).c_str());
		    cout << " " << ptmp->a[i];
		  }
		  cout << endl;
		}
		else if(word == "gene"){
		  gene = true;
		  exon = false;
		}
	      }
	    	  
	      cout << endl;
	      k = 0;
	      ptmp->seq = new char[ptmp->length];
	      if(exsum < ptmp->length)
		ptmp->eg = true;
	      while(getline(in,sline)){
		n = 0;	    
		word = GetWordsp(n,sline);
		if(word == "//")
		  break;
		for(i=0;i<6;i++){		  
		  word = GetWordsp(n,sline);
		  if(atoi(word.c_str()) > 0)
		    break;
		  for(j=0;j<word.size();j++){
		    ptmp->seq[k] = word[j];
		    k++;
		  }
		}
	      }
	      if(ptmp->ee[ptmp->excnt-1] > ptmp->length){
		cout << ptmp->g << " : structure and sequence sizes are not matched." << endl;
		ptmp->unuse = true;
	      }
	    }
	    if(word == "//")
	      break;
	  }	    
	  cout << endl;
	}
      }
      in.close();
    }
    else{
      cout << "Couldn't open dat file." << endl;
      exit(-1);
    }

    //The maximum number of CDS exons for classical HLA should be constant.
    ptmp = ptop;
    while(ptmp->next){
      ptmp = ptmp->next;
      for(i=0;i<CL_HLA_NUM;i++){
	if(ptmp->g == CL_HLA[i]){
	  if(ptmp->excnt > CL_HLA_EX[i]){
	    ptmp->excnt = CL_HLA_EX[i];
	  }
	}
      }
    }
    
    GENELC *gtop,*gtmp;
    gtop = new GENELC();
    gtop->next = NULL;
    gtop->g = "";
    gtmp = gtop;
    ptmp = ptop;        
    while(ptmp->next){
      ptmp = ptmp->next;
      if(ptmp->unuse)
	continue;
      if(ptmp->pse)
	continue;
      gtmp = gtop;
      while(gtmp->next){
	gtmp = gtmp->next;
	if(gtmp->g == ptmp->g)
	  break;
      }
      if(gtmp->g != ptmp->g){
	gtmp->next = new GENELC();
	gtmp = gtmp->next;
	gtmp->next = NULL;
	gtmp->g = ptmp->g;
	gtmp->ex = ptmp->eid[ptmp->excnt-1]+1;
	gtmp->cnt = 1;
	if(!ptmp->par){
	  gtmp->npcnt = 1;	  	
	  gtmp->mnlen = 0;
	  for(i=0;i<ptmp->excnt;i++)
	    gtmp->mnlen += ptmp->ee[i]-ptmp->es[i]+1;
	  if(ptmp->eg){
	    gtmp->egcnt = 1;
	    gtmp->mglen = ptmp->length;	    
	  }
	}
	gtmp->melen = new int[MAX_EXON];
	for(i=0;i<MAX_EXON;i++)
	  gtmp->melen[i] = 0;
	for(i=0;i<ptmp->excnt;i++)
	  if(ptmp->ee[i]-ptmp->es[i]+1 > gtmp->melen[ptmp->eid[i]]){
	    gtmp->melen[ptmp->eid[i]] = ptmp->ee[i]-ptmp->es[i]+1;
	  }      
	ptmp->glc = gtmp;
      }
      else{
	if(ptmp->eid[ptmp->excnt-1]+1 > gtmp->ex)
	  gtmp->ex = ptmp->eid[ptmp->excnt-1]+1;
	gtmp->cnt++;
	if(!ptmp->par){
	  gtmp->npcnt++;
	  int tmpnlen = 0;
	  for(i=0;i<ptmp->excnt;i++)
	    tmpnlen += ptmp->ee[i]-ptmp->es[i]+1;
	  if(tmpnlen > gtmp->mnlen)
	    gtmp->mnlen = tmpnlen;
	  if(ptmp->eg){
	    gtmp->egcnt++;
	    int tmpglen = 0;
	    if(ptmp->length > gtmp->mglen)
	      gtmp->mglen = ptmp->length;
	  }
	}
	for(i=0;i<ptmp->excnt;i++)
	  if(ptmp->ee[i]-ptmp->es[i]+1 > gtmp->melen[ptmp->eid[i]]){
	    gtmp->melen[ptmp->eid[i]] = ptmp->ee[i]-ptmp->es[i]+1;
	  }   
	ptmp->glc = gtmp;
      }
    }
    gtmp = gtop;
    
    while(gtmp->next){ 
      gtmp = gtmp->next;
      cout << gtmp->g << " " << gtmp->ex << " " << gtmp->cnt << " " << gtmp->npcnt
	   << " " << gtmp->egcnt << endl;          
    }
        
    string ODIR(oname);
    ODIR += "/";
    string outname,odir;
    ofstream out,outd,outm,outbw,outl,outa,outallgn,outallgnutr,outallexn,outallintn,outallutr;
    ofstream outeil;
    outname = ODIR + "create_dir.sh";
    outd.open(outname.c_str(),ios::out);
    outname = ODIR + "move_file.sh";
    outm.open(outname.c_str(),ios::out);   
    outname = ODIR + "bw_build.sh";
    outbw.open(outname.c_str(),ios::out);    
    outname = ODIR + "gene.list";
    outl.open(outname.c_str(),ios::out);
    outname = ODIR + "allele.list";
    outa.open(outname.c_str(),ios::out);
    outname = ODIR + "all_gen_N" + deci_to_st(N_BASE) +  ".fasta";
    outallgn.open(outname.c_str(),ios::out);
    //outbw << "bowtie2-build " << "all_gen_N" + deci_to_st(N_BASE) +  ".fasta "
    //	  << "all_gen_N" + deci_to_st(N_BASE) +  ".fasta "
    //	  << endl;
    
    outname = ODIR + "all_gen_w_utr_N" + deci_to_st(N_BASE) +  ".fasta";
    outallgnutr.open(outname.c_str(),ios::out);
    outbw << "bowtie2-build " << "all_gen_w_utr_N" + deci_to_st(N_BASE) +  ".fasta "
    	  << "all_gen_w_utr_N" + deci_to_st(N_BASE) +  ".fasta "
    	  << endl;
    
    outname = ODIR + "all_exon_N" + deci_to_st(N_BASE) +  ".fasta";
    outallexn.open(outname.c_str(),ios::out);
    outbw << "bowtie2-build " << "all_exon_N" + deci_to_st(N_BASE) +  ".fasta "
	  << "all_exon_N" + deci_to_st(N_BASE) +  ".fasta "
	  << endl;
    outname = ODIR + "all_intron_N" + deci_to_st(N_BASE) +  ".fasta";
    outallintn.open(outname.c_str(),ios::out);
    outbw << "bowtie2-build " << "all_intron_N" + deci_to_st(N_BASE) +  ".fasta "
    	  << "all_intron_N" + deci_to_st(N_BASE) +  ".fasta "
    	  << endl;
    outname = ODIR + "all_utr_N" + deci_to_st(N_BASE) +  ".fasta";
    outallutr.open(outname.c_str(),ios::out);
    //outbw << "bowtie2-build " << "all_utr_N" << deci_to_st(N_BASE) << ".fasta "
    //	  << "all_utr_N" << deci_to_st(N_BASE) <<  ".fasta "
    //	  << endl;
    outbw << "cat all_exon_N" << deci_to_st(N_BASE) <<  ".fasta "
	  << "all_intron_N" << deci_to_st(N_BASE) << ".fasta > "
	  << "all_exon_intron_N" << deci_to_st(N_BASE) <<  ".fasta " << endl;
    outbw << "bowtie2-build " << "all_exon_intron_N" << deci_to_st(N_BASE) 
	  <<  ".fasta " << "all_exon_intron_N" << deci_to_st(N_BASE) 
	  <<  ".fasta " << endl;

    outname = ODIR + "convert.list";
    outeil.open(outname.c_str(),ios::out);    
  
    gtmp = gtop;
    while(gtmp->next){
      gtmp = gtmp->next;
      outl << gtmp->g << endl;
      gtmp->odir = gtmp->g + "/";
      outd << "if [ -e " << gtmp->odir << " ]; then" << endl;
      outd << ":" << endl;
      outd << "else" << endl;
      outd << "mkdir " << gtmp->odir <<  endl;
      outd << "fi" << endl;
            
      outname = ODIR + gtmp->g + ".list";
      gtmp->outl.open(outname.c_str(),ios::out);
      gtmp->outl << "#Gene name\t" << gtmp->g << endl;
      gtmp->outl << "#Alleles\t" << gtmp->cnt << endl;
      gtmp->outl << "#Maxgene\t" << gtmp->mglen << endl;
      gtmp->outl << "#Maxnuc\t" << gtmp->mnlen << endl;
      gtmp->outl << "#Exon\t" << gtmp->ex;
      for(i=0;i<gtmp->ex;i++)
	gtmp->outl << "\t" << gtmp->melen[i];
      gtmp->outl << endl;
      gtmp->outl << "#HLA name\tAllele name\tgene\tnuc";
      for(i=0;i<gtmp->ex;i++){
	gtmp->outl << "\texon" << i+1;
      }
      gtmp->outl << endl;      
      outm << "mv " << gtmp->g << ".list " << gtmp->odir << endl;
     
      outname = ODIR + gtmp->g + "_gen_N" + deci_to_st(N_BASE) + ".fasta";
      gtmp->outgn.open(outname.c_str(),ios::out);
      outm << "mv " << gtmp->g << "_gen_N" << N_BASE << ".fasta " << gtmp->odir << endl;
      //outbw << "bowtie2-build " 
      //    << gtmp->odir << gtmp->g << "_gen_N" << N_BASE << ".fasta "
      //    << gtmp->odir << gtmp->g << "_gen_N" << N_BASE << ".fasta " 
      //    << endl;

      outname = ODIR + gtmp->g + "_gen_w_utr_N" + deci_to_st(N_BASE) + ".fasta";
      gtmp->outgnutr.open(outname.c_str(),ios::out);
      outm << "mv " << gtmp->g << "_gen_w_utr_N" << N_BASE << ".fasta " << gtmp->odir << endl;
      //outbw << "bowtie2-build " 
      //    << gtmp->odir << gtmp->g << "_gen_w_utr_N" << N_BASE << ".fasta "
      //    << gtmp->odir << gtmp->g << "_gen_w_utr_N" << N_BASE << ".fasta " 
      //    << endl;
 
      gtmp->outen = new ofstream[gtmp->ex];  
      gtmp->outin = new ofstream[gtmp->ex-1];

      for(i=0;i<gtmp->ex;i++){	
	outname = ODIR + gtmp->g + "_exon" + deci_to_st(i+1) + "_N" + deci_to_st(N_BASE) + ".fasta";
	gtmp->outen[i].open(outname.c_str(),ios::out);
	outm << "mv " << gtmp->g << "_exon" << i+1 << "_N" << N_BASE << ".fasta " 
	     << gtmp->odir << endl;
	//outbw << "bowtie2-build " 
	//    << gtmp->odir << gtmp->g << "_exon" << i+1 << "_N" << N_BASE << ".fasta "
	//    << gtmp->odir << gtmp->g << "_exon" << i+1 << "_N" << N_BASE << ".fasta "
	//    << endl;

	if(i < gtmp->ex-1){
	  outname = ODIR + gtmp->g + "_intron" + deci_to_st(i+1) + "_N" + deci_to_st(N_BASE) + ".fasta";

	  gtmp->outin[i].open(outname.c_str(),ios::out);
	  outm << "mv " << gtmp->g << "_intron" << i+1 << "_N" << N_BASE << ".fasta " 
	       << gtmp->odir << endl;
	  //outbw << "bowtie2-build " 
	  //	<< gtmp->odir << gtmp->g << "_intron" << i+1 << "_N" << N_BASE << ".fasta "
	  //	<< gtmp->odir << gtmp->g << "_intron" << i+1 << "_N" << N_BASE << ".fasta "
	  //	<< endl;
	}
      }
      
      outname = ODIR + gtmp->g + "_utr5.fasta";
      gtmp->outu5.open(outname.c_str(),ios::out);
      outm << "mv " << gtmp->g << "_utr5.fasta " << gtmp->odir << endl;
   
      outname = ODIR + gtmp->g + "_utr3.fasta";
      gtmp->outu3.open(outname.c_str(),ios::out);
      outm << "mv " << gtmp->g << "_utr3.fasta " << gtmp->odir << endl;
    }
    
    gtmp = gtop;
    while(gtmp->next){
      gtmp = gtmp->next;
      gtmp->sameex = new PROPERTY*[gtmp->ex];
      for(i=0;i<gtmp->ex;i++){
	gtmp->sameex[i] = new PROPERTY();
	gtmp->sameex[i]->next = NULL;
      }
      gtmp->samein = new PROPERTY*[gtmp->ex-1];
      for(i=0;i<gtmp->ex-1;i++){
	gtmp->samein[i] = new PROPERTY();
	gtmp->samein[i]->next = NULL;
      }
      gtmp->sameutr = new PROPERTY*[2];
      for(i=0;i<2;i++){
	gtmp->sameutr[i] = new PROPERTY();
	gtmp->sameutr[i]->next = NULL;
      }
    }

    PROPERTY *ptmp2;
    bool hit;
    ptmp = ptop;
    while(ptmp->next){
      ptmp = ptmp->next;
      if(ptmp->unuse)
	continue;

      for(i=0;i<ptmp->excnt;i++){
	hit = false;
	ptmp2 = ptmp->glc->sameex[ptmp->eid[i]];
	while(ptmp2->next){
	  ptmp2 = ptmp2->next;
	  hit = true;
	  if(ptmp2->length != ptmp->ee[i]-ptmp->es[i]+1){
	    hit = false;
	  }
	  else{	    
	    for(j=0;j<ptmp2->length;j++){
	      if(ptmp2->seq[j] != ptmp->seq[j+ptmp->es[i]-1]){
		hit = false;
		break;
	      }	      
	    }
	  }
	  if(hit)
	    break;
	}

	
	if(hit){
	  ptmp2->name += "," + ptmp->name + "." + ptmp->al;
	}
	else{
	  ptmp2->next = new PROPERTY();
	  ptmp2 = ptmp2->next;
	  ptmp2->next = NULL;
	  ptmp2->length = ptmp->ee[i]-ptmp->es[i]+1;
	  ptmp2->seq = new char[ptmp2->length];
	  for(j=0;j<ptmp2->length;j++){
	    ptmp2->seq[j] = ptmp->seq[j+ptmp->es[i]-1];
	  }
	  ptmp2->name = ptmp->name + "." + ptmp->al;
	}
	
	if(i < ptmp->excnt-1){
	  if(ptmp->es[i+1]-ptmp->ee[i] <= 1)
	    continue;
	  hit = false;
	  ptmp2 = ptmp->glc->samein[ptmp->eid[i]];
	  while(ptmp2->next){
	    ptmp2 = ptmp2->next;
	    hit = true;
	    if(ptmp2->length != ptmp->es[i+1]-ptmp->ee[i]-1){
	      hit = false;
	    }
	    else{	    
	      for(j=0;j<ptmp2->length;j++){
		if(ptmp2->seq[j] != ptmp->seq[j+ptmp->ee[i]]){
		  hit = false;
		  break;
		}	      
	      }
	    }
	    if(hit)
	      break;
	  }
	  if(hit){
	    ptmp2->name += "," + ptmp->name + "." + ptmp->al;
	  }
	  else{
	    ptmp2->next = new PROPERTY();
	    ptmp2 = ptmp2->next;
	    ptmp2->next = NULL;
	    ptmp2->length = ptmp->es[i+1]-ptmp->ee[i]-1;
	    ptmp2->seq = new char[ptmp2->length];
	    for(j=0;j<ptmp2->length;j++){
	      ptmp2->seq[j] = ptmp->seq[j+ptmp->ee[i]];
	    }
	    ptmp2->name = ptmp->name + "." + ptmp->al;
	  }
	}
	
      }

      cout << ptmp->name << "\t" << ptmp->al << "\t" << ptmp->es[0] << "\t" << ptmp->ee[ptmp->excnt-1] << "\t" << ptmp->length << endl;
      if(ptmp->es[0] > 1 && ptmp->ee[ptmp->excnt-1] < ptmp->length){
	hit = false;
	ptmp2 = ptmp->glc->sameutr[0];
	while(ptmp2->next){
	  ptmp2 = ptmp2->next;
	  hit = true;
	  if(ptmp2->length != ptmp->es[0]-1){
	    hit = false;
	  }
	  else{	    
	    for(j=0;j<ptmp2->length;j++){
	      if(ptmp2->seq[j] != ptmp->seq[j]){
		hit = false;
		break;
	      }	      
	    }
	  }
	  if(hit)
	    break;
	}
	if(hit){
	  ptmp2->name += "," + ptmp->name + "." + ptmp->al;
	}
	else{
	  ptmp2->next = new PROPERTY();
	  ptmp2 = ptmp2->next;
	  ptmp2->next = NULL;
	  ptmp2->length = ptmp->es[0]-1;
	  ptmp2->seq = new char[ptmp2->length];
	  for(j=0;j<ptmp2->length;j++){
	    ptmp2->seq[j] = ptmp->seq[j];
	  }
	  ptmp2->name = ptmp->name + "." + ptmp->al;
	}
      
	hit = false;
	ptmp2 = ptmp->glc->sameutr[1];
	while(ptmp2->next){
	  ptmp2 = ptmp2->next;
	  hit = true;
	  if(ptmp2->length != ptmp->length-ptmp->ee[ptmp->excnt-1]){
	    hit = false;
	  }
	  else{	    
	    for(j=0;j<ptmp2->length;j++){
	      if(ptmp2->seq[j] != ptmp->seq[ptmp->ee[ptmp->excnt-1]+j]){
		hit = false;
		break;
	      }	      
	    }
	  }
	  if(hit)
	    break;
	}
	if(hit){
	  ptmp2->name += "," + ptmp->name + "." + ptmp->al;
	}
	else{
	  ptmp2->next = new PROPERTY();
	  ptmp2 = ptmp2->next;
	  ptmp2->next = NULL;
	  ptmp2->length = ptmp->length-ptmp->ee[ptmp->excnt-1];
	  ptmp2->seq = new char[ptmp2->length];
	  for(j=0;j<ptmp2->length;j++){
	    ptmp2->seq[j] = ptmp->seq[ptmp->ee[ptmp->excnt-1]+j];
	  }
	  ptmp2->name = ptmp->name + "." + ptmp->al;
	}
      }
    }

    int exnum;
    k2 = 0;
    k3 = 0;
    gtmp = gtop;
    string tmpname;
    while(gtmp->next){
      gtmp = gtmp->next;
      for(i=0;i<gtmp->ex;i++){
	ptmp = gtmp->sameex[i];
	exnum = 1;
	while(ptmp->next){
	  ptmp = ptmp->next;
	  if(ptmp->unuse)
	    continue;
	  
	  tmpname = gtmp->g + ":Exon" + deci_to_st(i+1) + "_" + deci_to_st(exnum);
	  gtmp->outen[i] << ">" << tmpname 
			 << " " << ptmp->length << ".bp Nlen=" 
			 << N_BASE << ".bp";
	  outallexn << ">" << tmpname << " " << ptmp->length << ".bp Nlen=" 
		    << N_BASE << ".bp";
	  outeil << tmpname << "\t" << ptmp->name << endl;
	  k2 = 0;
	  for(j=0;j<N_BASE;j++){
	    if(k2 % 60 == 0){
	      outallexn  << endl;
	    }
	    outallexn  << "N";
	    k2++;
	  }
	  for(j=0;j<ptmp->length;j++){
	    if(k2 % 60 == 0){
	      outallexn  << endl;
	    }
	    outallexn << ptmp->seq[j];	 
	    k2++;
	  }
	  for(j=0;j<N_BASE;j++){
	    if(k2 % 60 == 0){
	      outallexn  << endl;
	    }
	    outallexn  << "N";
	    k2++;
	  }
	  outallexn << endl;

	  k = 0;
	  for(j=0;j<N_BASE;j++){
	    if(k % 60 == 0){
	      gtmp->outen[i]  << endl;
	    }
	    gtmp->outen[i]  << "N";
	    k++;
	  }
	  for(j=0;j<ptmp->length;j++){
	    if(k % 60 == 0){
	      gtmp->outen[i]  << endl;
	    }
	    gtmp->outen[i] << ptmp->seq[j];	 
	    k++;
	  }
	  for(j=0;j<N_BASE;j++){
	    if(k % 60 == 0){
	      gtmp->outen[i]  << endl;
	    }
	    gtmp->outen[i] << "N";
	    k++;
	  }
	  gtmp->outen[i] << endl;
	  exnum++;
	}

	if(i < gtmp->ex-1){
	  ptmp = gtmp->samein[i];
	  exnum = 1;
	  while(ptmp->next){
	    ptmp = ptmp->next;
	    if(ptmp->unuse)
	      continue;
	    
	    tmpname = gtmp->g + ":Intron" + deci_to_st(i+1) + "_" + deci_to_st(exnum);
	    gtmp->outin[i] << ">" << tmpname 
			   << " " << ptmp->length << ".bp Nlen=" 
			   << N_BASE << ".bp";
	    outallintn << ">" << tmpname << " " << ptmp->length << ".bp Nlen=" 
		       << N_BASE << ".bp";
	    outeil << tmpname << "\t" << ptmp->name << endl;

	    k2 = 0;
	    for(j=0;j<N_BASE;j++){
	      if(k2 % 60 == 0){
		outallintn  << endl;
	      }
	      outallintn  << "N";
	      k2++;
	    }
	    for(j=0;j<ptmp->length;j++){
	      if(k2 % 60 == 0){
		outallintn  << endl;
	      }
	      outallintn << ptmp->seq[j];	 
	      k2++;
	    }
	    for(j=0;j<N_BASE;j++){
	      if(k2 % 60 == 0){
		outallintn  << endl;
	      }
	      outallintn  << "N";
	      k2++;
	    }
	    outallintn << endl;

	    k = 0;
	    for(j=0;j<N_BASE;j++){
	      if(k % 60 == 0){
		gtmp->outin[i]  << endl;
	      }
	      gtmp->outin[i]  << "N";
	      k++;
	    }
	    for(j=0;j<ptmp->length;j++){
	      if(k % 60 == 0){
		gtmp->outin[i]  << endl;
	      }
	      gtmp->outin[i] << ptmp->seq[j];	 
	      k++;
	    }
	    for(j=0;j<N_BASE;j++){
	      if(k % 60 == 0){
		gtmp->outin[i]  << endl;
	      }
	      gtmp->outin[i] << "N";
	      k++;
	    }
	    gtmp->outin[i] << endl;	  
	    exnum++;
	  }
	}
      }
      
      exnum = 1;
      ptmp = gtmp->sameutr[0];
      while(ptmp->next){
	ptmp = ptmp->next;
	if(ptmp->unuse)
	  continue;
	
	tmpname = gtmp->g + ":5utr_" + deci_to_st(exnum);
	gtmp->outu5 << ">" << tmpname 
		    << " " << ptmp->length << ".bp Nlen=" 
		    << N_BASE << ".bp";
	outallutr << ">" << tmpname << " " << ptmp->length << ".bp Nlen=" 
		  << N_BASE << ".bp";
	outeil << tmpname << "\t" << ptmp->name << endl;
	k2 = 0;
	for(j=0;j<N_BASE;j++){
	  if(k2 % 60 == 0){
	    outallutr  << endl;
	  }
	  outallutr  << "N";
	  k2++;
	}
	for(j=0;j<ptmp->length;j++){
	  if(k2 % 60 == 0){
	    outallutr  << endl;
	  }
	  outallutr << ptmp->seq[j];	 
	  k2++;
	}
	for(j=0;j<N_BASE;j++){
	  if(k2 % 60 == 0){
	    outallutr  << endl;
	  }
	  outallutr  << "N";
	  k2++;
	}
	outallutr << endl;
	
	k = 0;
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    gtmp->outu5  << endl;
	  }
	  gtmp->outu5  << "N";
	  k++;
	}
	for(j=0;j<ptmp->length;j++){
	  if(k % 60 == 0){
	    gtmp->outu5  << endl;
	  }
	  gtmp->outu5 << ptmp->seq[j];	 
	  k++;
	}
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    gtmp->outu5  << endl;
	  }
	  gtmp->outu5 << "N";
	  k++;
	}
	gtmp->outu5 << endl;
	exnum++;
      }
      exnum = 1;
      ptmp = gtmp->sameutr[1];
      while(ptmp->next){
	ptmp = ptmp->next;
	if(ptmp->unuse)
	  continue;
	
	tmpname = gtmp->g + ":3utr_" + deci_to_st(exnum);
	gtmp->outu3 << ">" << tmpname 
		    << " " << ptmp->length << ".bp Nlen=" 
		    << N_BASE << ".bp";
	outallutr << ">" << tmpname << " " << ptmp->length << ".bp Nlen=" 
		  << N_BASE << ".bp";
	outeil << tmpname << "\t" << ptmp->name << endl;
	k2 = 0;
	for(j=0;j<N_BASE;j++){
	  if(k2 % 60 == 0){
	    outallutr  << endl;
	  }
	  outallutr  << "N";
	  k2++;
	}
	for(j=0;j<ptmp->length;j++){
	  if(k2 % 60 == 0){
	    outallutr  << endl;
	  }
	  outallutr << ptmp->seq[j];	 
	  k2++;
	}
	for(j=0;j<N_BASE;j++){
	  if(k2 % 60 == 0){
	    outallutr  << endl;
	  }
	  outallutr  << "N";
	  k2++;
	}
	outallutr << endl;
	
	k = 0;
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    gtmp->outu3  << endl;
	  }
	  gtmp->outu3  << "N";
	  k++;
	}
	for(j=0;j<ptmp->length;j++){
	  if(k % 60 == 0){
	    gtmp->outu3  << endl;
	  }
	  gtmp->outu3 << ptmp->seq[j];	 
	  k++;
	}
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    gtmp->outu3  << endl;
	  }
	  gtmp->outu3 << "N";
	  k++;
	}
	gtmp->outu3 << endl;
	exnum++;
      }
    }
			  			  
    bool *exexist = new bool[MAX_EXON];
    ptmp = ptop;
    while(ptmp->next){
      ptmp = ptmp->next;
      if(ptmp->unuse)
	continue;
      
      if(ptmp->pse)
	continue;
      outa << ptmp->name << "\t" << ptmp->al << endl;
      ptmp->glc->outl << ptmp->name << "\t" << ptmp->al;
      if(ptmp->eg)
	ptmp->glc->outl << "\t+";
      else
	ptmp->glc->outl << "\t-";
      if(!ptmp->par)
	ptmp->glc->outl << "\t+";
      else
	ptmp->glc->outl << "\t-";
      for(i=0;i<ptmp->glc->ex;i++)
	exexist[i] = false;
      for(i=0;i<ptmp->excnt;i++)
	exexist[ptmp->eid[i]] = true;
      for(i=0;i<ptmp->glc->ex;i++){
	if(exexist[i])
	  ptmp->glc->outl << "\t+";
	else
	  ptmp->glc->outl << "\t-";
      }
      ptmp->glc->outl << endl;
        

      if(ptmp->eg){	
	k = 0;
	outallgn << ">" << ptmp->g << ":" << ptmp->name << " " << ptmp->al << " " 
		 << ptmp->ee[ptmp->excnt-1]-ptmp->es[0]+1
		 << ".bp Nlen=" << N_BASE << ".bp";
      	
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    outallgn  << endl;
	  }
	  outallgn  << "N";
	  k++;
	}
	for(i=ptmp->es[0]-1;i<ptmp->ee[ptmp->excnt-1];i++){
	  if(k % 60 == 0){
	    outallgn << endl;
	  }
	  outallgn << ptmp->seq[i];
	  k++;
	}
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    outallgn << endl;
	  }
	  outallgn << "N";
	  k++;
	}
	outallgn << endl;

	k = 0;
	outallgnutr << ">" << ptmp->g << ":" << ptmp->name << " " << ptmp->al << " " 
		    << ptmp->length
		    << ".bp Nlen=" << N_BASE << ".bp";
      	
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    outallgnutr  << endl;
	  }
	  outallgnutr  << "N";
	  k++;
	}
	for(i=0;i<ptmp->length;i++){
	  if(k % 60 == 0){
	    outallgnutr << endl;
	  }
	  outallgnutr << ptmp->seq[i];
	  k++;
	}
	for(j=0;j<N_BASE;j++){
	  if(k % 60 == 0){
	    outallgnutr << endl;
	  }
	  outallgnutr << "N";
	  k++;
	}
	outallgnutr << endl;

	
	ptmp->glc->outgn << ">" << ptmp->name << " " << ptmp->al << " " << ptmp->ee[ptmp->excnt-1]-ptmp->es[0]+1
			 << ".bp Nlen=" << N_BASE << ".bp";
	
	k = 0;
	for(i=0;i<N_BASE;i++){
	  if(k % 60 == 0){
	    ptmp->glc->outgn  << endl;
	  }
	  ptmp->glc->outgn  << "N";
	  k++;
	}
	for(i=ptmp->es[0]-1;i<ptmp->ee[ptmp->excnt-1];i++){
	  if(k % 60 == 0){
	    ptmp->glc->outgn  << endl;
	  }
	  ptmp->glc->outgn << ptmp->seq[i];	 
	  k++;
	}
	for(i=0;i<N_BASE;i++){
	  if(k % 60 == 0){
	    ptmp->glc->outgn  << endl;
	  }
	  ptmp->glc->outgn  << "N";
	  k++;
	}
	ptmp->glc->outgn << endl;


	ptmp->glc->outgnutr << ">" << ptmp->name << " " << ptmp->al << " " << ptmp->length
			    << ".bp Nlen=" << N_BASE << ".bp";
	
	k = 0;
	for(i=0;i<N_BASE;i++){
	  if(k % 60 == 0){
	    ptmp->glc->outgnutr  << endl;
	  }
	  ptmp->glc->outgnutr  << "N";
	  k++;
	}
	for(i=0;i<ptmp->length;i++){
	  if(k % 60 == 0){
	    ptmp->glc->outgnutr  << endl;
	  }
	  ptmp->glc->outgnutr << ptmp->seq[i];	 
	  k++;
	}
	for(i=0;i<N_BASE;i++){
	  if(k % 60 == 0){
	    ptmp->glc->outgnutr  << endl;
	  }
	  ptmp->glc->outgnutr  << "N";
	  k++;
	}
	ptmp->glc->outgnutr << endl;
       
      }

    }

    string gname;
    string pname,pbase;
    
    if(useother){
      in.open(otname.c_str(),ios::in);
      while(getline(in,sline)){
	n = 0;
	word = GetWordtab(n,sline);
	gname = GetWordtab(n,sline);
	pbase = "";
	in2.open(word.c_str(),ios::in);
	if(in2){
	  while(getline(in2,sline)){
	    if(sline[0] == '>'){
	      if(pbase.size() > 0){
		outallexn << ">" << gname << ":" << pname << " " << pbase.size() << ".bp Nlen=" 
			  << N_BASE << ".bp";
		outallgn << ">" << gname << ":" << pname << " " << pbase.size() << ".bp Nlen=" 
			 << N_BASE << ".bp";
		k2 = 0;
		for(j=0;j<N_BASE;j++){
		  if(k2 % 60 == 0){
		    outallexn  << endl;
		  }
		  outallexn  << "N";
		  k2++;
		}
		for(j=0;j<pbase.size();j++){
		  if(k2 % 60 == 0){
		    outallexn  << endl;
		  }
		  outallexn << pbase[j];	 
		  k2++;
		}
		for(j=0;j<N_BASE;j++){
		  if(k2 % 60 == 0){
		    outallexn  << endl;
		  }
		  outallexn  << "N";
		  k2++;
		}
		outallexn << endl;

		k2 = 0;
		for(j=0;j<N_BASE;j++){
		  if(k2 % 60 == 0){
		    outallgn  << endl;
		  }
		  outallgn  << "N";
		  k2++;
		}
		for(j=0;j<pbase.size();j++){
		  if(k2 % 60 == 0){
		    outallgn  << endl;
		  }
		  outallgn << pbase[j];	 
		  k2++;
		}
		for(j=0;j<N_BASE;j++){
		  if(k2 % 60 == 0){
		    outallgn  << endl;
		  }
		  outallgn  << "N";
		  k2++;
		}
		outallgn << endl;

		outa << pname << "\t" << pname <<  endl;		
		outeil << gname << ":" << pname << "\t" << pname << "." << pname << endl;
	      }
	      pbase = "";
	      n = 0;
	      pname = GetWordsp(n,sline);
	      pname = pname.substr(1,pname.size()-1);
	      cout << pname << "\t" << pname << endl;
	    }
	    else{
	      pbase += sline;
	    }	    	    
	  }
	  if(pbase.size() > 0){
	    outallexn << ">" << gname << ":" << pname << " " << pbase.size() << ".bp Nlen=" 
		      << N_BASE << ".bp";

	    outallgn << ">" << gname << ":" << pname << " " << pbase.size() << ".bp Nlen=" 
		     << N_BASE << ".bp";

	    k2 = 0;
	    for(j=0;j<N_BASE;j++){
	      if(k2 % 60 == 0){
		outallexn  << endl;
	      }
	      outallexn  << "N";
	      k2++;
	    }
	    for(j=0;j<pbase.size();j++){
	      if(k2 % 60 == 0){
		outallexn  << endl;
	      }
	      outallexn << pbase[j];	 
	      k2++;
	    }
	    for(j=0;j<N_BASE;j++){
	      if(k2 % 60 == 0){
		outallexn  << endl;
	      }
	      outallexn  << "N";
	      k2++;
	    }
	    outallexn << endl;

	    k2 = 0;
	    for(j=0;j<N_BASE;j++){
	      if(k2 % 60 == 0){
		outallgn  << endl;
	      }
	      outallgn  << "N";
	      k2++;
	    }
	    for(j=0;j<pbase.size();j++){
	      if(k2 % 60 == 0){
		outallgn  << endl;
	      }
	      outallgn << pbase[j];	 
	      k2++;
	    }
	    for(j=0;j<N_BASE;j++){
	      if(k2 % 60 == 0){
		outallgn  << endl;
	      }
	      outallgn  << "N";
	      k2++;
	    }
	    outallgn << endl;

	    outa << pname << "\t" << pname << endl;
	    outeil << gname << ":" << pname << "\t" << pname << "." << pname << endl;
	  }
	  in2.close();
	}
	else{
	  cout << word << " is not found." << endl;
	  exit(0);
	}
      }
      in.close();
    }

  }
};

int main(int narg, char** args){
  if(narg != 4 && narg != 5){
    cout << args[0] << " dat_file outputdir N (other file)" << endl;
    exit(0);
  }

  if(narg == 5){
    useother = true;
    otname = (string)args[4];
  }

  Create_FASTA_F_DAT *c = new Create_FASTA_F_DAT(args[1],args[2],args[3]);
}
  
