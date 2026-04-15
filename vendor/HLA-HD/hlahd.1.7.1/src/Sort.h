void Sort_Only_Intence_All(RANK_TREE *rtree,int lcnt){
  int i,j,k,l,s,ii,jj,kk,ll,ss,n;
  int nowid;
  bool hit,hit2;
  bool tsame,tsame2;
  MPED_FQ *mptmp,*mptmp2,*mptpm3;    
  int tcnt1,tcnt2,tcnt3;         
  int tmpl;
  string word;

  struct FQ_HASH{
    FASTQ_HASH *fq;    
    FQ_HASH *next;
  };
  FQ_HASH *****fh;
  FQ_HASH *fqtmp,*fqtmp2;

  int hcnt,hid;
  char sw;
  int *ls = new int[hsize2];
  hcnt = 1;
  for(i=0;i<hsize2;i++){
    hcnt *= 10;
  }
  
  ls[0] = 1;
  for(i=1;i<hsize2;i++)
    ls[i] = 10*ls[i-1];
  
  fh = new FQ_HASH****[rtree->hithla];
  for(j=0;j<rtree->hithla;j++){
    fh[j] = NULL;
  }

  int tmpsameidcnt = rtree->sameidcnt;
  
  for(i=0;i<rtree->sameidcnt;i++){
    for(j=0;j<rtree->hithla;j++){
      if(rtree->sameid[j]==i){
	fh[j] = new FQ_HASH***[lcnt];
	for(k=0;k<lcnt;k++){
	  if(rtree->used[k] == 0)
	    continue;
	  
	  fh[j][k] = new FQ_HASH**[2];	  
	  for(ii=0;ii<2;ii++){
	    
	    fh[j][k][ii] = new FQ_HASH*[hcnt];		    

	    for(l=0;l<hcnt;l++){
	      fh[j][k][ii][l] = NULL;
	      //fh[j][k][ii][l] = new FQ_HASH();
	      //fh[j][k][ii][l]->next = NULL;		
	    }
	    
	    mptmp = rtree->hrank[j]->mp[ii][k];
	    while(mptmp->next){
	      mptmp = mptmp->next;
	      
	      word = mptmp->fq->id;
	      hid = 0;	      
	      jj = word.size();
	      for(kk=0;kk<hsize2;kk++){
		if(hpos[kk]>=jj)
		  continue;
		sw = word[jj-hpos[kk]-1];
		if ( sw < '0' || sw > '9' ) {
		  ss = 0;
		} else {	
		  ss = (int)(sw - '0');
		}
		hid += ls[kk]*ss;
	      }	      	      
	 	
	      if(!fh[j][k][ii][hid]){
		fh[j][k][ii][hid] = new FQ_HASH();
		fqtmp = fh[j][k][ii][hid];
		fqtmp->next = NULL;
		fqtmp->next = new FQ_HASH();
		fqtmp = fqtmp->next;
		fqtmp->next = NULL;
		fqtmp->fq = mptmp->fq;
	      }else{
		fqtmp = fh[j][k][ii][hid];
		while(fqtmp->next){
		  fqtmp = fqtmp->next;
		}
		fqtmp->next = new FQ_HASH();
		fqtmp = fqtmp->next;
		fqtmp->next = NULL;
		fqtmp->fq = mptmp->fq;
	      }
	    }
	  }
	}
	break;
      }
    }
  }
  
  nowid = 0;
  for(i=0;i<rtree->sameidcnt;i++){
    tsame = false;
    tsame2 = false;
    for(j=0;j<rtree->hithla;j++){
      if(rtree->sameid[j] == i){
	for(k=0;k<rtree->sameidcnt;k++){
	  for(s=0;s<rtree->hithla;s++){
	    if(rtree->sameid[s] == k){
	      if(i==k)
		continue;
	      if(rtree->hrank[j]->tmpmm < rtree->hrank[s]->tmpmm){
		k = rtree->sameidcnt;
		break;
	      }
	      tcnt1 = 0;
	      tcnt2 = 0;
	      hit2 = true;
	      for(ii=0;ii<2;ii++){
		for(l=0;l<lcnt;l++){
		  if(rtree->used[l] == 0)
		    continue;
		  
		  mptmp = rtree->hrank[j]->mp[ii][l];
		  
		  while(mptmp->next){
		    mptmp = mptmp->next;
		    tcnt1++;
		    
		    word = mptmp->fq->id;
		    hid = 0;
		    jj = word.size();

		    for(kk=0;kk<hsize2;kk++){
		      if(hpos[kk]>=jj)
			continue;
		      sw = word[jj-hpos[kk]-1];
		      if ( sw < '0' || sw > '9' ) {
			ss = 0;
		      } else {	
			ss = (int)(sw - '0');
		      }
		      hid += ls[kk]*ss;
		    }		    		    	

		    hit = false;
		    
		    fqtmp = fh[s][l][ii][hid];
		    if(fqtmp){
		      while(fqtmp->next){
			fqtmp = fqtmp->next;
			if(fqtmp->fq == mptmp->fq){
			  hit = true;
			  tcnt2++;
			  break;
			}
		      }
		    }

		    if(!hit){
		      hit2 = false;
		      break;
		    }
		  }
		  if(!hit2)
		    break;
		}
		if(!hit2)
		  break;
	      }
	  
	      if(hit2){
		tcnt3 = 0;
		for(ii=0;ii<2;ii++){
		  for(l=0;l<lcnt;l++){
		    if(rtree->used[l] == 0)
		      continue;
		    mptmp2 = rtree->hrank[s]->mp[ii][l];

		    while(mptmp2->next){
		      mptmp2 = mptmp2->next;
		      if(mptmp->fq == mptmp2->fq){
			hit = true;
			tcnt3++;
			break;
		      }
		    }
		  }
		}	
		if(tcnt3 > tcnt1){
		  if(rtree->hrank[j]->tmpmm == rtree->hrank[s]->tmpmm){
		    if(rtree->hrank[j]->tmprsum < rtree->hrank[s]->tmprsum){	
		      tsame = true;
		    }
		  }
		  else
		    tsame = true;			  
		}
	      }
	      
	      //3,Oct,2014 overcome HLA-B*15:01:01:02N problem
	      
	      if(!tsame && rtree->hrank[j]->tmpmm > 0 && rtree->hrank[s]->tmpmm == 0){
		hit = false;

		for(l=0;l<lcnt;l++){
		  if(rtree->used[l] == 0)
		    continue;		
	
		  tmpl=rtree->hrank[s]->ee[l]-rtree->hrank[s]->es[l]+1;
		  
		  if(rtree->hrank[j]->ee[l]-rtree->hrank[j]->es[l]+1 < tmpl){
		    hit = true;
		    break;
		  }
		  hit2 = true;
		  for(n=0;n<tmpl;n++){
		    if(rtree->hrank[j]->code[l][rtree->hrank[j]->es[l]+n] != 
		       rtree->hrank[s]->code[l][rtree->hrank[s]->es[l]+n]){
		      hit2 = false;
		      break;
		    }
		  }
		  if(!hit2){
		    hit2 = true;
		    for(n=0;n<tmpl;n++){
		      if(rtree->hrank[j]->code[l][rtree->hrank[j]->ee[l]-n] != 
			 rtree->hrank[s]->code[l][rtree->hrank[s]->ee[l]-n]){
			hit2 = false;
			break;
		      }
		    }
		  }
	
		  if(!hit2){
		    hit = true;
		    break;
		  }
		}
		if(!hit)
		  tsame2 = true;
	      }
	           
	      break;		  
	    } 
	  }
	  if(tsame || tsame2)
	    break;
	}
	break;
      }
    }
    if(!tsame && !tsame2){
      for(j=0;j<rtree->hithla;j++){
	if(rtree->sameid[j] == i){
	  rtree->sameid[j] = nowid;
	}
      }
      nowid++;
      if(nowid == SORTMAX)
	break;
    }  
    else{    
      for(j=0;j<rtree->hithla;j++){
	if(rtree->sameid[j] == i){
	  rtree->sameid[j] = rtree->hithla;
	}
      }
    }  
  }
  rtree->sameidcnt = nowid;

  for(i=0;i<tmpsameidcnt;i++){
    for(j=0;j<rtree->hithla;j++){
      if(rtree->sameid[j]==i){
	for(k=0;k<lcnt;k++){
	  if(rtree->used[k] == 0)
	    continue;
	  
	  for(ii=0;ii<2;ii++){
	    for(l=0;l<hcnt;l++){
	      if(fh[j][k][ii][l]){
		fqtmp = fh[j][k][ii][l];
		while(fqtmp->next){
		  fqtmp2 = fqtmp->next;
		  delete fqtmp;
		  fqtmp = fqtmp2;
		}
		delete fqtmp;
	      }   
	    }
	    delete[] fh[j][k][ii];	    
	  }		
	  delete[] fh[j][k];
	}
	delete[] fh[j];
	break;
      }
    }
  } 
  delete[] fh;
  delete[] ls;

}


void ReSort_Best(RANK_TREE *rtree){
  int i,j,s;
  int *bestid = new int[rtree->bestcnt];
  for(i=0;i<rtree->bestcnt;i++){
    for(j=0;j<rtree->hithla;j++){
      if(rtree->hrank[j] == rtree->hlabest[i]){
	bestid[i] = rtree->sameid[j];
	break;
      }
    }
  }

  bool hit;
  for(i=0;i<rtree->hithla;i++){
    hit = false;
    for(j=0;j<rtree->bestcnt;j++){
      if(rtree->sameid[i] == bestid[j]){
	rtree->sameid[i] = j;
	hit = true;
	break;
      }
    }
    if(!hit)
      rtree->sameid[i] = rtree->hithla;
  }
}


