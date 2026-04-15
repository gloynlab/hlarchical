void Calc_Rank_All(RANK_TREE *rtree,HLA_HASH **hh,int lcnt,int hcnt,int **lmode,int *maxlm){
  int i,j,k,l,s,ii,jj,kk,ll,ss,n,n2;
  
  HLA_HASH *htmp;
  int nowid;
  bool hit,hit2;
  string word,word2,word3,word4;
  rtree->hithla = 0;
  rtree->sameidcnt = 0;

  int tmpmm;
  int tmpmmp;
  int tmpnm,tmpnmp;
  int tmpv;
  double tmpvcnt,tmpv2cnt;
  int tmpn;
  int tmplen;
  int tmprsum;
  double mmr;

  MPED_FQ *mptmp,*mptmp2;

  clock_t start, end;
  
  for(i=0;i<hcnt;i++){
    htmp = hh[i];
    while(htmp->next){
      htmp = htmp->next;
                     
      hit = true;
      for(l=0;l<lcnt;l++){
	if(rtree->used[l] == 0)
	  continue;
	if(!htmp->hit[l]){
	  hit = false;
	  break;
	}
	hit = false;
	for(j=0;j<maxlm[l];j++){
	  if(htmp->ee[l]-htmp->es[l]+1 == lmode[l][j]){
	    hit = true;	
	    break;
	  }
	}
	if(!hit)
	  break;
      }
      if(!hit)
	continue;
      
      tmpnm = 0;
      tmpnmp = 0;
      tmpmm = 0;
      tmpmmp = 0;
      tmplen = 0;
      tmpvcnt = 0;
      tmpv2cnt = 0;
      tmprsum = 0;
      
      for(l=0;l<lcnt;l++){
	if(rtree->used[l] == 0)
	  continue;
	tmpnm += htmp->nmcnt[l];
	tmpnmp += htmp->nmcntp[l];
	tmpmm += htmp->mm[l];
	tmpmmp += htmp->mmp[l];
	tmpvcnt += htmp->exvcnt[l];
	tmpv2cnt += htmp->exv2cnt[l];
	tmplen += htmp->ee[l] - htmp->es[l] + 1;
	tmprsum += htmp->rsum[l];
	
      }

      if(tmpnm <= 0)
	continue;
      
      if((double)tmpmm/tmplen > rtree->covth)
	continue;
      rtree->hrank[rtree->hithla] = htmp;
      htmp->tmpmm = tmpmm;
      htmp->tmpmmp = tmpmmp;
      htmp->tmpnm = tmpnm;
      htmp->tmpnmp = tmpnmp;
      htmp->tmpv = (double)tmpmm/tmplen;
      htmp->tmplen = tmplen;
      htmp->tmpvcnt = tmpvcnt;
      htmp->tmpv2cnt = tmpv2cnt;
      htmp->tmprsum = tmprsum; 
      rtree->hithla++;
    }
  }
  
  for(i=0;i<rtree->hithla-1;i++){    
    for(j=i+1;j<rtree->hithla;j++){
      if(rtree->hrank[i]->tmpnm == 0){
	htmp = rtree->hrank[i];
	rtree->hrank[i] = rtree->hrank[j];
	rtree->hrank[j] = htmp;	
      }
      else{
	if(rtree->hrank[i]->tmpmm > rtree->hrank[j]->tmpmm){
	  htmp = rtree->hrank[i];
	  rtree->hrank[i] = rtree->hrank[j];
	  rtree->hrank[j] = htmp;
	}
	else if(rtree->hrank[i]->tmpmm == rtree->hrank[j]->tmpmm){
	  if(rtree->hrank[i]->tmpmmp > rtree->hrank[j]->tmpmmp){
	    htmp = rtree->hrank[i];
	    rtree->hrank[i] = rtree->hrank[j];
	    rtree->hrank[j] = htmp;
	  }
	  else if(rtree->hrank[i]->tmpmmp == rtree->hrank[j]->tmpmmp){
	    if(rtree->hrank[i]->tmpvcnt < rtree->hrank[j]->tmpvcnt){
	      htmp = rtree->hrank[i];
	      rtree->hrank[i] = rtree->hrank[j];
	      rtree->hrank[j] = htmp;
	    }
	  }
	}
      }	  
    }
  }
 
  rtree->sameid[0] = 0;
  nowid = 0;


  int hsum,hid;
  char sw;
  int *ls = new int[hsize];
  hsum = 1;
  for(i=0;i<hsize;i++){
    hsum *= 10;
  }

  struct FQ_HASH{
    FASTQ_HASH *fq;    
    FQ_HASH *next;
  };
  FQ_HASH **fhi,**fhj;
  FQ_HASH *fqtmp,*fqtmp2;
  
  ls[0] = 1;
  for(i=1;i<hsize;i++)
    ls[i] = 10*ls[i-1];
    
  for(i=0;i<rtree->hithla;i++){		
    hit = false;
    for(j=i-1;j>=0;j--){         
      if(rtree->hrank[i]->tmpmm != rtree->hrank[j]->tmpmm)
	break;
      if(rtree->hrank[i]->tmpnm != rtree->hrank[j]->tmpnm)
	continue;
      hit = true;
      for(l=0;l<lcnt;l++){
	if(rtree->used[l]==0)
	  continue;
	if(rtree->hrank[i]->len[l] != rtree->hrank[j]->len[l]){
	  hit=false;
	  break;
	}
      }
      if(!hit)
	continue;
      hit = true;
      for(l=0;l<lcnt;l++){
	if(rtree->used[l]==0)
	  continue;
	for(k=0;k<rtree->hrank[i]->len[l];k++){
	  if(rtree->hrank[i]->code[l][k] != rtree->hrank[j]->code[l][k]){
	    hit = false;
	    break;
	  }
	}
	if(!hit)
	  break;
      }
      if(hit){
	rtree->sameid[i] = rtree->sameid[j];
	break;
      }

      //mismatch same 23, May, 2015
      if(rtree->hrank[i]->tmpvcnt == rtree->hrank[j]->tmpvcnt){
	hit2 = true;
	for(ii=0;ii<2;ii++){
	  for(l=0;l<lcnt;l++){
	    if(rtree->used[l] == 0)
	      continue;
	       
	    fhi = new FQ_HASH*[hsum];		    
	    fhj = new FQ_HASH*[hsum];		    
	    for(jj=0;jj<hsum;jj++){
	      fhi[jj]= new FQ_HASH();
	      fhi[jj]->next = NULL;
	      fhj[jj]= new FQ_HASH();
	      fhj[jj]->next = NULL;
	    }

	    mptmp = rtree->hrank[i]->mp[ii][l];
	    while(mptmp->next){
	      mptmp = mptmp->next;
	      
	      word = mptmp->fq->id;
	      hid = 0;
	      jj = word.size();
	      kk = 0;
	      ll = 0;
	      while(1){
		if(jj-ll-1 < 0)
		  break;
		sw = word[jj-ll-1];
		if ( sw < '0' || sw > '9' ) {
		  ll++;
		  continue;
		} else {	
		  ss = (int)(sw - '0');
		}
		hid += ls[kk]*ss;
		kk++;
		if(kk == hsize)
		  break;
		ll++;
	      }
	 	
	      fqtmp = fhi[hid];
	      while(fqtmp->next){
		fqtmp = fqtmp->next;
	      }
	      fqtmp->next = new FQ_HASH();
	      fqtmp = fqtmp->next;
	      fqtmp->next = NULL;
	      fqtmp->fq = mptmp->fq;
	    }

	    mptmp = rtree->hrank[j]->mp[ii][l];
	    while(mptmp->next){
	      mptmp = mptmp->next;
	      
	      word = mptmp->fq->id;
	      hid = 0;
	      jj = word.size();
	      kk = 0;
	      ll = 0;
	      while(1){
		if(jj-ll-1 < 0)
		  break;
		sw = word[jj-ll-1];
		if ( sw < '0' || sw > '9' ) {
		  ll++;
		  continue;
		} else {	
		  ss = (int)(sw - '0');
		}
		hid += ls[kk]*ss;
		kk++;
		if(kk == hsize)
		  break;
		ll++;
	      }
	 	
	      fqtmp = fhj[hid];
	      while(fqtmp->next){
		fqtmp = fqtmp->next;
	      }
	      fqtmp->next = new FQ_HASH();
	      fqtmp = fqtmp->next;
	      fqtmp->next = NULL;
	      fqtmp->fq = mptmp->fq;
	    }
	    
	    
	    mptmp = rtree->hrank[i]->mp[ii][l];
	    while(mptmp->next){
	      mptmp = mptmp->next;

	      word = mptmp->fq->id;
	      hid = 0;
	      jj = word.size();
	      kk = 0;
	      ll = 0;
	      while(1){
		if(jj-ll-1 < 0)
		  break;
		sw = word[jj-ll-1];
		if ( sw < '0' || sw > '9' ) {
		  ll++;
		  continue;
		} else {	
		  ss = (int)(sw - '0');
		}
		hid += ls[kk]*ss;
		kk++;
		if(kk == hsize)
		  break;
		ll++;
	      }

	      hit = false;
	      fqtmp = fhj[hid];
	      
	      while(fqtmp->next){
		fqtmp = fqtmp->next;
		if(fqtmp->fq == mptmp->fq){
		  hit = true;
		  break;
		}
	      }

	      /*
	      mptmp2 = rtree->hrank[j]->mp[ii][l];
	      hit = false;
	      while(mptmp2->next){
		mptmp2 = mptmp2->next;
		if(mptmp->fq == mptmp2->fq){
		  hit = true;
		  break;
		}
	      }
	      */
	      
	      if(!hit){
		hit2 = false;
		break;
	      }
	    }
	 

	    if(hit2){
	      mptmp = rtree->hrank[j]->mp[ii][l];
	      while(mptmp->next){
		mptmp = mptmp->next;
		
		word = mptmp->fq->id;
		hid = 0;
		jj = word.size();
		kk = 0;
		ll = 0;
		while(1){
		  if(jj-ll-1 < 0)
		    break;
		  sw = word[jj-ll-1];
		  if ( sw < '0' || sw > '9' ) {
		    ll++;
		    continue;
		  } else {	
		    ss = (int)(sw - '0');
		  }
		  hid += ls[kk]*ss;
		  kk++;
		  if(kk == hsize)
		    break;
		  ll++;
		}
		
		hit = false;
		fqtmp = fhi[hid];
		
		while(fqtmp->next){
		  fqtmp = fqtmp->next;
		  if(fqtmp->fq == mptmp->fq){
		    hit = true;
		    break;
		  }
		}
		
		/*
		mptmp2 = rtree->hrank[i]->mp[ii][l];
		
		hit = false;
		while(mptmp2->next){
		  mptmp2 = mptmp2->next;
		  if(mptmp->fq == mptmp2->fq){
		    hit = true;
		    break;
		  }
		}
		*/

		if(!hit){
		  hit2 = false;
		  break;
		}
	      }
	    }

	    for(jj=0;jj<hsum;jj++){	    
	      fqtmp = fhi[jj];
	      while(fqtmp->next){
		fqtmp2 = fqtmp->next;
		delete fqtmp;
		fqtmp = fqtmp2;
	      }
	      delete fqtmp;	

	      fqtmp = fhj[jj];
	      while(fqtmp->next){
		fqtmp2 = fqtmp->next;
		delete fqtmp;
		fqtmp = fqtmp2;
	      }
	      delete fqtmp;
	    }
	    delete[] fhi;
	    delete[] fhj;
	    if(!hit2)
	      break;
	  }
	  if(!hit2)
	    break;
	}
	if(hit2){
	  rtree->sameid[i] = rtree->sameid[j];
	  break;
	}
      }
    }	  	      	 
    if(!hit){
      rtree->sameid[i] = nowid;
      nowid++;
    }
  }
  rtree->sameidcnt = nowid;
  
  cout << "#Pattern\t" << nowid << endl;
     
  for(i=0;i<rtree->hithla;i++){
    cout << rtree->hrank[i]->id << "\t" << rtree->hrank[i]->aname << "\t" << rtree->hrank[i]->tmpmm
	 << "\t" << rtree->hrank[i]->tmpnm << "\t" <<  rtree->hrank[i]->tmpmmp << "\t" << rtree->hrank[i]->tmpnmp << "\t" << rtree->hrank[i]->tmprsum
	 << "\t" << rtree->sameid[i];
    cout << "\t" << rtree->hrank[i]->tmpvcnt;
    cout << "\t";
    for(j=0;j<lcnt;j++)
      cout << rtree->hrank[i]->hit[j];    
    cout << endl;
  }     
  cout << endl;
  
}
