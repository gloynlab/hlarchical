void Count_Map_Exon(HLA_HASH **hh,int lcnt,int hcnt,List_Set *lset){
  int l,i,j,l2;
  HLA_HASH *htmp,*htmp2;
  MPED_FQ *mptmp,*mptmp2,*mptmp3;
  int hitcnt,hitcntp;
  int tmps[3],tmpe[3]; 
  bool allhit,allhit2;
  int llen,hitlen,hitlen2;
  int *tmphit,*tmphitp;

  for(l=0;l<lcnt;l++){
    for(i=0;i<hcnt;i++){      
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;
	htmp->nmcnt[l]=0;
	htmp->nmcntp[l]=0;
	htmp->rsum[l] = 0;
	if(!htmp->hit[l])
	  continue;
	if(htmp->samee[l])
	  continue;

	allhit = false;
	allhit2 = false;
	llen = htmp->ee[l]-htmp->es[l]+1;
	hitlen = 0;
	hitlen2 = 0;
	tmphit = new int[htmp->len[l]];
	tmphitp = new int[htmp->len[l]];
	for(j=0;j<htmp->len[l];j++){
	  tmphit[j] = 0;
	  tmphitp[j] = 0;
	}
	
	mptmp = htmp->mp[0][l];
	while(mptmp->next){	    
	  mptmp = mptmp->next;
	  
	  if(htmp->aname == "HLA-A*26:01:01:01" || htmp->aname == "HLA-A*26:01:01"){
	    cout << htmp->aname << "\t" << mptmp->fq->id << "\t" << mptmp->s << "\t" << mptmp->e
		 << "\t" << mptmp->fq->w[0] << "\t" << mptmp->fq->w[1] << endl;
	  }
	  
	  if(!mptmp->edge && mptmp->fq->edge[0])
	    continue;

	  if(mptmp->s < 0 || mptmp->e >= htmp->len[l]){
	    continue;
	  }
	  
	  if(!allhit){
	    if(mptmp->fq->w[0] > 0){
	      for(j=mptmp->s;j<=mptmp->e;j++){
		tmphit[j]++;
		if(tmphit[j] >= MIN_MM_COV){
		  if(!htmp->hitcnt[l][j]){
		    htmp->hitcnt[l][j]=true;
		    hitlen++;
		  }
		}
	      }
	    }
	    if(hitlen>=llen)
	      allhit = true;
	  }	  
	  htmp->rsum[l] += mptmp->e-mptmp->s+1;	   
	  htmp->nmcnt[l] += mptmp->fq->ts[0];

	  mptmp2 = htmp->mp[1][l];
	  while(mptmp2->next){
	    mptmp2 = mptmp2->next;
	    if(mptmp->fq == mptmp2->fq){
	      if(!mptmp2->edge && mptmp2->fq->edge[1])
		break;
	      if(!allhit2){
		if(mptmp->fq->w[0] > 0 && mptmp->fq->w[1] > 0){
		  for(j=mptmp->s;j<=mptmp->e;j++){
		    tmphitp[j]++;
		    if(tmphitp[j] >= MIN_MM_COV){
		      if(!htmp->hitcntp[l][j]){
			htmp->hitcntp[l][j] = true;
			hitlen2++;
		      }
		    }
		  }
		  for(j=mptmp2->s;j<=mptmp2->e;j++){
		    tmphitp[j]++;
		    if(tmphitp[j] >= MIN_MM_COV){
		      if(!htmp->hitcntp[l][j]){
			htmp->hitcntp[l][j] = true;
			hitlen2++;
		      }
		    }
		  }
		}
		if(hitlen2 >= llen)
		  allhit2 = true;
	      }
	      //htmp->nmcntp[l]++;
	      mptmp->pair = mptmp2;
	      mptmp2->pair = mptmp;
	      tmps[0]=mptmp->s-mptmp->NS;
	      tmpe[0]=mptmp->e+mptmp->NE;
	      tmps[1]=mptmp2->s-mptmp2->NS;
	      tmpe[1]=mptmp2->e+mptmp2->NE;
	      if(tmpe[0]<tmps[1] || tmpe[1]<tmps[0]){
		htmp->nmcntp[l] += tmpe[1]-tmps[1]+1 + tmpe[0]-tmps[0]+1;
	      }
	      else{
		htmp->nmcntp[l] += abs(tmps[0]-tmps[1]) + abs(tmpe[0]-tmpe[1]);
	      }
	    
	      mptmp->paired = true;
	      mptmp2->paired = true;
	      mptmp->fq->paired = true;
	      break;
	    }
	  }

	  //27-Nov-2014 Intron pair inclusion
	  if(mptmp->fq->inhit){
	    if(!mptmp->paired && lset->intidl[l] >= 0){
	      l2 = lset->intidl[l];
	      if(mptmp->fq->inid[1][l2]){
		if(!allhit2){
		  if(mptmp->fq->w[0] > 0 && mptmp->fq->w[1] > 0){
		    for(j=mptmp->s;j<=mptmp->e;j++){
		      tmphitp[j]++;
		      if(tmphitp[j] >= MIN_MM_COV){
			if(!htmp->hitcntp[l][j]){
			  htmp->hitcntp[l][j] = true;
			  hitlen2++;
			}
		      }
		    }
		  }
		  if(hitlen2 >= llen)
		    allhit2 = true;
		}
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		htmp->nmcntp[l] += tmpe[0]-tmps[0]+1;
		mptmp->opaired[0] = true;
		mptmp->fq->paired = true;
	      }
	    }
	    if(!mptmp->paired && !mptmp->opaired[0] && lset->intidr[l] >= 0){
	      l2 = lset->intidr[l];
	      if(mptmp->fq->inid[1][l2]){
		if(!allhit2){
		  if(mptmp->fq->w[0] > 0 && mptmp->fq->w[1] > 0){
		    for(j=mptmp->s;j<=mptmp->e;j++){
		      tmphitp[j]++;
		      if(tmphitp[j] >= MIN_MM_COV){
			if(!htmp->hitcntp[l][j]){
			  htmp->hitcntp[l][j] = true;
			  hitlen2++;
			}
		      }
		    }
		  }
		  if(hitlen2 >= llen)
		    allhit2 = true;
		}
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		htmp->nmcntp[l] += tmpe[0]-tmps[0]+1;
		mptmp->opaired[0] = true;
		mptmp->fq->paired = true;
	      }
	    }
	  }
	}
	
	mptmp = htmp->mp[1][l];
	while(mptmp->next){
	  mptmp = mptmp->next;	  
	  if(!mptmp->edge && mptmp->fq->edge[1])
	    continue;

	  if(mptmp->s < 0 || mptmp->e >= htmp->len[l]){
	    continue;
	  }
	  
	  if(!allhit){
	    if(mptmp->fq->w[1] > 0){
	      for(j=mptmp->s;j<=mptmp->e;j++){
		tmphit[j]++;
		if(tmphit[j] >= MIN_MM_COV){
		  if(!htmp->hitcnt[l][j]){
		    htmp->hitcnt[l][j] = true;
		    hitlen++;
		  }
		}
	      }
	    }
	    if(hitlen >= llen)
	      allhit = true;
	  }
	  htmp->rsum[l] += mptmp->e-mptmp->s+1;	 
	  htmp->nmcnt[l] += mptmp->fq->ts[1];

	  //27-Nov-2014 Intron pair inclusion
	  if(mptmp->fq->inhit){
	    if(!mptmp->paired && lset->intidl[l] >= 0){
	      l2 = lset->intidl[l];
	      if(mptmp->fq->inid[0][l2]){
		if(!allhit2){
		  if(mptmp->fq->w[0] > 0 && mptmp->fq->w[1] > 0){
		    for(j=mptmp->s;j<=mptmp->e;j++){
		      tmphitp[j]++;
		      if(tmphitp[j] >= MIN_MM_COV){			
			if(!htmp->hitcntp[l][j]){
			  htmp->hitcntp[l][j] = true;
			  hitlen2++;
			}
		      }
		    }
		  }
		  if(hitlen2 >= llen)
		    allhit2 = true;		  
		}
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		htmp->nmcntp[l] += tmpe[0]-tmps[0]+1;
		mptmp->opaired[1] = true;
		mptmp->fq->paired = true;
	      }
	    }
	    if(!mptmp->paired && !mptmp->opaired[1] && lset->intidr[l] >= 0){
	      l2 = lset->intidr[l];
	      if(mptmp->fq->inid[0][l2]){
		if(!allhit2){
		  if(mptmp->fq->w[0] > 0 && mptmp->fq->w[1] > 0){	    
		    for(j=mptmp->s;j<=mptmp->e;j++){
		      tmphitp[j]++;
		      if(tmphitp[j] >= MIN_MM_COV){
			if(!htmp->hitcntp[l][j]){
			  htmp->hitcntp[l][j] = true;
			  hitlen2++;
			}
		      }
		    }
		  }
		  if(hitlen2 >= llen)
		    allhit2 = true;
		}
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		htmp->nmcntp[l] += tmpe[0]-tmps[0]+1;
		mptmp->opaired[1] = true;
		mptmp->fq->paired = true;
	      }
	    }
	  }
	}
	 
	hitcnt = 0;
	hitcntp = 0;
	for(j=htmp->es[l];j<=htmp->ee[l];j++){
	  if(htmp->hitcnt[l][j])
	    hitcnt++;
	  if(htmp->hitcntp[l][j])
	    hitcntp++;
	}
	htmp->mm[l] = htmp->ee[l]-htmp->es[l]+1-hitcnt;
	htmp->mmp[l] = htmp->ee[l]-htmp->es[l]+1-hitcntp;
      
	delete[] tmphit;
	delete[] tmphitp;
      }
    }
  }

  for(l=0;l<lcnt;l++){
    bool wch;
    for(i=0;i<hcnt;i++){
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;
	if(!htmp->hit[l])
	  continue;
	htmp->exvcnt[l] = 0;
	htmp->exv2cnt[l] = 0;

	wch = false;
	mptmp = htmp->mp[0][l];
	while(mptmp->next){	    
	  mptmp = mptmp->next;
	  if(!mptmp->edge && mptmp->fq->edge[0])
	    continue;

	  htmp->exvcnt[l] += (int)(mptmp->fq->w[0]*(double)mptmp->fq->ts[0]);	
	  //htmp->exvcnt[l] += (int)(mptmp->fq->w[0]*(double)(mptmp->e-mptmp->s+1));	
	  if(mptmp->paired){
	    tmps[0]=mptmp->s-mptmp->NS;
	    tmpe[0]=mptmp->e+mptmp->NE;
	    tmps[1]=mptmp->pair->s-mptmp->pair->NS;
	    tmpe[1]=mptmp->pair->e+mptmp->pair->NE;
	    if(tmpe[0]<tmps[1] || tmpe[1]<tmps[0]){
	      htmp->exv2cnt[l] += (int)(mptmp->fq->w[1]*(double)(tmpe[1]-tmps[1]+1) + (int)mptmp->fq->w[0]*(double)(tmpe[0]-tmps[0]+1));
	    }
	    else{
	      if(tmps[0] < tmps[1]){
		htmp->exv2cnt[l] += (int)(mptmp->fq->w[0]*(double)(abs(tmps[0]-tmps[1])));
	      }
	      else{
		htmp->exv2cnt[l] += (int)(mptmp->fq->w[1]*(double)(abs(tmps[0]-tmps[1])));
	      }
	      if(tmpe[0] > tmpe[1]){
		htmp->exv2cnt[l] += (int)(mptmp->fq->w[0]*(double)(abs(tmpe[0]-tmpe[1])));				
	      }
	      else{
		htmp->exv2cnt[l] += (int)(mptmp->fq->w[1]*(double)(abs(tmpe[0]-tmpe[1])));				
	      }
	      //htmp->exv2cnt[l] += (int)(0.5*(mptmp->fq->w[0]+mptmp->fq->w[1])*(double)(abs(tmps[0]-tmps[1]) + abs(tmpe[0]-tmpe[1])));
	    }
	  }
	  else if(mptmp->opaired[0]){
	    tmps[0]=mptmp->s-mptmp->NS;
	    tmpe[0]=mptmp->e+mptmp->NE;
	    htmp->exv2cnt[l] += (int)mptmp->fq->w[0]*(double)(tmpe[0]-tmps[0]+1);
	  }
	  
	  if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[0] && mptmp->fq->paired)){
	    htmp->exvcnt[l]-= (int)(mptmp->fq->w[0]*(double)(mptmp->fq->ts[0]));
	    //htmp->exvcnt[l]-= (int)(mptmp->fq->w[0]*(double)(mptmp->e-mptmp->s+1));
	    htmp->rsum[l] -= mptmp->e-mptmp->s+1;	   
	  }
	  
	}
	mptmp = htmp->mp[1][l];
	while(mptmp->next){	    
	  mptmp = mptmp->next;
	  if(!mptmp->edge && mptmp->fq->edge[1])
	    continue;

	  htmp->exvcnt[l] += (int)(mptmp->fq->w[1]*(double)(mptmp->fq->ts[1]));
		   
	  if(mptmp->opaired[1]){
	    tmps[0]=mptmp->s-mptmp->NS;
	    tmpe[0]=mptmp->e+mptmp->NE;
	    htmp->exv2cnt[l] += (int)mptmp->fq->w[1]*(double)(tmpe[0]-tmps[0]+1);
	  }

	  if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[1] && mptmp->fq->paired)){
	    htmp->exvcnt[l] -= (int)(mptmp->fq->w[1]*(double)(mptmp->fq->ts[1]));
	
	    htmp->rsum[l]-=mptmp->e-mptmp->s-1;	       	  
	  }

	}
      }
    }
  }

  for(l=0;l<lcnt;l++){
    for(i=0;i<hcnt;i++){      
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;
	if(!htmp->hit[l])
	  continue;
	if(!htmp->samee[l])
	  continue;
	htmp2 = htmp->samee[l];
	mptmp = htmp->mp[0][l];
	mptmp2 = htmp2->mp[0][l];
	while(mptmp2->next){
	  mptmp2 = mptmp2->next;
	  mptmp = mptmp->next;
	  mptmp->pair = mptmp2->pair;
	  mptmp->paired = mptmp2->paired;
	  for(j=0;j<2;j++)
	    mptmp->opaired[j] = mptmp2->opaired[j];
	  mptmp->fq->paired = mptmp2->fq->paired;
	}
	mptmp = htmp->mp[1][l];
	mptmp2 = htmp2->mp[1][l];
	while(mptmp2->next){
	  mptmp2 = mptmp2->next;
	  mptmp = mptmp->next;
	  mptmp->pair = mptmp2->pair;
	  mptmp->paired = mptmp2->paired;
	  for(j=0;j<2;j++)
	    mptmp->opaired[j] = mptmp2->opaired[j];
	  mptmp->fq->paired = mptmp2->fq->paired;
	}

	htmp->rsum[l] = htmp2->rsum[l];
	htmp->nmcnt[l] = htmp2->nmcnt[l];
	htmp->nmcntp[l] = htmp2->nmcntp[l];
	htmp->mm[l] = htmp2->mm[l];
	htmp->mmp[l] = htmp2->mmp[l];
	htmp->exvcnt[l] = htmp2->exvcnt[l];
	htmp->exv2cnt[l] = htmp2->exv2cnt[l];
	for(j=htmp->es[l];j<=htmp->ee[l];j++){
	  htmp->hitcnt[l][j] = htmp2->hitcnt[l][j];
	  htmp->hitcntp[l][j] = htmp2->hitcntp[l][j];
	}
      }
    }
  }
}

void Count_Map_Intron(HLA_HASH **hh,int lcnt,int hcnt){
  int l,i,j;
  HLA_HASH *htmp,*htmp2;
  MPED_FQ *mptmp,*mptmp2,*mptmp3;
  int hitcnt,hitcntp;
  int tmps[3],tmpe[3];  
  bool allhit,allhit2;
  int hitlen,hitlen2;
  int llen;

  for(l=0;l<lcnt-1;l++){
    for(i=0;i<hcnt;i++){
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;
	htmp->inmcnt[l]=0;
	htmp->inmcntp[l]=0;
	htmp->irsum[l] = 0;

	if(!htmp->ihit[l])
	  continue;
	if(htmp->samei[l])
	  continue;

	llen = htmp->ie[l]-htmp->is[l]+1;
	allhit = false;
	allhit2 = false;
	hitlen = 0;
	hitlen2 = 0;
	
	mptmp = htmp->imp[0][l];
	while(mptmp->next){	    
	  mptmp = mptmp->next;
	  
	  if(!allhit){
	    for(j=mptmp->s;j<=mptmp->e;j++){
	      if(!htmp->ihitcnt[l][j]){
		htmp->ihitcnt[l][j] = true;
		hitlen++;
	      }
	    }
	    if(hitlen >= llen)
	      allhit = true;
	  }
	  htmp->irsum[l]+= mptmp->e-mptmp->s+1;
	  
	  //htmp->inmcnt[l]++;
	  htmp->inmcnt[l] += mptmp->fq->ts[0];

	  mptmp2 = htmp->imp[1][l];
	  while(mptmp2->next){
	    mptmp2 = mptmp2->next;
	    if(mptmp->fq == mptmp2->fq){
	      if(!allhit2){
		for(j=mptmp->s;j<=mptmp->e;j++){
		  if(!htmp->ihitcntp[l][j]){
		    htmp->ihitcntp[l][j] = true;
		    hitlen2++;
		  }
		}
		for(j=mptmp2->s;j<=mptmp2->e;j++){
		  if(!htmp->ihitcntp[l][j]){
		    htmp->ihitcntp[l][j] = true;
		    hitlen2++;
		  }
		}
		if(hitlen2 >= llen)
		  allhit = true;
	      }

	      //htmp->inmcntp[l]++;
	      mptmp->pair = mptmp2;
	      mptmp2->pair = mptmp;
	      tmps[0]=mptmp->s-mptmp->NS;
	      tmpe[0]=mptmp->e+mptmp->NE;
	      tmps[1]=mptmp2->s-mptmp2->NS;
	      tmpe[1]=mptmp2->e+mptmp2->NE;
	      if(tmpe[0]<tmps[1] || tmpe[1]<tmps[0]){
		htmp->inmcntp[l] += tmpe[1]-tmps[1]+1 + tmpe[0]-tmps[0]+1;
	      }
	      else{
		htmp->inmcntp[l] += abs(tmps[0]-tmps[1]) + abs(tmpe[0]-tmpe[1]);
	      }
	      mptmp->paired = true;
	      mptmp2->paired = true;	    
	      mptmp->fq->ipaired = true;

	      break;
	    }
	  }	     
	}
	
	mptmp = htmp->imp[1][l];
	while(mptmp->next){
	  mptmp = mptmp->next;

	  if(!allhit){
	    for(j=mptmp->s;j<=mptmp->e;j++){
	      if(!htmp->ihitcnt[l][j]){
		htmp->ihitcnt[l][j] = true;
		hitlen++;
	      }
	    }
	    if(hitlen >= llen)
	      allhit = true;
	  }
	  htmp->irsum[l]+=mptmp->e-mptmp->s+1;
	  
	  htmp->inmcnt[l] += mptmp->fq->ts[1];
	}
	 
	hitcnt = 0;
	hitcntp = 0;
	for(j=htmp->is[l];j<=htmp->ie[l];j++){
	  if(htmp->ihitcnt[l][j])
	    hitcnt++;
	  if(htmp->ihitcntp[l][j])
	    hitcntp++;
	}
	htmp->imm[l] = htmp->ie[l]-htmp->is[l]+1-hitcnt;
	htmp->immp[l] = htmp->ie[l]-htmp->is[l]+1-hitcntp;
      }
    }
  }

  for(l=0;l<lcnt-1;l++){
    for(i=0;i<hcnt;i++){
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;
	if(!htmp->ihit[l])
	  continue;
	htmp->invcnt[l] = 0;
	mptmp = htmp->imp[0][l];
	while(mptmp->next){	    
	  mptmp = mptmp->next;

    	  //htmp->invcnt[l]++;
	  htmp->invcnt[l] += mptmp->fq->w[0]*mptmp->fq->ts[0];
	  if(!mptmp->paired && mptmp->fq->paired){
	    htmp->invcnt[l] -= mptmp->fq->w[0]*mptmp->fq->ts[0];
	    htmp->irsum[l]-= mptmp->e-mptmp->s+1;	     
	  }		     
	}
	mptmp = htmp->imp[1][l];
	while(mptmp->next){	    
	  mptmp = mptmp->next;

	  htmp->invcnt[l] += mptmp->fq->w[1]*mptmp->fq->ts[1];
	  if(!mptmp->paired && mptmp->fq->paired){
	    htmp->invcnt[l] -= mptmp->fq->w[1]*mptmp->fq->ts[1];
	    htmp->irsum[l] -= mptmp->e-mptmp->s+1;
	  }	
	}	  
      }
    }
  }

  for(l=0;l<lcnt-1;l++){
    for(i=0;i<hcnt;i++){      
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;

	if(!htmp->ihit[l])
	  continue;
	if(htmp->samei[l]){
	  htmp2 = htmp->samei[l];
	  mptmp = htmp->imp[0][l];
	  mptmp2 = htmp2->imp[0][l];
	  
	  while(mptmp2->next){
	    mptmp2 = mptmp2->next;
	    mptmp = mptmp->next;
	    mptmp->pair = mptmp2->pair;
	    mptmp->paired = mptmp2->paired;
	    mptmp->fq->ipaired = mptmp2->fq->ipaired;
	  }
	  
	  mptmp = htmp->imp[1][l];
	  mptmp2 = htmp2->imp[1][l];
	  while(mptmp2->next){
	    mptmp2 = mptmp2->next;
	    mptmp = mptmp->next;
	    mptmp->pair = mptmp2->pair;
	    mptmp->paired = mptmp2->paired;
	    mptmp->fq->ipaired = mptmp2->fq->ipaired;
	  }
	   
	  htmp->irsum[l] = htmp2->irsum[l];
	  htmp->inmcnt[l] = htmp2->inmcnt[l];
	  htmp->inmcntp[l] = htmp2->inmcntp[l];
	  htmp->imm[l] = htmp2->imm[l];
	  htmp->immp[l] = htmp2->immp[l];
	  htmp->invcnt[l] = htmp2->invcnt[l];
	  for(j=htmp->is[l];j<=htmp->ie[l];j++){
	    htmp->ihitcnt[l][j] = htmp2->ihitcnt[l][j];
	    htmp->ihitcntp[l][j] = htmp2->ihitcntp[l][j];
	  }	  
	}
      }
    }
  }

}

void Count_Diffcnt(RANK_TREE *rtree,int lcnt){
  int i,j,k,ii,l,s,n;
  double hitcnt,hitcnt2,hitcnt3,hitcnt4;
  bool hit,hit2,record,record2;
  HLA_HASH *hlatmp,*hlatmp2;
  MPED_FQ *mptmp,*mptmp2,*mptmp3;
  int tmps[3],tmpe[3];  

  int nowid,nowid2;
  rtree->bestcnt = 0;
  double *tmpdiff1 = new double[BESTMAX];
  double *tmpdiff2 = new double[BESTMAX];
  double *tmpdiff3 = new double[BESTMAX];
  double *tmpdiff4 = new double[BESTMAX];

  FASTQ_HASH **fqlist[BESTMAX][2][lcnt];
  MPED_FQ **mplist[BESTMAX][2][lcnt];
  int mpcnt[BESTMAX][2][lcnt];
  FASTQ_HASH **fqlist2[BESTMAX][lcnt];
  MPED_FQ **mplist2[BESTMAX][lcnt];
  int mpcnt2[BESTMAX][lcnt];

  int mpctmp[2];
  bool *mpfin[2];
  int nows[2];
  
  for(i=0;i<rtree->hithla;i++){
    if(rtree->sameid[i] == 0)
      break;
  }
  if(rtree->hrank[i]->tmpvcnt == 0)
    return;
  rtree->hlabest[rtree->bestcnt] = rtree->hrank[i];
  rtree->mm[rtree->bestcnt] = rtree->hlabest[rtree->bestcnt]->tmpmm;
  rtree->pmm[rtree->bestcnt] = rtree->hlabest[rtree->bestcnt]->tmpmmp;
  rtree->rsum[rtree->bestcnt] = rtree->hlabest[rtree->bestcnt]->tmprsum;
  rtree->diffcnt[rtree->bestcnt][rtree->bestcnt] = 0;
  rtree->diffcnt2[rtree->bestcnt][rtree->bestcnt] = 0;
  hlatmp = rtree->hrank[i];
 
  //Fast calculation 1,Oct,2014

  clock_t start, end;
  start = clock();          

  for(l=0;l<lcnt;l++){
   if(rtree->used[l] == 0)
      continue;
    for(ii=0;ii<2;ii++){
      mpcnt[rtree->bestcnt][ii][l] = 0;
      mptmp = hlatmp->mp[ii][l];
 
      while(mptmp->next){
	mptmp = mptmp->next;

	if(!mptmp->edge && mptmp->fq->edge[ii])
	  continue;	
	if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
	  continue;      
	mpcnt[rtree->bestcnt][ii][l]++;
      }
      fqlist[rtree->bestcnt][ii][l] = new FASTQ_HASH*[mpcnt[rtree->bestcnt][ii][l]];
      mplist[rtree->bestcnt][ii][l] = new MPED_FQ*[mpcnt[rtree->bestcnt][ii][l]];
      
      i = 0;
      mptmp = hlatmp->mp[ii][l];
  
      while(mptmp->next){
	mptmp = mptmp->next;
	if(!mptmp->edge && mptmp->fq->edge[ii])
	  continue;
	if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
	  continue;      

	fqlist[rtree->bestcnt][ii][l][i] = mptmp->fq;
	mplist[rtree->bestcnt][ii][l][i] = mptmp;
	i++;
      }
    }
  }

  end = clock();
  cout << "Create list" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;
      
  start = clock();          
  for(l=0;l<lcnt;l++){
    if(rtree->used[l] == 0)
      continue;
    mpcnt2[rtree->bestcnt][l] = 0;
    mptmp = hlatmp->mp[0][l];
 
    while(mptmp->next){
      mptmp = mptmp->next;

      if(!mptmp->edge && mptmp->fq->edge[0])
	continue;
      
      mptmp2 = mptmp->pair;
      if(mptmp2){
	if(!mptmp2->edge && mptmp2->fq->edge[1])
	  continue;
	if(!mptmp->paired)
	  continue;

	mpcnt2[rtree->bestcnt][l]++;
      }
      else if(mptmp->opaired[0]){
	mpcnt2[rtree->bestcnt][l]++;
      }
    }
    mptmp = hlatmp->mp[1][l];
    
    while(mptmp->next){
      mptmp = mptmp->next;
      if(mptmp->opaired[1]){
	mpcnt2[rtree->bestcnt][l]++;
      }
    }

    fqlist2[rtree->bestcnt][l] = new FASTQ_HASH*[mpcnt2[rtree->bestcnt][l]];
    mplist2[rtree->bestcnt][l] = new MPED_FQ*[mpcnt2[rtree->bestcnt][l]];

    i = 0;
    mptmp = hlatmp->mp[0][l];
  
    while(mptmp->next){
      mptmp = mptmp->next;
      if(!mptmp->edge && mptmp->fq->edge[0])
	continue;
      
      mptmp2 = mptmp->pair;
      if(mptmp2){
	if(!mptmp2->edge && mptmp2->fq->edge[1])
	  continue;
	if(!mptmp->paired)
	  continue;

	fqlist2[rtree->bestcnt][l][i] = mptmp->fq;
	mplist2[rtree->bestcnt][l][i] = mptmp;
	i++;
      }
      else if(mptmp->opaired[0]){
	fqlist2[rtree->bestcnt][l][i] = mptmp->fq;
	mplist2[rtree->bestcnt][l][i] = mptmp;
	i++;
      }
    }
    mptmp = hlatmp->mp[1][l];
   
    while(mptmp->next){
      mptmp = mptmp->next;
      if(mptmp->opaired[1]){
	fqlist2[rtree->bestcnt][l][i] = mptmp->fq;
	mplist2[rtree->bestcnt][l][i] = mptmp;
	i++;
      }
    }    
  }
  end = clock();
  cout << "Set_List:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;

  //End Fast calculation 1,Oct,2014
    
  int alen = 0;
  for(l=0;l<lcnt;l++){
    if(rtree->used[l] == 0)
      continue;
    alen += hlatmp->len[l];
  }
  
  rtree->bestcnt++;
  if(rtree->bestcnt == BESTMAX || rtree->bestcnt == rtree->sameidcnt)
    return;
  nowid = 1;
   
  bool samecode;
  double w0,w1;

  start = clock();
    
  for(i=0;i<rtree->hithla;i++){
    if(rtree->sameid[i] == nowid){
      hlatmp = rtree->hrank[i];
      if(hlatmp->tmpvcnt == 0){
	nowid++;
	continue;       
      }
      //Fast calculation 1,Oct,2014
      for(l=0;l<lcnt;l++){
	if(rtree->used[l] == 0)
	  continue;
	for(ii=0;ii<2;ii++){
	  mpcnt[rtree->bestcnt][ii][l] = 0;
	  mptmp = hlatmp->mp[ii][l];

	  while(mptmp->next){
	    mptmp = mptmp->next;

	    if(!mptmp->edge && mptmp->fq->edge[ii])
	      continue;
	    if((!mptmp->paired && mptmp->fq->paired)&&(!mptmp->opaired[ii] && mptmp->fq->paired))
	      continue;	  
	    mpcnt[rtree->bestcnt][ii][l]++;
	  }
	  fqlist[rtree->bestcnt][ii][l] = new FASTQ_HASH*[mpcnt[rtree->bestcnt][ii][l]];
	  mplist[rtree->bestcnt][ii][l] = new MPED_FQ*[mpcnt[rtree->bestcnt][ii][l]];
	  
	  j = 0;
	  mptmp = hlatmp->mp[ii][l];
	    
	  while(mptmp->next){
	    mptmp = mptmp->next;
	    if(!mptmp->edge && mptmp->fq->edge[ii])
	      continue;
	    if((!mptmp->paired && mptmp->fq->paired)&&(!mptmp->opaired[ii] && mptmp->fq->paired))
	      continue;

	    fqlist[rtree->bestcnt][ii][l][j] = mptmp->fq;
	    mplist[rtree->bestcnt][ii][l][j] = mptmp;
	    j++;
	  }
	}
      }

      for(l=0;l<lcnt;l++){
	if(rtree->used[l] == 0)
	  continue;
	mpcnt2[rtree->bestcnt][l] = 0;
	mptmp = hlatmp->mp[0][l];
	  
	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(!mptmp->edge && mptmp->fq->edge[0])
	    continue;

	  mptmp2 = mptmp->pair;
	  if(mptmp2){
	    if(!mptmp2->edge && mptmp2->fq->edge[1])
	      continue;
	    if(!mptmp->paired)
	      continue;

	    mpcnt2[rtree->bestcnt][l]++;
	  }
	  else if(mptmp->opaired[0]){
	    mpcnt2[rtree->bestcnt][l]++;
	  }
	}
	mptmp = hlatmp->mp[1][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(mptmp->opaired[1]){
	    mpcnt2[rtree->bestcnt][l]++;
	  }
	}
	fqlist2[rtree->bestcnt][l] = new FASTQ_HASH*[mpcnt2[rtree->bestcnt][l]];
	mplist2[rtree->bestcnt][l] = new MPED_FQ*[mpcnt2[rtree->bestcnt][l]];

	j = 0;
	mptmp = hlatmp->mp[0][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(!mptmp->edge && mptmp->fq->edge[0])
	    continue;

	  mptmp2 = mptmp->pair;
	  if(mptmp2){
	    if(!mptmp2->edge && mptmp2->fq->edge[1])
	      continue;
	    if(!mptmp->paired)
	      continue;

	    fqlist2[rtree->bestcnt][l][j] = mptmp->fq;
	    mplist2[rtree->bestcnt][l][j] = mptmp;
	    j++;
	  }
	  else if(mptmp->opaired[0]){
	    fqlist2[rtree->bestcnt][l][j] = mptmp->fq;
	    mplist2[rtree->bestcnt][l][j] = mptmp;
	    j++;
	  }
	}
	mptmp = hlatmp->mp[1][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(mptmp->opaired[1]){
	    fqlist2[rtree->bestcnt][l][j] = mptmp->fq;
	    mplist2[rtree->bestcnt][l][j] = mptmp;
	    j++;
	  }
	}

      }
      //End Fast calculation 1,Oct,2014
      
      record = true;
      record2 = true;
      for(j=0;j<rtree->bestcnt;j++){
	hlatmp2 = rtree->hlabest[j];

	hitcnt = 0;
	hitcnt2 = 0;
	hitcnt3 = 0;
	hitcnt4 = 0;

	for(l=0;l<lcnt;l++){
	  if(rtree->used[l] == 0)
	    continue;
	  samecode = false;
	  if(hlatmp->len[l] == hlatmp2->len[l]){
	    samecode = true;
	    for(s=0;s<hlatmp->len[l];s++){
	      if(hlatmp->code[l][s] != hlatmp2->code[l][s]){
		samecode = false;
		break;
	      }
	    }
	  }
	  if(samecode){
	    continue;
	  }
	  //Fast calculation 1,Oct,2014
	  for(ii=0;ii<2;ii++){
	    mpfin[0] = new bool[mpcnt[rtree->bestcnt][ii][l]];
	    for(s=0;s<mpcnt[rtree->bestcnt][ii][l];s++)
	      mpfin[0][s] = false;
	    mpfin[1] = new bool[mpcnt[j][ii][l]];
	    for(s=0;s<mpcnt[j][ii][l];s++)
	      mpfin[1][s] = false;
	    nows[0] = 0; nows[1] = 0;
	    mpctmp[0] = mpcnt[rtree->bestcnt][ii][l]; mpctmp[1] = mpcnt[j][ii][l];
	    	    
	    while(nows[0]<mpctmp[0] || nows[1]<mpctmp[1]){
	      if(nows[0]<mpctmp[0]){
		if(!mpfin[0][nows[0]]){
		  hit = false;
		  for(s=nows[1];s<mpctmp[1];s++){
		    if(mpfin[1][s])
		      continue;
		    if(fqlist[rtree->bestcnt][ii][l][nows[0]] == fqlist[j][ii][l][s]){
		      mpfin[1][s] = true;
		      hit = true;
		      break;
		    }
		  }
		  if(!hit){
		    hitcnt += (int)(fqlist[rtree->bestcnt][ii][l][nows[0]]->w[ii]*(double)(fqlist[rtree->bestcnt][ii][l][nows[0]]->ts[ii]));
		 		  	
		  }

		  mpfin[0][nows[0]] = true;
		}  	          
		nows[0]++;
	      }

	      if(nows[1]<mpctmp[1]){
		if(!mpfin[1][nows[1]]){
		  hit = false;
		  for(s=nows[0];s<mpctmp[0];s++){
		    if(mpfin[0][s])
		      continue;
		    if(fqlist[rtree->bestcnt][ii][l][s] == fqlist[j][ii][l][nows[1]]){
		      mpfin[0][s] = true;
		      hit = true;
		      break;
		    }
		  }
		  if(!hit){
		    hitcnt2 += (int)(fqlist[j][ii][l][nows[1]]->w[ii]*(double)(fqlist[j][ii][l][nows[1]]->ts[ii]));
		 
		  }
		  
		  mpfin[1][nows[1]] = true;
		}
		nows[1]++;
	      }
	    }
	    	    
	
	    for(k=0;k<2;k++)
	      delete[] mpfin[k];
	  }
	}

	if(record){
	  for(l=0;l<lcnt;l++){
	    if(rtree->used[l] == 0)
	      continue;
	    
	    samecode = false;
	    if(hlatmp->len[l] == hlatmp2->len[l]){
	      samecode = true;
	      for(s=0;s<hlatmp->len[l];s++){
		if(hlatmp->code[l][s] != hlatmp2->code[l][s]){
		  samecode = false;
		  break;
		}
	      }
	    }
	    if(samecode)
	      continue;

	    mpfin[0] = new bool[mpcnt2[rtree->bestcnt][l]];
	    for(s=0;s<mpcnt2[rtree->bestcnt][l];s++)
	      mpfin[0][s] = false;
	    mpfin[1] = new bool[mpcnt2[j][l]];
	    for(s=0;s<mpcnt2[j][l];s++)
	      mpfin[1][s] = false;

	    nows[0] = 0; nows[1] = 0;
	    mpctmp[0] = mpcnt2[rtree->bestcnt][l]; mpctmp[1] = mpcnt2[j][l];
	    	  
	    while(nows[0]<mpctmp[0] || nows[1]<mpctmp[1]){
	      if(nows[0]<mpctmp[0]){
		if(!mpfin[0][nows[0]]){
		  hit = false;
		  for(s=nows[1];s<mpctmp[1];s++){
		    if(mpfin[1][s])
		      continue;
		    if(fqlist2[rtree->bestcnt][l][nows[0]] == fqlist2[j][l][s]){
		      mpfin[1][s] = true;
		      hit = true;
		      break;
		    }
		  }
		  if(!hit){		    			    
		    mptmp = mplist2[rtree->bestcnt][l][nows[0]];
		    		    
		    if(mptmp->opaired[0]){
		      w0 = fqlist2[rtree->bestcnt][l][nows[0]]->w[0];
		      tmps[0]=mptmp->s-mptmp->NS;
		      tmpe[0]=mptmp->e+mptmp->NE;
		      hitcnt3 += (int)(w0*(double)(tmpe[0]-tmps[0]+1));
		    }
		    else if(mptmp->opaired[1]){
		      w1 = fqlist2[rtree->bestcnt][l][nows[0]]->w[1];
		      tmps[0]=mptmp->s-mptmp->NS;
		      tmpe[0]=mptmp->e+mptmp->NE;
		      hitcnt3 += (int)(w1*(double)(tmpe[0]-tmps[0]+1));		    
		    }
		    else{
		      w0 = fqlist2[rtree->bestcnt][l][nows[0]]->w[0];
		      w1 = fqlist2[rtree->bestcnt][l][nows[0]]->w[1];
		      
		      tmps[0]=mptmp->s-mptmp->NS;
		      tmpe[0]=mptmp->e+mptmp->NE;
		      tmps[1]=mptmp->pair->s-mptmp->pair->NS;
		      tmpe[1]=mptmp->pair->e+mptmp->pair->NE;
	
		      if(tmpe[0]<tmps[1] || tmpe[1]<tmps[0]){
			hitcnt3 += (int)(w1*(double)(tmpe[1]-tmps[1]+1)) + (int)(w0*(double)(tmpe[0]-tmps[0]+1));
		      }
		      else{	
			if(tmps[0] < tmps[1]){
			  hitcnt3 += (int)(w0*(double)(abs(tmps[0]-tmps[1])));
			}
			else{
			  hitcnt3 += (int)(w1*(double)(abs(tmps[0]-tmps[1])));
			}
			if(tmpe[0] > tmpe[1]){
			  hitcnt3 += (int)(w0*(double)(abs(tmpe[0]-tmpe[1])));
			}
			else{
			  hitcnt3 += (int)(w1*(double)(abs(tmpe[0]-tmpe[1])));
			}
		      }
		    }
		    		   		    
		  }
		  mpfin[0][nows[0]] = true;
		}  	          
		nows[0]++;
	      }

	      if(nows[1]<mpctmp[1]){
		if(!mpfin[1][nows[1]]){
		  hit = false;
		  for(s=nows[0];s<mpctmp[0];s++){
		    if(mpfin[0][s])
		      continue;
		    if(fqlist2[rtree->bestcnt][l][s] == fqlist2[j][l][nows[1]]){
		      mpfin[0][s] = true;
		      hit = true;
		      break;
		    }
		  }
		  if(!hit){
		    mptmp = mplist2[j][l][nows[1]];
		    		    	       		    
		    if(mptmp->opaired[0]){
		      w0 = fqlist2[j][l][nows[1]]->w[0];
		      tmps[0]=mptmp->s-mptmp->NS;
		      tmpe[0]=mptmp->e+mptmp->NE;
		      hitcnt4 += (int)(w0*(double)(tmpe[0]-tmps[0]+1));		      
		    }
		    else if(mptmp->opaired[1]){
		      w1 = fqlist2[j][l][nows[1]]->w[1];
		      tmps[0]=mptmp->s-mptmp->NS;
		      tmpe[0]=mptmp->e+mptmp->NE;
		      hitcnt4 += (int)(w1*(double)(tmpe[0]-tmps[0]+1));		      	
		    }
		    else{
		      w0 = fqlist2[j][l][nows[1]]->w[0];
		      w1 = fqlist2[j][l][nows[1]]->w[1];

		      tmps[0]=mptmp->s-mptmp->NS;
		      tmpe[0]=mptmp->e+mptmp->NE;
		      tmps[1]=mptmp->pair->s-mptmp->pair->NS;
		      tmpe[1]=mptmp->pair->e+mptmp->pair->NE;
		      if(tmpe[0]<tmps[1] || tmpe[1]<tmps[0]){
			hitcnt4 += (int)(w1*(double)(tmpe[1]-tmps[1]+1)) + (int)(w0*(double)(tmpe[0]-tmps[0]+1));
		      }
		      else{
			if(tmps[0] < tmps[1]){
			  hitcnt4 += (int)(w0*(double)(abs(tmps[0]-tmps[1])));
			}
			else{
			  hitcnt4 += (int)(w1*(double)(abs(tmps[0]-tmps[1])));
			}
			if(tmpe[0] > tmpe[1]){
			  hitcnt4 += (int)(w0*(double)(abs(tmpe[0]-tmpe[1])));
			}
			else{
			  hitcnt4 += (int)(w1*(double)(abs(tmpe[0]-tmpe[1])));
			}		   		
		      }	
		    }

		  }
		  mpfin[1][nows[1]] = true;
		}
		nows[1]++;
	      }
	    }
	  	  
	    for(k=0;k<2;k++)
	      delete[] mpfin[k];	
	    //End Fast calculation 1,Oct,2014	  
	  }
	}
           	

	if(hitcnt4 + hitcnt2 > 0 || hitcnt3 + hitcnt > 0){
	  if((double)(hitcnt4 + hitcnt2)/(hlatmp2->tmpnm)
	     >= (double)homoth*(hitcnt3 + hitcnt)/(hlatmp->tmpnm))
	    record = false;
	}


	if(!record){
	  for(l=0;l<lcnt;l++){
	    if(rtree->used[l] == 0)
	      continue;
	    for(ii=0;ii<2;ii++){		
	      delete[] fqlist[rtree->bestcnt][ii][l];
	    }
	    delete[] fqlist2[rtree->bestcnt][l];
	  }	  
	  break;
	}

	tmpdiff1[j] = hitcnt;
	tmpdiff2[j] = hitcnt2;
	tmpdiff3[j] = hitcnt3;
	tmpdiff4[j] = hitcnt4;
      }
      if(record){
	rtree->hlabest[rtree->bestcnt] = rtree->hrank[i];
	rtree->mm[rtree->bestcnt] = rtree->hlabest[rtree->bestcnt]->tmpmm;
	rtree->pmm[rtree->bestcnt] = rtree->hlabest[rtree->bestcnt]->tmpmmp;
	rtree->rsum[rtree->bestcnt] = rtree->hlabest[rtree->bestcnt]->tmprsum;
	rtree->diffcnt[rtree->bestcnt][rtree->bestcnt] = 0;
	rtree->diffcnt2[rtree->bestcnt][rtree->bestcnt] = 0;
	
	for(j=0;j<rtree->bestcnt;j++){
	  rtree->diffcnt[j][rtree->bestcnt] = tmpdiff2[j];
	  rtree->diffcnt[rtree->bestcnt][j] = tmpdiff1[j];	
	  rtree->diffcnt2[j][rtree->bestcnt] = tmpdiff4[j];
	  rtree->diffcnt2[rtree->bestcnt][j] = tmpdiff3[j];	
	}

	rtree->bestcnt++;
	if(rtree->bestcnt == BESTMAX)
	  return;
      }   
      nowid++;
      if(nowid == rtree->sameidcnt)
	break;
    }
  }
  end = clock();
  cout << "Count_List:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;

  delete[] tmpdiff1;
  delete[] tmpdiff2;
  delete[] tmpdiff3;
  delete[] tmpdiff4;
}

void Get_Exon_Mode(HLA_HASH **hh,int hlacnt,int hcnt,int lcnt,int **&lmode,int *&maxlm){
  int i,j,k,l;
  HLA_HASH *htmp;
  int *cnt = new int[hlacnt];
  int *len = new int[hlacnt];
  int nowcnt;
  int sumcnt,upcnt;
  int tmpl;

  lmode = new int*[lcnt];
  maxlm = new int[lcnt];
  for(l=0;l<lcnt;l++){    
    for(i=0;i<hlacnt;i++){
      cnt[i] = 0;
      len[i] = 0;
    }
    nowcnt = 0;

    for(i=0;i<hcnt;i++){
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;
	if(!htmp->hit[l])
	  continue;
	for(k=0;k<nowcnt;k++){
	  if(len[k] == htmp->ee[l]-htmp->es[l]+1){
	    cnt[k]++;
	    break;
	  }
	}
	if(k==nowcnt){
	  len[nowcnt] = htmp->ee[l]-htmp->es[l]+1;
	  cnt[nowcnt] = 1;
	  nowcnt++;
	}
      }
    }
    
    lmode[l] = new int[nowcnt];
    maxlm[l] = 0;
    for(i=0;i<nowcnt;i++){
      lmode[l][i] = -1;
    }

    for(i=0;i<nowcnt-1;i++){
      for(j=i+1;j<nowcnt;j++){
	if(cnt[i] < cnt[j]){
	  tmpl = len[i];
	  len[i] = len[j];
	  len[j] = tmpl;
	  tmpl = cnt[i];
	  cnt[i] = cnt[j];
	  cnt[j] = tmpl;
	}
      }
    }
    
    sumcnt = 0;
    for(i=0;i<nowcnt;i++){
      sumcnt += cnt[i];
    }
    upcnt = 0;
    for(i=0;i<nowcnt;i++){
      if((double)cnt[i]/sumcnt < 0.01){
	break;
      }      
      lmode[l][i] = len[i];
      upcnt += cnt[i];         
      
    }
    maxlm[l] = i;
  }  
  delete[] cnt;
  delete[] len;
}
