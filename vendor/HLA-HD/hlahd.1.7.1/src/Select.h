void Select_Best_Allele(RANK_TREE *rtree,int lcnt){
  int i,j,k,l,s,t,ii;

  if(rtree->bestcnt == 1){
    rtree->bestpair[0][0] = 0;
    rtree->bestpair[0][1] = -1;
    rtree->paircnt = 1;
  }


  double tmpco;
  double maxco = -10000000;
  rtree->paircnt = 0;

  bool same = true;
  for(i=0;i<rtree->bestcnt-1;i++){
    for(j=i+1;j<rtree->bestcnt;j++){
      if(rtree->diffcnt[i][j] != 0 || rtree->diffcnt[j][i] != 0){
	same = false;
	i = rtree->bestcnt;
	break;
      }
    }
  }

  int chcnt;
  int checkl;
  int checkid;
  
  double **comatrix;
  comatrix = new double*[rtree->bestcnt];
  for(i=0;i<rtree->bestcnt;i++){
    comatrix[i] = new double[rtree->bestcnt];
    for(j=0;j<rtree->bestcnt;j++){
      comatrix[i][j] = 0.0;
    }
  }  

  bool *incl = new bool[rtree->bestcnt];
  for(i=0;i<rtree->bestcnt;i++)
    incl[i] = true;
  if(!same){   
    for(i=0;i<rtree->bestcnt-1;i++){
      for(j=i+1;j<rtree->bestcnt;j++){
	if(rtree->diffcnt[i][j] > 0 || rtree->diffcnt[j][i] > 0){	 
	  if(rtree->mm[j] > 0 && rtree->diffcnt[i][j]+rtree->diffcnt2[i][j] > exthomoth*(rtree->diffcnt[j][i]+rtree->diffcnt2[j][i])){
	    incl[j] = false;
	  }
	  else if(rtree->readcnt[i]/rtree->readcnt[j] > exthomoth){
	    incl[j] = false;
	  }
	  else if(rtree->mm[i] > 0 && rtree->diffcnt[j][i]+rtree->diffcnt2[j][i] > exthomoth*(rtree->diffcnt[i][j]+rtree->diffcnt2[i][j])){
	    incl[i] = false;
	  }
	  else if(rtree->readcnt[j]/rtree->readcnt[i] > exthomoth){
	    incl[i] = false;
	  }
	  else if((double)(rtree->diffcnt[i][j]+rtree->diffcnt2[i][j])/(rtree->readcnt3[i])
		  >= homoth*((double)(rtree->diffcnt[j][i]+rtree->diffcnt2[j][i])/(rtree->readcnt3[j]))){
	    incl[j] = false;
	  }
	  else if((double)(rtree->diffcnt[j][i]+rtree->diffcnt2[j][i])/(rtree->readcnt3[j])
		  >= homoth*((double)(rtree->diffcnt[i][j]+rtree->diffcnt2[i][j])/(rtree->readcnt3[i]))){
	    incl[i] = false;
	  }	  
	  else if(pairth*rtree->diffcnt2[i][j] > rtree->diffcnt[i][j] &&
		  rtree->diffcnt[i][j] > rtree->diffcnt[j][i]){
	    if(rtree->diffcnt2[i][j] > exthomoth*(rtree->diffcnt2[j][i])){
	      if(rtree->mm[j] > 0)
		incl[j] = false;
	    }
	  }
	  else if(pairth*rtree->diffcnt2[j][i] > rtree->diffcnt[j][i] &&
		  rtree->diffcnt[j][i] > rtree->diffcnt[i][j]){
	    if(rtree->diffcnt2[j][i] > exthomoth*(rtree->diffcnt2[i][j])){
	      if(rtree->mm[i] > 0)
		incl[i] = false;
	    }
	  }
	  
	}
      }
    }
    int inclcnt = 0;
    for(i=0;i<rtree->bestcnt;i++){
      if(incl[i])
	inclcnt++;
    }
  
    if(inclcnt <= 0){
      cout << "Wrong combination." << endl;
      exit(0);
    }
    else if(inclcnt == 1){
      for(i=0;i<rtree->bestcnt;i++){
	if(incl[i]){
	  rtree->bestpair[0][0] = i;
	  rtree->bestpair[0][1] = -1;	 
	  rtree->paircnt = 1;
	  break;
	}
      }
    }
    else{
      for(i=0;i<rtree->bestcnt-1;i++){	
	for(j=i+1;j<rtree->bestcnt;j++){
	  tmpco = (double)(rtree->readcnt[i] + rtree->diffcnt[j][i] + rtree->readcnt[j] + rtree->diffcnt[i][j] 
			   + rtree->readcnt2[i] + rtree->diffcnt2[j][i] + rtree->readcnt2[j] + rtree->diffcnt2[i][j]) - mmco*(double)(rtree->pmm[i]+rtree->pmm[j]); 
	  comatrix[i][j] = tmpco;
	  comatrix[j][i] = tmpco;
	  if(!incl[i] || !incl[j])
	    continue;

	  if(fabs(tmpco - maxco) < 0.0000001){
	    int b1 = rtree->bestpair[0][0];
	    int b2 = rtree->bestpair[0][1];

	    rtree->bestpair[rtree->paircnt][0] = i;
	    rtree->bestpair[rtree->paircnt][1] = j;
	    rtree->paircnt++;
	  }	  
	  else if(tmpco > maxco){
	    maxco = tmpco;
	    rtree->bestpair[0][0] = i;
	    rtree->bestpair[0][1] = j;
	    rtree->paircnt = 1;
	  }
	
	}
      }
      
      for(i=0;i<rtree->paircnt;i++){
	int i0 = rtree->bestpair[i][0];
	int i1 = rtree->bestpair[i][1];

	if(rtree->mm[i1] > 0 && rtree->diffcnt[i0][i1]+rtree->diffcnt2[i0][i1] > exthomoth*(rtree->diffcnt[i1][i0]+rtree->diffcnt2[i1][i0])){
	  rtree->bestpair[i][1] = -1;
	}
	else if(rtree->readcnt[i0]/rtree->readcnt[i1] > exthomoth){
	  rtree->bestpair[i][1] = -1;
	}
	else if(rtree->mm[i0] > 0 && rtree->diffcnt[i1][i0]+rtree->diffcnt2[i1][i0] > exthomoth*(rtree->diffcnt[i0][i1]+rtree->diffcnt2[i0][i1])){
	  rtree->bestpair[i][0] = rtree->bestpair[i][1];
	  rtree->bestpair[i][1] = -1;	
	}
	else if(rtree->readcnt[i1]/rtree->readcnt[i0] > exthomoth){
	  rtree->bestpair[i][0] = rtree->bestpair[i][1];
	  rtree->bestpair[i][1] = -1;	
	}
	else if((double)(rtree->diffcnt[i0][i1]+rtree->diffcnt2[i0][i1])/(rtree->readcnt3[i0]) >= homoth*((double)(rtree->diffcnt[i1][i0]+rtree->diffcnt2[i1][i0])/(rtree->readcnt3[i1]))){
	  rtree->bestpair[i][1] = -1;
	}
	else if((double)(rtree->diffcnt[i1][i0]+rtree->diffcnt2[i1][i0])/(rtree->readcnt3[i1])  >= homoth*((double)(rtree->diffcnt[i0][i1]+rtree->diffcnt2[i0][i1])/(rtree->readcnt3[i0]))){
	  rtree->bestpair[i][0] = rtree->bestpair[i][1];
	  rtree->bestpair[i][1] = -1;
	}	
	else if(pairth*rtree->diffcnt2[i0][i1] > rtree->diffcnt[i0][i1]){
	  if(rtree->diffcnt2[i0][i1] > exthomoth*rtree->diffcnt2[i1][i0]){
	    if(rtree->mm[i1] > 0)
	      rtree->bestpair[i][1] = -1;
	  }
	}
	else if(pairth*rtree->diffcnt2[i1][i0] > rtree->diffcnt[i1][i0]){
	  if(rtree->diffcnt2[i1][i0] > exthomoth*rtree->diffcnt2[i0][i1]){
	    if(rtree->mm[i0] > 0){
	      rtree->bestpair[i][0] = rtree->bestpair[i][1];
	      rtree->bestpair[i][1] = -1;
	    }
	  }
	}
	
      }
    }  
  }
  else{
    rtree->paircnt = rtree->bestcnt;
    for(i=0;i<rtree->bestcnt;i++){
      rtree->bestpair[i][0] = i;
      rtree->bestpair[i][1] = -1;
    }
  }

  int tmpbestcnt = rtree->paircnt;
  for(i=rtree->paircnt-1;i>1;i--){
    for(j=i-1;j>=0;j--){
      if(rtree->bestpair[i][0] == rtree->bestpair[j][0] && rtree->bestpair[i][1] == rtree->bestpair[j][1]){
	
	for(k=i;k<tmpbestcnt-1;k++){
	  rtree->bestpair[k][0]=rtree->bestpair[k+1][0];
	  rtree->bestpair[k][1]=rtree->bestpair[k+1][1];
	}
	tmpbestcnt--;
	break;
      }
    }
  }
  rtree->paircnt = tmpbestcnt;
  
  //Fuzzy Decision 15,Jan,2015
  
  MPED_FQ *mptmp,*mptmp2;
  MPED_FQ **mplist[2][2][lcnt],**mplist2[2][lcnt];
  FASTQ_HASH **fqlist[2][2][lcnt];
  FASTQ_HASH **fqlist2[2][lcnt];
  FASTQ_HASH *fqtmp,*fqtmp2;
  int mpcnt[2][2][lcnt],mpcnt2[2][lcnt]; 
  int tmpmpcnt;
  bool hit;
  bool *fqhit[2][lcnt],*fqhit2[lcnt];
  double hitcnt1,hitcnt2,hitcnt3;
  double w0,w1;
  int tmps[2],tmpe[2];

  bool *lsame = new bool[lcnt];
  bool hit2[2];
  int tmplen;

  for(i=0;i<rtree->paircnt;i++){
    int i0 = rtree->bestpair[i][0];
    int i1 = rtree->bestpair[i][1];
    if(i1 >= 0){
      for(l=0;l<lcnt;l++){
	if(rtree->used[l] == 0)
	  continue;
	for(ii=0;ii<2;ii++){	
	  mpcnt[0][ii][l] = 0;

	  mptmp = rtree->hlabest[i0]->mp[ii][l];
	
	  while(mptmp->next){
	    mptmp = mptmp->next;	    
	    if(!mptmp->edge && mptmp->fq->edge[ii])
	      continue;	
	    if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
	      continue;      
	    mpcnt[0][ii][l]++;
	  }
	  tmpmpcnt = mpcnt[0][ii][l];
	  mptmp = rtree->hlabest[i1]->mp[ii][l];

	  while(mptmp->next){
	    mptmp = mptmp->next;	    
	    if(!mptmp->edge && mptmp->fq->edge[ii])
	      continue;	
	    if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
	      continue;      
	    mpcnt[0][ii][l]++;
	  }
	  fqlist[0][ii][l] = new FASTQ_HASH*[mpcnt[0][ii][l]];
	  mplist[0][ii][l] = new MPED_FQ*[mpcnt[0][ii][l]];

	  j = 0;
	  mptmp = rtree->hlabest[i0]->mp[ii][l];
	
	  while(mptmp->next){
	    mptmp = mptmp->next;
	    if(!mptmp->edge && mptmp->fq->edge[ii])
	      continue;
	    if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
	      continue;      	    
	    fqlist[0][ii][l][j] = mptmp->fq;
	    mplist[0][ii][l][j] = mptmp;
	    j++;
	  }
	  mptmp = rtree->hlabest[i1]->mp[ii][l];
	    
	  while(mptmp->next){
	    mptmp = mptmp->next;
	    if(!mptmp->edge && mptmp->fq->edge[ii])
	      continue;
	    if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
	      continue;    
	    hit = false;
	    for(k=0;k<tmpmpcnt;k++){
	      if(fqlist[0][ii][l][k] == mptmp->fq){
		hit = true;
		break;
	      }
	    }
	    if(hit)
	      continue;
	    fqlist[0][ii][l][j] = mptmp->fq;
	    mplist[0][ii][l][j] = mptmp;
	    j++;
	  }
	  mpcnt[0][ii][l] = j;	 
	}

	mpcnt2[0][l] = 0;
	mptmp = rtree->hlabest[i0]->mp[0][l];

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

	    mpcnt2[0][l]++;
	  }
	  else if(mptmp->opaired[0]){
	    mpcnt2[0][l]++;
	  }
	}
	mptmp = rtree->hlabest[i0]->mp[1][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(mptmp->opaired[1]){
	    mpcnt2[0][l]++;
	  }
	}
	tmpmpcnt = mpcnt2[0][l];
	
	mptmp = rtree->hlabest[i1]->mp[0][l];

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

	    mpcnt2[0][l]++;
	  }
	  else if(mptmp->opaired[0]){
	    mpcnt2[0][l]++;
	  }
	}
	mptmp = rtree->hlabest[i1]->mp[1][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(mptmp->opaired[1]){
	    mpcnt2[0][l]++;
	  }
	}

	fqlist2[0][l] = new FASTQ_HASH*[mpcnt2[0][l]];
	mplist2[0][l] = new MPED_FQ*[mpcnt2[0][l]];
	
	j = 0;

	mptmp = rtree->hlabest[i0]->mp[0][l];

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

	    fqlist2[0][l][j] = mptmp->fq;
	    mplist2[0][l][j] = mptmp;
	    j++;
	  }
	  else if(mptmp->opaired[0]){
	    fqlist2[0][l][j] = mptmp->fq;
	    mplist2[0][l][j] = mptmp;
	    j++;
	  }
	}
	mptmp = rtree->hlabest[i0]->mp[1][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(mptmp->opaired[1]){
	    fqlist2[0][l][j] = mptmp->fq;
	    mplist2[0][l][j] = mptmp;
	    j++;
	  }
	}    

	mptmp = rtree->hlabest[i1]->mp[0][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(!mptmp->edge && mptmp->fq->edge[0])
	    continue;
      	  hit = false;
	  for(k=0;k<tmpmpcnt;k++){
	    if(fqlist2[0][l][k] == mptmp->fq){
	      hit = true;
	      break;
	    }
	  }
	  if(hit)
	    continue;
	  
	  mptmp2 = mptmp->pair;
	  if(mptmp2){
	    if(!mptmp2->edge && mptmp2->fq->edge[1])
	      continue;
	    if(!mptmp->paired)
	      continue;

	    fqlist2[0][l][j] = mptmp->fq;
	    mplist2[0][l][j] = mptmp;
	    j++;
	  }
	  else if(mptmp->opaired[0]){
	    fqlist2[0][l][j] = mptmp->fq;
	    mplist2[0][l][j] = mptmp;
	    j++;
	  }
	}
	mptmp = rtree->hlabest[i1]->mp[1][l];

	while(mptmp->next){
	  mptmp = mptmp->next;
	  if(mptmp->opaired[1]){
	    hit = false;
	    for(k=0;k<tmpmpcnt;k++){
	      if(fqlist2[0][l][k] == mptmp->fq){
		hit = true;
		break;
	      }
	    }
	    if(hit)
	      continue;

	    fqlist2[0][l][j] = mptmp->fq;
	    mplist2[0][l][j] = mptmp;
	    j++;
	  }
	}   
	mpcnt2[0][l] = j;
      }
      
      for(t=0;t<rtree->bestcnt-1;t++){
	for(s=t+1;s<rtree->bestcnt;s++){
	  if(i0 == t && i1 == s)
	    continue;
	  if(!incl[t] || !incl[s])
	    continue;

	  if(comatrix[t][s]/comatrix[i0][i1] < fuzzyth)
	    continue;
	 
	  for(l=0;l<lcnt;l++){
	    lsame[l] = false;
	    if(rtree->used[l] == 0)
	      continue;

	    for(ii=0;ii<2;ii++)
	      hit2[ii] = false;
	    if(rtree->hlabest[i0]->len[l] == rtree->hlabest[t]->len[l]){
	      hit2[0] = true;
	      tmplen = rtree->hlabest[i0]->len[l];
	      for(j=0;j<tmplen;j++){
		if(rtree->hlabest[i0]->code[l][j] != rtree->hlabest[t]->code[l][j]){
		  hit2[0] = false;
		  break;
		}	    
	      }
	    }
	    if(hit2[0]){
	      if(rtree->hlabest[i1]->len[l] == rtree->hlabest[s]->len[l]){
		hit2[1] = true;
		tmplen = rtree->hlabest[i1]->len[l];
		for(j=0;j<tmplen;j++){
		  if(rtree->hlabest[i1]->code[l][j] != rtree->hlabest[s]->code[l][j]){
		    hit2[1] = false;
		    break;
		  }	    
		}
	      }
	    }
	    if(!hit2[0] || !hit2[1]){
	      hit2[0] = false; hit2[1] = false;
	      if(rtree->hlabest[i0]->len[l] == rtree->hlabest[s]->len[l]){
		hit2[0] = true;
		tmplen = rtree->hlabest[i0]->len[l];
		for(j=0;j<tmplen;j++){
		  if(rtree->hlabest[i0]->code[l][j] != rtree->hlabest[s]->code[l][j]){
		    hit2[0] = false;
		    break;
		  }	    
		}
	      }
	      if(hit2[0]){
		if(rtree->hlabest[i1]->len[l] == rtree->hlabest[t]->len[l]){
		  hit2[1] = true;
		  tmplen = rtree->hlabest[i1]->len[l];
		  for(j=0;j<tmplen;j++){
		    if(rtree->hlabest[i1]->code[l][j] != rtree->hlabest[t]->code[l][j]){
		      hit2[1] = false;
		      break;
		    }	    
		  }
		}
	      }
	    }
	    if(hit2[0] && hit2[1]){
	      lsame[l] = true;
	      continue;
	    }
	     
	  }
	  
	  for(l=0;l<lcnt;l++){
	    if(rtree->used[l] == 0)
	      continue;
	    
	    for(ii=0;ii<2;ii++){
	      mpcnt[1][ii][l] = 0;
	      
	      mptmp = rtree->hlabest[t]->mp[ii][l];
	 
	      while(mptmp->next){
		mptmp = mptmp->next;	    
		if(!mptmp->edge && mptmp->fq->edge[ii])
		  continue;	
		if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
		  continue;      
		mpcnt[1][ii][l]++;
	      }
	      tmpmpcnt = mpcnt[1][ii][l];

	      mptmp = rtree->hlabest[s]->mp[ii][l];
	
	      while(mptmp->next){
		mptmp = mptmp->next;	    
		if(!mptmp->edge && mptmp->fq->edge[ii])
		  continue;	
		if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
		  continue;      
		mpcnt[1][ii][l]++;
	      }
	      fqlist[1][ii][l] = new FASTQ_HASH*[mpcnt[1][ii][l]];
	      mplist[1][ii][l] = new MPED_FQ*[mpcnt[1][ii][l]];

	      j = 0;

	      mptmp = rtree->hlabest[t]->mp[ii][l];
	     
	      while(mptmp->next){
		mptmp = mptmp->next;
		if(!mptmp->edge && mptmp->fq->edge[ii])
		  continue;
		if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
		  continue;      	    
		fqlist[1][ii][l][j] = mptmp->fq;
		mplist[1][ii][l][j] = mptmp;
		j++;
	      }

	      mptmp = rtree->hlabest[s]->mp[ii][l];
	   
	      while(mptmp->next){
		mptmp = mptmp->next;
		if(!mptmp->edge && mptmp->fq->edge[ii])
		  continue;
		if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
		  continue;    
		hit = false;
		for(k=0;k<tmpmpcnt;k++){
		  if(fqlist[1][ii][l][k] == mptmp->fq){
		    hit = true;
		    break;
		  }
		}
		if(hit)
		  continue;
		fqlist[1][ii][l][j] = mptmp->fq;
		mplist[1][ii][l][j] = mptmp;
		j++;
	      }
	      mpcnt[1][ii][l] = j;
	    }

	    mpcnt2[1][l] = 0;

	    mptmp = rtree->hlabest[t]->mp[0][l];
	   
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

		mpcnt2[1][l]++;
	      }
	      else if(mptmp->opaired[0]){
		mpcnt2[1][l]++;
	      }
	    }
	    mptmp = rtree->hlabest[t]->mp[1][l];
	 
	    while(mptmp->next){
	      mptmp = mptmp->next;
	      if(mptmp->opaired[1]){
		mpcnt2[1][l]++;
	      }
	    }
	    tmpmpcnt = mpcnt2[1][l];

	    mptmp = rtree->hlabest[s]->mp[0][l];

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

		mpcnt2[1][l]++;
	      }
	      else if(mptmp->opaired[0]){
		mpcnt2[1][l]++;
	      }
	    }

	    mptmp = rtree->hlabest[s]->mp[1][l];

	    while(mptmp->next){
	      mptmp = mptmp->next;
	      if(mptmp->opaired[1]){
		mpcnt2[1][l]++;
	      }
	    }
	    fqlist2[1][l] = new FASTQ_HASH*[mpcnt2[1][l]];
	    mplist2[1][l] = new MPED_FQ*[mpcnt2[1][l]];
	
	    j = 0;
	    
	    mptmp = rtree->hlabest[t]->mp[0][l];

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

		fqlist2[1][l][j] = mptmp->fq;
		mplist2[1][l][j] = mptmp;
		j++;
	      }
	      else if(mptmp->opaired[0]){
		fqlist2[1][l][j] = mptmp->fq;
		mplist2[1][l][j] = mptmp;
		j++;
	      }
	    }
	    mptmp = rtree->hlabest[t]->mp[1][l];

	    while(mptmp->next){
	      mptmp = mptmp->next;
	      if(mptmp->opaired[1]){
		fqlist2[1][l][j] = mptmp->fq;
		mplist2[1][l][j] = mptmp;
		j++;
	      }
	    }    

	    mptmp = rtree->hlabest[s]->mp[0][l];

	    while(mptmp->next){
	      mptmp = mptmp->next;
	      if(!mptmp->edge && mptmp->fq->edge[0])
		continue;
	      hit = false;
	      for(k=0;k<tmpmpcnt;k++){
		if(fqlist2[1][l][k] == mptmp->fq){
		  hit = true;
		  break;
		}
	      }
	      if(hit)
		continue;
	  
	      mptmp2 = mptmp->pair;
	      if(mptmp2){
		if(!mptmp2->edge && mptmp2->fq->edge[1])
		  continue;
		if(!mptmp->paired)
		  continue;

		fqlist2[1][l][j] = mptmp->fq;
		mplist2[1][l][j] = mptmp;
		j++;
	      }
	      else if(mptmp->opaired[0]){
		fqlist2[1][l][j] = mptmp->fq;
		mplist2[1][l][j] = mptmp;
		j++;
	      }
	    }

	    mptmp = rtree->hlabest[s]->mp[1][l];
	
	    while(mptmp->next){
	      mptmp = mptmp->next;
	      if(mptmp->opaired[1]){
		hit = false;
		for(k=0;k<tmpmpcnt;k++){
		  if(fqlist2[1][l][k] == mptmp->fq){
		    hit = true;
		    break;
		  }
		}
		if(hit)
		  continue;

		fqlist2[1][l][j] = mptmp->fq;
		mplist2[1][l][j] = mptmp;
		j++;
	      }
	    }  
	    mpcnt2[1][l] = j;
	  }

	  for(l=0;l<lcnt;l++){
	    if(rtree->used[l] == 0 || lsame[l])
	      continue;
	    for(ii=0;ii<2;ii++){
	      fqhit[ii][l] = new bool[mpcnt[1][ii][l]];
	      for(j=0;j<mpcnt[1][ii][l];j++)
		fqhit[ii][l][j] = false;
	    }
	    fqhit2[l] = new bool[mpcnt2[1][l]];
	    for(j=0;j<mpcnt2[1][l];j++)
	      fqhit2[l][j] = false;
	  }
	  hitcnt1 = 0; hitcnt2 = 0; hitcnt3 = 0;
	  for(l=0;l<lcnt;l++){
	    if(rtree->used[l] == 0 || lsame[l])
	      continue;
	    for(ii=0;ii<2;ii++){
	      for(j=0;j<mpcnt[0][ii][l];j++){
		mptmp = mplist[0][ii][l][j];
		fqtmp = fqlist[0][ii][l][j];
		hit = false;
		for(k=0;k<mpcnt[1][ii][l];k++){
		  if(fqtmp == fqlist[1][ii][l][k]){
		    fqhit[ii][l][k] = true;
		    hit = true;
		  }
		}	

		if(!hit){
		  hitcnt1 += (int)(fqlist[0][ii][l][j]->w[ii]*(double)(fqlist[0][ii][l][j]->ts[ii]));	 
		}     
		else{
		  hitcnt3 += (int)(fqlist[0][ii][l][j]->w[ii]*(double)(fqlist[0][ii][l][j]->ts[ii]));
		}
	      }
	    }
	  

	    for(j=0;j<mpcnt2[0][l];j++){
	      mptmp = mplist2[0][l][j];
	      fqtmp = fqlist2[0][l][j];
	      hit = false;
	      for(k=0;k<mpcnt2[1][l];k++){
		if(fqtmp == fqlist2[1][l][k]){
		  fqhit2[l][k] = true;
		  hit = true;
		}
	      }
	      if(mptmp->opaired[0]){
		w0 = fqlist2[0][l][j]->w[0];
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		if(!hit)
		  hitcnt1 += (int)(w0*(double)(tmpe[0]-tmps[0]+1));
		else
		  hitcnt3 += (int)(w0*(double)(tmpe[0]-tmps[0]+1));
	      }
	      else if(mptmp->opaired[1]){
		w1 = fqlist2[0][l][j]->w[1];
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		if(!hit)
		  hitcnt1 += (int)(w1*(double)(tmpe[0]-tmps[0]+1));		    
		else
		  hitcnt3 += (int)(w1*(double)(tmpe[0]-tmps[0]+1));		    
	      }
	      else{
		w0 = fqlist2[0][l][j]->w[0];
		w1 = fqlist2[0][l][j]->w[1];
		
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		tmps[1]=mptmp->pair->s-mptmp->pair->NS;
		tmpe[1]=mptmp->pair->e+mptmp->pair->NE;
		
		if(tmpe[0]<tmps[1] || tmpe[1]<tmps[0]){
		  if(!hit)
		    hitcnt1 += (int)(w1*(double)(tmpe[1]-tmps[1]+1)) + (int)(w0*(double)(tmpe[0]-tmps[0]+1));
		  else
		    hitcnt3 += (int)(w1*(double)(tmpe[1]-tmps[1]+1)) + (int)(w0*(double)(tmpe[0]-tmps[0]+1));
		}
		else{	
		  if(tmps[0] < tmps[1]){
		    if(!hit)
		      hitcnt1 += (int)(w0*(double)(abs(tmps[0]-tmps[1])));
		    else
		      hitcnt3 += (int)(w0*(double)(abs(tmps[0]-tmps[1])));
		  }
		  else{
		    if(!hit)
		      hitcnt1 += (int)(w1*(double)(abs(tmps[0]-tmps[1])));
		    else
		      hitcnt3 += (int)(w1*(double)(abs(tmps[0]-tmps[1])));
		  }
		  if(tmpe[0] > tmpe[1]){
		    if(!hit)
		      hitcnt1 += (int)(w0*(double)(abs(tmpe[0]-tmpe[1])));
		    else
		      hitcnt3 += (int)(w0*(double)(abs(tmpe[0]-tmpe[1])));
		  }
		  else{
		    if(!hit)
		      hitcnt1 += (int)(w1*(double)(abs(tmpe[0]-tmpe[1])));
		    else
		      hitcnt3 += (int)(w1*(double)(abs(tmpe[0]-tmpe[1])));
		  }
		}
	
	      }	 
	    }

	    for(ii=0;ii<2;ii++){
	      for(j=0;j<mpcnt[1][ii][l];j++){
		mptmp = mplist[1][ii][l][j];
		fqtmp = fqlist[1][ii][l][j];
		if(fqhit[ii][l][j])
		  continue;

		hitcnt2 += (int)(fqlist[1][ii][l][j]->w[ii]*(double)(fqlist[1][ii][l][j]->ts[ii]));	
	      }
	    }
	      	    
	    for(j=0;j<mpcnt2[1][l];j++){
	      mptmp = mplist2[1][l][j];
	      fqtmp = fqlist2[1][l][j];

	      if(fqhit2[l][j])
		continue;

	      if(mptmp->opaired[0]){
		w0 = fqlist2[1][l][j]->w[0];
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		hitcnt2 += (int)(w0*(double)(tmpe[0]-tmps[0]+1));
	      }
	      else if(mptmp->opaired[1]){
		w1 = fqlist2[1][l][j]->w[1];
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		hitcnt2 += (int)(w1*(double)(tmpe[0]-tmps[0]+1));		    
	      }
	      else{
		w0 = fqlist2[1][l][j]->w[0];
		w1 = fqlist2[1][l][j]->w[1];
		
		tmps[0]=mptmp->s-mptmp->NS;
		tmpe[0]=mptmp->e+mptmp->NE;
		tmps[1]=mptmp->pair->s-mptmp->pair->NS;
		tmpe[1]=mptmp->pair->e+mptmp->pair->NE;
		
		if(tmpe[0]<tmps[1] || tmpe[1]<tmps[0]){
		  hitcnt2 += (int)(w1*(double)(tmpe[1]-tmps[1]+1)) + (int)(w0*(double)(tmpe[0]-tmps[0]+1));
		}
		else{	
		  if(tmps[0] < tmps[1]){
		    hitcnt2 += (int)(w0*(double)(abs(tmps[0]-tmps[1])));
		  }
		  else{
		    hitcnt2 += (int)(w1*(double)(abs(tmps[0]-tmps[1])));
		  }
		  if(tmpe[0] > tmpe[1]){
		    hitcnt2 += (int)(w0*(double)(abs(tmpe[0]-tmpe[1])));
		  }
		  else{
		    hitcnt2 += (int)(w1*(double)(abs(tmpe[0]-tmpe[1])));
		  }
		}
	      }		   
	    }
	  }
	  
	  if((double)fuzzyth2*hitcnt1 < (double)hitcnt2){
	    hit = false;
	    for(j=0;j<tmpbestcnt;j++){
	      if((rtree->bestpair[j][0] == t && rtree->bestpair[j][1] == s) ||
		 (rtree->bestpair[j][0] == s && rtree->bestpair[j][1] == t)){
		hit = true;
		break;
	      }
	    }
	    if(!hit){
	      rtree->bestpair[tmpbestcnt][0]=t;
	      rtree->bestpair[tmpbestcnt][1]=s;	    
	      tmpbestcnt++;
	    }	  
	  }
	  else if((double)(hitcnt1-hitcnt2) < (double)fuzzyth3*hitcnt3){
	    hit = false;
	    for(j=0;j<tmpbestcnt;j++){
	      if((rtree->bestpair[j][0] == t && rtree->bestpair[j][1] == s) ||
		 (rtree->bestpair[j][0] == s && rtree->bestpair[j][1] == t)){
		hit = true;
		break;
	      }
	    }
	    if(!hit){
	      rtree->bestpair[tmpbestcnt][0]=t;
	      rtree->bestpair[tmpbestcnt][1]=s;	    
	      tmpbestcnt++;
	    }	
	  }	  	  
	  
	  for(l=0;l<lcnt;l++){
	    if(rtree->used[l] == 0 || lsame[l])
	      continue;
	    for(ii=0;ii<2;ii++){	      
	      delete[] fqlist[1][ii][l];
	      delete[] mplist[1][ii][l];
	      delete[] fqhit[ii][l];
	    }
	    delete[] fqlist2[1][l];
	    delete[] mplist2[1][l];
	    delete[] fqhit2[l];
	  }	  
	}
      }

      for(l=0;l<lcnt;l++){
	if(rtree->used[l] == 0)
	  continue;
	for(ii=0;ii<2;ii++){
	  delete[] fqlist[0][ii][l];
	  delete[] mplist[0][ii][l];
	}
	delete[] fqlist2[0][l];
	delete[] mplist2[0][l];
      }	  
    }
  }

  delete[] incl;    
  delete[] lsame;
  rtree->paircnt = tmpbestcnt;  

  for(i=0;i<rtree->bestcnt;i++)
    delete[] comatrix[i];
  delete[] comatrix;
  
}
		    

void Check_Tree(RANK_TREE *rtree,HLA_HASH **hh, List_Set *lset,int lcnt,int hcnt,int hlacnt){
  int i,j,k,l;
  int MAX_STACK = 10000;
  
  RANK_TREE *rtstack[MAX_STACK];
  int stcnt;
  RANK_TREE *stktree;
  HLA_HASH *htmp;
  ALIST *atmp,*atmp2;
  int ccnt = 0;
  int id1;
  bool hit1,hit2;
 
  struct ID_List{
    ALIST *alist[2];
    int bestucnt;
    ID_List *next;
    bool same;
  };

  ID_List *itop,*itmp,*itmp2;
  itop = new ID_List();
  itop->next = NULL;
  itmp = itop;

  rtstack[0] = rtree;
  stcnt = 1;
  int maxbestucnt = 0;
  string word;

  bool comp = true;

  while(stcnt > 0){
    stcnt--;
    stktree = rtstack[stcnt];

    if(!stktree->comp)
      comp = false;
    
    if(stktree->fin){
      itmp->next = new ID_List();
      itmp = itmp->next;
      itmp->next = NULL;
      itmp->same = false;
      for(k=0;k<2;k++)
	itmp->alist[k] = stktree->alist[k];
      itmp->bestucnt = stktree->bestucnt;
      if(maxbestucnt < itmp->bestucnt)
	maxbestucnt = itmp->bestucnt;
    } 
    else{
      for(i=0;i<stktree->ranksum;i++){
        rtstack[stcnt] = stktree->next[i];
	stcnt++;
      }
    }
  }

  bool hit;
  itmp = itop;
  while(itmp->next){
    itmp = itmp->next;
    hit = false;
    itmp2 = itop;
    while(itmp2->next){
      itmp2 = itmp2->next;
      if(itmp2==itmp)
	break;
      hit = false;
      for(k=0;k<2;k++){
	atmp = itmp->alist[k];
	while(atmp->next){
	  atmp = atmp->next;
	  hit = false;
	  for(j=0;j<2;j++){
	    atmp2 = itmp2->alist[j];
	    while(atmp2->next){
	      atmp2 = atmp2->next;
	      if(atmp->hla->aname == atmp2->hla->aname){
		hit = true;
		break;
	      }
	    }
	    if(hit)
	      break;
	  }
	  if(!hit)
	    break;
	}
	if(!hit)
	  break;
      }
      if(hit)
	break;
    }
    if(hit){
      cout << "Same allele pair is created." << endl;
      itmp->same = true;
    }
  }

  int pcnt = 0;
  int amcnt = 0;
  itmp = itop;
  while(itmp->next){
    itmp = itmp->next;
    if(itmp->same)
      continue;
    if(!itmp->alist[0]){
      cout << " :Allele is not recorded." << endl;
      exit(0);
    }
    cout << "Allele:" << pcnt+1 << "\t" << itmp->bestucnt << endl;
    for(i=0;i<2;i++){
      if(itmp->alist[i]){
	cout << i+1 << ":" << endl;
	atmp = itmp->alist[i];
	while(atmp->next){
	  atmp = atmp->next;
	  cout << atmp->hla->aname << endl;
	}
      }
    }
    if(itmp->bestucnt == maxbestucnt)
      pcnt++;
    amcnt++;
  }
  
  ofstream out;
  out.open(oname.c_str(),ios::out);
  ofstream outr;
  if(plotmreads){
    outr.open(plrname.c_str(),ios::out);    
  }

  if(pcnt == 0){
    out << "No candidate." << endl;    
  }
  else{
    out << "#Pair count\t" << amcnt << endl;
    out << "#Best allele pair\t" << pcnt << endl;
 
    itmp = itop;
    while(itmp->next){
      itmp = itmp->next;
      if(itmp->same)
	continue;
      if(itmp->bestucnt == maxbestucnt){
	atmp = itmp->alist[0];
	i = 0;
	if(plotmreads){
	  if(atmp->next){
	    Plot_Reads(outr,atmp->next->hla,lset,lcnt);
	  }
	}
	while(atmp->next){
	  atmp = atmp->next;
	  if(i==0){
	    out << atmp->hla->aname;
	  }
	  else{
	    out << "," << atmp->hla->aname;
	  }
	  i++;
	}

	out << "\t";
	if(!itmp->alist[1]->next){
	  out << "-";
	}
	else{
	  atmp = itmp->alist[1];

	  if(plotmreads){
	    if(atmp->next){
	      Plot_Reads(outr,atmp->next->hla,lset,lcnt);
	    }
	  }

	  i = 0;
	  while(atmp->next){
	    atmp = atmp->next;
	    if(i==0){
	      out << atmp->hla->aname;
	    }
	    else{
	      out << "," << atmp->hla->aname;
	    }
	    i++;
	  }
	}
	
	htmp = itmp->alist[0]->next->hla;
	j = 0;
	out << "\t";
	for(l=0;l<lcnt;l++){
	  if(lset->used[l]){
	    if(j==0)
	      out << lset->exname[l];
	    else
	      out << "," << lset->exname[l];
	    out << ":" << (double)htmp->rsum[l]/(htmp->ee[l]-htmp->es[l]+1);
	    if(htmp->mm[l] == 0){
	      out << ":comp.0";
	    }
	    else{
	      out << ":incomp." << htmp->mm[l];
	    }
	    j++;
	  }	  
	}


	if(!itmp->alist[1]->next){
	  out << "\t-";
	}
	else{
	  out << "\t";
	  htmp = itmp->alist[1]->next->hla;
	  j = 0;
	  for(l=0;l<lcnt;l++){
	    if(lset->used[l]){
	      if(j==0)
		out << lset->exname[l];
	      else
		out << "," << lset->exname[l];
	      out << ":" << (double)htmp->rsum[l]/(htmp->ee[l]-htmp->es[l]+1);
	      if(htmp->mm[l] == 0){
		out << ":comp.0";
	      }
	      else{
		out << ":incomp." << htmp->mm[l];
	      }
	      j++;
	    }	  
	  }
	}
	out << endl;
      }
    }

    if(amcnt-pcnt > 0){
      out << "#Other ambiguous pair\t" << amcnt-pcnt << endl;
      itmp = itop;
      while(itmp->next){
	itmp = itmp->next;
	if(itmp->same)
	  continue;
	if(itmp->bestucnt < maxbestucnt){
	  atmp = itmp->alist[0];
	  i = 0;
	  while(atmp->next){
	    atmp = atmp->next;
	    if(i==0){
	      out << atmp->hla->aname;
	    }
	    else{
	      out << "," << atmp->hla->aname;
	    }
	    i++;
	  }
	  out << "\t";
	  if(!itmp->alist[1]){
	    out << "--";
	  }
	  else{
	    atmp = itmp->alist[1];
	  i = 0;
	  while(atmp->next){
	    atmp = atmp->next;
	    if(i==0){
	      out << atmp->hla->aname;
	    }
	    else{
	      out << "," << atmp->hla->aname;
	    }
	    i++;
	  }
	  }
	  out << endl;
	}
      }
    }

    if(AMB_REDUCE_CNT > 0){
      out << "#Rejected ambiguous pair\t" << AMB_REDUCE_CNT << endl; 
    }
  }

  if(tallele.size() > 0){
    hit = false;
    for(i=0;i<hcnt;i++){
      htmp = hh[i];
      while(htmp->next){
	htmp = htmp->next;
	if(htmp->aname == tallele){
	  hit = true;
	  break;
	}
      }
      if(hit)
	break;
    }
    
    if(hit){
      Plot_Reads(outr,htmp,lset,lcnt);
    }

  }

  

  out.close();
  
  if(plotmreads)
    outr.close();
}

