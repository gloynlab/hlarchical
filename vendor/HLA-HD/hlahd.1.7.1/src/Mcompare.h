void Missmatch_Comparing(RANK_TREE *rtree,int lcnt){
  int i,j,k,l,s,t,u,ii,s2,t2;
  
  MPED_FQ *mptmp,*mptmp2;
  MPED_FQ **mplist[2][2],**mplist2[2];
  FASTQ_HASH **fqlist[2][2];
  FASTQ_HASH **fqlist2[2];
  FASTQ_HASH *fqtmp,*fqtmp2;
  int mpcnt[2][2],mpcnt2[2]; 
  bool hit;
  int tmps[2],tmpe[2];
  double *amiss = new double[rtree->paircnt];
  for(i=0;i<rtree->paircnt;i++)
    amiss[i] = 0;
  int bi[2];
  int es[2];

  cout << "Missmatch comparing" << endl;
  bool samel;
  int lsize,lsize2;

  bool single = false;

  for(i=0;i<rtree->paircnt;i++){
    bi[0] = rtree->bestpair[i][0];
    bi[1] = rtree->bestpair[i][1];
    if(bi[1] < 0){
      single = true;
      break;
    }
  }

  if(single){
    for(l=0;l<lcnt;l++){
      if(rtree->used[l] == 0)
	continue;
      lsize = rtree->hlabest[rtree->bestpair[0][0]]->ee[l] -
	rtree->hlabest[rtree->bestpair[0][0]]->es[l] + 1;
      
      samel = true;
      for(i=1;i<rtree->paircnt;i++){
	lsize2 = rtree->hlabest[rtree->bestpair[i][0]]->ee[l] -
	  rtree->hlabest[rtree->bestpair[i][0]]->es[l] + 1;	
	if(lsize != lsize2){
	  samel = false;
	  break;
	}
      }      
      if(!samel)
	continue;    

      for(i=0;i<rtree->paircnt;i++){
	bi[0] = rtree->bestpair[i][0];       
	amiss[i] += rtree->hlabest[bi[0]]->tmpmm;
      }
    }
  }
  else{
    for(l=0;l<lcnt;l++){
      if(rtree->used[l] == 0)
	continue;

      samel = true;
      lsize = rtree->hlabest[rtree->bestpair[0][0]]->ee[l] -
	rtree->hlabest[rtree->bestpair[0][0]]->es[l] + 1;
      lsize2 = rtree->hlabest[rtree->bestpair[0][1]]->ee[l] -
	rtree->hlabest[rtree->bestpair[0][1]]->es[l] + 1;
      if(lsize != lsize2)
	samel = false;
      if(!samel)
	continue;
    
      for(i=1;i<rtree->paircnt;i++){
	for(k=0;k<2;k++){
	  lsize2 = rtree->hlabest[rtree->bestpair[i][k]]->ee[l] -
	    rtree->hlabest[rtree->bestpair[i][k]]->es[l] + 1;	
	  if(lsize != lsize2){
	    samel = false;
	    break;
	  }
	}
      }
      if(!samel)
	continue;    
      
      bool **ehit[2][2];
      bool **ehit2[2][2];
    
      for(k=0;k<2;k++){
	for(ii=0;ii<2;ii++){
	  ehit[k][ii] = new bool*[rtree->paircnt];
	  ehit2[k][ii] = new bool*[rtree->paircnt];
	  for(i=0;i<rtree->paircnt;i++){	  
	    bi[k] = rtree->bestpair[i][k];
	    ehit[k][ii][i] = new bool[rtree->hlabest[bi[k]]->len[l]];
	    ehit2[k][ii][i] = new bool[rtree->hlabest[bi[k]]->len[l]];
	    for(j=0;j<rtree->hlabest[bi[k]]->len[l];j++){
	      ehit[k][ii][i][j] = false;
	      ehit2[k][ii][i][j] = false;
	    }
	  }
	}
      }

      for(i=0;i<rtree->paircnt;i++){
	bi[0] = rtree->bestpair[i][0];
	bi[1] = rtree->bestpair[i][1];
	for(k=0;k<2;k++){
	  for(ii=0;ii<2;ii++){	
	    mpcnt[k][ii] = 0;
	    
	    mptmp = rtree->hlabest[bi[k]]->mp[ii][l];
	    
	    while(mptmp->next){
	      mptmp = mptmp->next;	    
	      if(!mptmp->edge && mptmp->fq->edge[ii])
		continue;	
	      if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
		continue;      
	      mpcnt[k][ii]++;
	    }
	    fqlist[k][ii] = new FASTQ_HASH*[mpcnt[k][ii]];
	    mplist[k][ii] = new MPED_FQ*[mpcnt[k][ii]];

	    j = 0;
	    mptmp = rtree->hlabest[bi[k]]->mp[ii][l];
	   
	    while(mptmp->next){
	      mptmp = mptmp->next;
	      if(!mptmp->edge && mptmp->fq->edge[ii])
		continue;
	      if((!mptmp->paired && mptmp->fq->paired) && (!mptmp->opaired[ii] && mptmp->fq->paired))
		continue;      	    
	      fqlist[k][ii][j] = mptmp->fq;
	      mplist[k][ii][j] = mptmp;
	      j++;
	    }
	  }
	
	  mpcnt2[k] = 0;
	  mptmp = rtree->hlabest[bi[k]]->mp[0][l];
	  
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
	      
	      mpcnt2[k]++;
	    }
	    else if(mptmp->opaired[0]){
	      mpcnt2[k]++;
	    }
	  }
	  mptmp = rtree->hlabest[bi[k]]->mp[1][l];
	  
	  while(mptmp->next){
	    mptmp = mptmp->next;
	    if(mptmp->opaired[1]){
	      mpcnt2[k]++;
	    }
	  }

	  fqlist2[k] = new FASTQ_HASH*[mpcnt2[k]];
	  mplist2[k] = new MPED_FQ*[mpcnt2[k]];
	
	  j = 0;
	  
	  mptmp = rtree->hlabest[bi[k]]->mp[0][l];
	  
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
	      
	      fqlist2[k][j] = mptmp->fq;
	      mplist2[k][j] = mptmp;
	      j++;
	    }
	    else if(mptmp->opaired[0]){
	      fqlist2[k][j] = mptmp->fq;
	      mplist2[k][j] = mptmp;
	      j++;
	    }
	  }
	  mptmp = rtree->hlabest[bi[k]]->mp[1][l];

	  while(mptmp->next){
	    mptmp = mptmp->next;
	    if(mptmp->opaired[1]){
	      fqlist2[k][j] = mptmp->fq;
	      mplist2[k][j] = mptmp;
	      j++;
	    }
	  }    
	}
	
	for(ii=0;ii<2;ii++){
	  for(j=0;j<mpcnt[0][ii];j++){
	    if(fqlist[0][ii][j]->w[0] == 0)
	      continue;
	    hit = false;
	    for(k=0;k<mpcnt[1][ii];k++){
	      if(fqlist[0][ii][j] == fqlist[1][ii][k]){
		hit = true;
		break;
	      }
	    }	  	    
	    if(hit){
	      s = mplist[0][ii][j]->s;
	      t = mplist[0][ii][j]->e;
	      for(u=s;u<=t;u++)
		ehit2[0][1][i][u] = true;
	      s = mplist[1][ii][k]->s;
	      t = mplist[1][ii][k]->e;
	      for(u=s;u<=t;u++)
		ehit2[1][1][i][u] = true;
	      //cout << fqlist[0][ii][j]->id << "\t" << s << "\t" << t << "\t" << fqlist[0][ii][j]->w[0]
	      //   << "\t" << fqlist[0][ii][j]->w[1] << "\tDhit" << endl;
	    }
	    else{
	      s = mplist[0][ii][j]->s;
	      t = mplist[0][ii][j]->e;
	      for(u=s;u<=t;u++)
		ehit2[0][0][i][u] = true;
	      //cout << fqlist[0][ii][j]->id << "\t" << s << "\t" << t << "\t" << fqlist[0][ii][j]->w[0] << "\t" << fqlist[0][ii][j]->w[1] << "\t1hit" << endl;
	    }
	  }
	}
	for(ii=0;ii<2;ii++){
	  for(j=0;j<mpcnt[1][ii];j++){	  
	    if(fqlist[1][ii][j]->w[1] == 0)
	      continue;
	    hit = false;
	    for(k=0;k<mpcnt[0][ii];k++){
	      if(fqlist[1][ii][j] == fqlist[0][ii][k]){
		hit = true;
		break;
	      }
	    }
	    if(!hit){		      
	      s = mplist[1][ii][j]->s;
	      t = mplist[1][ii][j]->e;   
	      for(u=s;u<=t;u++)
		ehit2[1][0][i][u] = true;
	      //cout << fqlist[1][ii][j]->id << "\t" << s << "\t" << t << "\t" << fqlist[1][ii][j]->w[0] << "\t" << fqlist[1][ii][j]->w[1] << "\t2hit" << endl;	      	    
	    }
	  }
	}
	
	for(j=0;j<mpcnt2[0];j++){
	  hit = false;
	  if(fqlist2[0][j]->w[0] == 0 || fqlist2[0][j]->w[1] == 0)
	    continue;
	  for(k=0;k<mpcnt2[1];k++){
	    if(fqlist2[0][j] == fqlist2[1][k]){
	      hit = true;
	      break;
	    }
	  }	  
	  if(hit){
	    if(mplist2[0][j]->opaired[0]){
	      s = mplist2[0][j]->s;
	      t = mplist2[0][j]->e;
	      s2 = -1;
	    }
	    else if(mplist2[0][j]->opaired[1]){
	      s = mplist2[0][j]->s;
	      t = mplist2[0][j]->e;
	      s2 = -1;
	    }
	    else{
	      s = mplist2[0][j]->s;
	      t = mplist2[0][j]->e;
	      s2 = mplist2[0][j]->pair->s;
	      t2 = mplist2[0][j]->pair->e;
	    }	
	    for(u=s;u<=t;u++)
	      ehit2[0][1][i][u] = true;
	    if(s2 >= 0){
	      for(u=s2;u<=t2;u++)
		ehit2[0][1][i][u] = true;
	    }
	    
	    if(mplist2[1][k]->opaired[0]){
	      s = mplist2[1][k]->s;
	      t = mplist2[1][k]->e;
	      s2 = -1;
	    }
	    else if(mplist2[1][k]->opaired[1]){
	      s = mplist2[1][k]->s;
	      t = mplist2[1][k]->e;
	      s2 = -1;
	    }
	    else{
	      s = mplist2[1][k]->s;
	      t = mplist2[1][k]->e;
	      s2 = mplist2[1][k]->pair->s;
	      t2 = mplist2[1][k]->pair->e;
	    }	
	    for(u=s;u<=t;u++)
	      ehit2[1][1][i][u] = true;
	    if(s2 >= 0){
	      for(u=s2;u<=t2;u++)
		ehit2[1][1][i][u] = true;
	    }	    
	  }
	  else{
	    if(mplist2[0][j]->opaired[0]){
	      s = mplist2[0][j]->s;
	      t = mplist2[0][j]->e;
	      s2 = -1;
	    }
	    else if(mplist2[0][j]->opaired[1]){
	      s = mplist2[0][j]->s;
	      t = mplist2[0][j]->e;
	      s2 = -1;
	    }
	    else{
	      s = mplist2[0][j]->s;
	      t = mplist2[0][j]->e;
	      s2 = mplist2[0][j]->pair->s;
	      t2 = mplist2[0][j]->pair->e;
	    }	
	    for(u=s;u<=t;u++)
	      ehit2[0][0][i][u] = true;
	    if(s2 >= 0){
	      for(u=s2;u<=t2;u++)
		ehit2[0][0][i][u] = true;
	    }
	  }
	}
 
	for(j=0;j<mpcnt2[1];j++){
	  if(fqlist2[1][j]->w[0] == 0 || fqlist2[1][j]->w[1] == 0)
	    continue;
	  hit = false;
	  for(k=0;k<mpcnt2[0];k++){
	    if(fqlist2[1][j] == fqlist2[0][k]){
	      hit = true;
	      break;
	    }
	  }
	  
	  if(!hit){	 
	    if(mplist2[1][j]->opaired[0]){
	      s = mplist2[1][j]->s;
	      t = mplist2[1][j]->e;
	      s2 = -1;
	    }
	    else if(mplist2[1][j]->opaired[1]){
	      s = mplist2[1][j]->s;
	      t = mplist2[1][j]->e;
	      s2 = -1;
	    }
	    else{
	      s = mplist2[1][j]->s;
	      t = mplist2[1][j]->e;
	      s2 = mplist2[1][j]->pair->s;
	      t2 = mplist2[1][j]->pair->e;
	    }	
	    for(u=s;u<=t;u++)
	      ehit2[1][0][i][u] = true;
	    if(s2 >= 0){
	      for(u=s2;u<=t2;u++)
		ehit2[1][0][i][u] = true;
	    }
	  }
	}
	
	for(k=0;k<2;k++){
	  for(ii=0;ii<2;ii++){
	    for(j=rtree->hlabest[bi[k]]->es[l];j<=rtree->hlabest[bi[k]]->ee[l];j++){
	      //cout << ehit2[k][ii][i][j];
	    }
	    //cout << endl; 
	  }
	}
	
	for(k=0;k<2;k++){
	  for(ii=0;ii<2;ii++){
	    delete[] fqlist[k][ii];
	    delete[] mplist[k][ii];
	  }
	  delete[] fqlist2[k];
	  delete[] mplist2[k];
	} 
      }
            
      bool *hitpos = new bool[lsize];
      for(j=0;j<lsize;j++)
	hitpos[j] = false;
      
      for(i=0;i<rtree->paircnt;i++){
	es[0] = rtree->hlabest[rtree->bestpair[i][0]]->es[l];
	es[1] = rtree->hlabest[rtree->bestpair[i][1]]->es[l];
	for(j=0;j<lsize;j++){
	  if(ehit2[0][0][i][j+es[0]] && ehit2[1][0][i][j+es[1]])
	    hitpos[j] = true;
	}
      }
        
      for(i=0;i<rtree->paircnt;i++){
	es[0] = rtree->hlabest[rtree->bestpair[i][0]]->es[l];
	es[1] = rtree->hlabest[rtree->bestpair[i][1]]->es[l];
	for(j=0;j<lsize;j++){
	  if(hitpos[j]){
	    if((!ehit2[0][0][i][j+es[0]] && !ehit2[0][1][i][j+es[0]])||
	       (!ehit2[1][0][i][j+es[1]] && !ehit2[1][1][i][j+es[1]]))
	    amiss[i]++;
	  }
	}
      }

      for(k=0;k<2;k++){
	for(ii=0;ii<2;ii++){
	  for(t=0;t<rtree->paircnt;t++){
	    delete[] ehit[k][ii][t];
	    delete[] ehit2[k][ii][t];
	  }
	  delete[] ehit[k][ii];
	  delete[] ehit2[k][ii];
	}
      }
      delete[] hitpos;    
    }
  }

  int minmiss = 10000000;
  for(i=0;i<rtree->paircnt;i++){
    if(rtree->bestpair[i][1] >= 0)
      cout << rtree->hlabest[rtree->bestpair[i][0]]->aname << "\t" << rtree->hlabest[rtree->bestpair[i][1]]->aname << "\t" << amiss[i] << endl;
    else
      cout << rtree->hlabest[rtree->bestpair[i][0]]->aname << "\t-" 
    	    << amiss[i] << endl;

    if(amiss[i] < minmiss)
      minmiss = amiss[i];
  }
  int tmpbestcnt = 0;
  
  if(minmiss <= 0.0000001){
    for(i=0;i<rtree->paircnt;i++){
      if(amiss[i] <= 0.0000001){
	rtree->bestpair[tmpbestcnt][0] = rtree->bestpair[i][0];
	rtree->bestpair[tmpbestcnt][1] = rtree->bestpair[i][1];
	tmpbestcnt++;
      }
    }
  }
  else{
    for(i=0;i<rtree->paircnt;i++){
      rtree->bestpair[tmpbestcnt][0] = rtree->bestpair[i][0];
      rtree->bestpair[tmpbestcnt][1] = rtree->bestpair[i][1];
      tmpbestcnt++;
    }  
  }
  rtree->paircnt = tmpbestcnt;

  delete[] amiss;
}
