void Create_Rank_Tree(RANK_TREE *rtree,HLA_HASH **hh,int lcnt,int hcnt,int hlacnt,int **lmode,int *maxlm,List_Set *lset){
  int i,j,k,s,l,n;
  HLA_HASH *htmp;
  ALIST *atmp,*atmp2;
  string word,word2;
  int DEPTH_MAX = 100;
  int maxl;
  int usedcnt;

  cout << "Depth:" << rtree->depth << endl;   
  if(rtree->depth > DEPTH_MAX){
    cout << "Not consistent." << endl;
    return;    
  }
 
  usedcnt = 0;
  for(i=0;i<lcnt;i++){
    if(rtree->used[i]){
      cout << lset->exname[i] << endl;
      usedcnt++;
    }
  }
  maxl = 0;
  for(i=0;i<2;i++){
    cout << "Allele List:" << i+1 << endl;
    atmp = rtree->alist[i];
    while(atmp->next){
      atmp = atmp->next;
      cout << atmp->hla->aname << "\t";
      j = 0;
      for(l=0;l<lcnt;l++){
	cout << atmp->hla->hit[l];
	if(atmp->hla->hit[l])
	  j++;
      }
      if(j > maxl)
	maxl = j;
      cout << endl;
    }
  }

  clock_t start, end;
  start = clock();          
  Calc_Rank_All(rtree,hh,lcnt,hcnt,lmode,maxlm);
  end = clock();
  cout << "Calc Rank:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl; 
  
  start = clock();          
  Sort_Only_Intence_All(rtree,lcnt);
  end = clock();
  cout << "Sort list:" << (double)(end-start)/CLOCKS_PER_SEC << "sec"  << endl;
 
  if(rtree->hithla > 0){
    Count_Diffcnt(rtree,lcnt);
    ReSort_Best(rtree);    
  }

  if(rtree->bestcnt == 0){
    rtree->covth = 0.5;
    Calc_Rank_All(rtree,hh,lcnt,hcnt,lmode,maxlm);
    Sort_Only_Intence_All(rtree,lcnt);
    
    if(rtree->hithla > 0){
      Count_Diffcnt(rtree,lcnt);
      ReSort_Best(rtree);    
    }
  }
  
  int *readcntall,*readcntall2;
  bool hit;
  int bcnt;
  
  rtree->paircnt = 0;
  if(rtree->bestcnt > 0){
    rtree->readcnt = new double[rtree->bestcnt];
    rtree->readcnt2 = new double[rtree->bestcnt];
    rtree->readcnt3 = new double[rtree->bestcnt];
    for(i=0;i<rtree->bestcnt;i++){
      rtree->readcnt[i] = rtree->hlabest[i]->tmpvcnt;
      rtree->readcnt2[i] = rtree->hlabest[i]->tmpv2cnt;
      rtree->readcnt3[i] = rtree->hlabest[i]->tmpnm;  
    }
    bcnt = rtree->bestcnt*rtree->bestcnt;
    rtree->bestpair = new int*[bcnt];
    for(i=0;i<bcnt;i++){
      rtree->bestpair[i] = new int[2];
    }     
    Select_Best_Allele(rtree,lcnt);
    if(rtree->paircnt >= 2){
      Missmatch_Comparing(rtree,lcnt);
    }
    if(rtree->paircnt >= 2){
      Re_Select_Best(rtree,lcnt);
    }
  }  
 
  if(rtree->bestcnt == 0)
    return;

  if(rtree->depth == 1 && rtree->paircnt >= FIRST_AMB_MAX)
    return;

  for(i=0;i<rtree->bestcnt;i++){  
    cout << rtree->hlabest[i]->id << "\t" << rtree->hlabest[i]->aname 
	 << "\t" << rtree->hlabest[i]->tmpmm
	 << "\t" << rtree->hlabest[i]->tmpnm 
	 << "\t" << rtree->hlabest[i]->tmpmmp << "\t" 
	 << rtree->hlabest[i]->tmprsum << "\t" << rtree->hlabest[i]->tmpnmp
	 << "\t" << rtree->hlabest[i]->tmpvcnt << endl;
  }
  cout << endl;
        
  for(i=0;i<rtree->bestcnt;i++)
    cout << "\t" << rtree->hlabest[i]->tmpvcnt;
  cout << endl;
  for(i=0;i<rtree->bestcnt;i++){
    cout << i;
    for(j=0;j<rtree->bestcnt;j++){
      cout << "\t" << rtree->diffcnt[i][j];
    }
    cout << endl;
  }
  cout << endl;

  for(i=0;i<rtree->bestcnt;i++)
    cout << "\t" << rtree->hlabest[i]->tmpv2cnt;
  cout << endl;
  for(i=0;i<rtree->bestcnt;i++){
    cout << i;
    for(j=0;j<rtree->bestcnt;j++){
      cout << "\t" << rtree->diffcnt2[i][j];
    }
    cout << endl;
  }
  cout << endl;

  if((rtree->depth == 1 && rtree->bestcnt == 0 )|| (rtree->depth == 1 && rtree->paircnt >= FIRST_AMB_MAX)){
    return;
  }
    
  for(i=0;i<rtree->paircnt;i++){
    cout << "Pair" << i+1;      
    for(k=0;k<2;k++){
      if(rtree->bestpair[i][k] >= 0){
	cout << "\t" << rtree->bestpair[i][k];
      }
      else{
	cout << "\t--" << endl;
      }
    }
    cout << endl;
  }

  int *anew[2];
  for(i=0;i<2;i++){
    rtree->rankid[i] = new int[rtree->paircnt];
    anew[i] = new int[rtree->paircnt];
    for(j=0;j<rtree->paircnt;j++)
      anew[i][j] = -1;
  }
  rtree->ranksum = 0;
  bool hits[2];
  int hitsk[2];
  int tmprank[2]; 

  bool afin[2];
  int hitcnt;
  for(s=0;s<2;s++){
    hit = false;
    atmp = rtree->alist[s];
    while(atmp->next){
      atmp = atmp->next;
      hit = true;
      k = 0;
      hitcnt = 0;
      for(i=0;i<lcnt;i++){
	if(rtree->used[i]){
	  k++;
	  if(rtree->used[i] == atmp->hla->hit[i]){
	    hitcnt++;
	  }
	}
      }
      if(k <= hitcnt){
	hit = false;
	break;
      }      
    }
    afin[s] = hit;
  }
  cout << "Fin:" << afin[0] << "\t" << afin[1] << endl;
  
  for(j=0;j<rtree->paircnt;j++){
    hitsk[0] = 0; hitsk[1] = 1;
    for(k=0;k<2;k++){
      hits[k] = false;
      if(rtree->bestpair[j][k] < 0){
	if(rtree->alist[k]->next == NULL){
	  hits[k] = true;
	  tmprank[k] = -1;
	  hitsk[k] = 1;
	  continue;
	}
      }
      for(i=0;i<rtree->bestcnt;i++){
	if(rtree->bestpair[j][k] == i){
	  for(s=0;s<2;s++){
	    if(afin[s])
	      continue;
	    atmp = rtree->alist[s];
	    if(atmp->all){
	      cout << "Best:" << i << endl;
	      hits[k] = true;
	      tmprank[k] = i;	 
	      break;
	    }	  
	    for(l=0;l<rtree->hithla;l++){
	      if(rtree->sameid[l] == i){	    		
		atmp = rtree->alist[s];
		if(atmp->next){
		  while(atmp->next){
		    atmp = atmp->next;
		    if(rtree->hrank[l]->id == atmp->hla->id){
		      break;
		    }
		  }
		  if(rtree->hrank[l]->id == atmp->hla->id){
		    hits[k] = true;
		    tmprank[k] = i;
		    hitsk[k] = s;
		    break;
		  }
		}
	      }
	      if(hits[k])
		break;
	    }
	    if(hits[k])
	      break;
	  }
	}
      }
    }

    if(hits[0] && hits[1]){
      if(rtree->alistcnt[0] > 0 && rtree->alistcnt[1] > 0)
	if(hitsk[0] == hitsk[1])
	  continue;
      cout << "Best:" << j << endl;
      rtree->rankid[0][rtree->ranksum] = tmprank[0];
      rtree->rankid[1][rtree->ranksum] = tmprank[1];
      anew[0][rtree->ranksum] = hitsk[0];
      anew[1][rtree->ranksum] = hitsk[1];
      rtree->ranksum++;      
    }
    else if(hits[0] || hits[1]){
      if(!rtree->alist[1]->next){
	hit = true;
	for(i=0;i<rtree->ranksum;i++){	  
	  if(hits[0]){
	    if(rtree->rankid[0][i] == tmprank[0]){
	      hit = false;
	      break;
	    }
	  }
	  else{
	    if(rtree->rankid[1][i] == tmprank[1]){
	      hit = false;
	      break;
	    }
	  }
	}     	
	if(hit){
	  if(hits[0]){
	    rtree->rankid[0][rtree->ranksum] = tmprank[0];
	    rtree->rankid[1][rtree->ranksum] = rtree->bestpair[j][1-hitsk[0]];
	    anew[0][rtree->ranksum] = hitsk[0];
	    anew[1][rtree->ranksum] = -1;
	  }
	  if(hits[1]){
	    rtree->rankid[0][rtree->ranksum] = rtree->bestpair[j][1-hitsk[1]];
	    rtree->rankid[1][rtree->ranksum] = tmprank[1];
	    anew[0][rtree->ranksum] = -1;
	    anew[1][rtree->ranksum] = hitsk[1];
	  }
	  cout << rtree->ranksum << "\tANew:" << rtree->rankid[0][rtree->ranksum]
	       << "\t" << anew[0][rtree->ranksum]
	       << "\t" << rtree->rankid[1][rtree->ranksum] << "\t" 
	       <<   anew[1][rtree->ranksum] << endl;
	  rtree->ranksum++;      
	}
      }
      else{
	if(hits[0])
	  s = 1-hitsk[0];
	else
	  s = 1-hitsk[1];
	if(afin[s]){
	  hit = true;
	  for(i=0;i<rtree->ranksum;i++){	  
	    if(hits[0]){
	      if(rtree->rankid[0][i] == tmprank[0]){
		hit = false;
		break;
	      }
	    }
	    else{
	      if(rtree->rankid[1][i] == tmprank[1]){
		hit = false;
		break;
	      }
	    }
	  }     
	  if(hit){
	    rtree->rankid[0][rtree->ranksum] = tmprank[0];
	    rtree->rankid[1][rtree->ranksum] = tmprank[1];
	    if(hits[0]){
	      anew[0][rtree->ranksum] = hitsk[0];
	      anew[1][rtree->ranksum] = 1-hitsk[0];
	    }
	    if(hits[1]){
	      anew[0][rtree->ranksum] = 1-hitsk[1];
	      anew[1][rtree->ranksum] = hitsk[1];
	    }
	    rtree->ranksum++;      
	  }
	}
      }
    }
  }    

  if(rtree->ranksum == 0 && maxl == usedcnt){
    for(j=0;j<rtree->paircnt;j++){
      hitsk[0] = 0; hitsk[1] = 1;
      for(k=0;k<2;k++){
	hits[k] = false;
	if(rtree->bestpair[j][k] < 0){
	  if(rtree->alist[k]->next == NULL){
	    hits[k] = true;
	    tmprank[k] = -1;
	    hitsk[k] = 1;
	    continue;
	  }
	}
	for(i=0;i<rtree->bestcnt;i++){
	  if(rtree->bestpair[j][k] == i){
	    for(s=0;s<2;s++){
	      if(afin[s])
		continue;
	      atmp = rtree->alist[s];
	      if(atmp->all){
		cout << "Best:" << i << endl;
		hits[k] = true;
		tmprank[k] = i;	 
		break;
	      }	  
	      for(l=0;l<rtree->hithla;l++){
		if(rtree->sameid[l] == i){	    		
		  atmp = rtree->alist[s];
		  if(atmp->next){
		    while(atmp->next){
		      atmp = atmp->next;
		      if(rtree->hrank[l]->id == atmp->hla->id){
			break;
		      }
		    }
		    if(rtree->hrank[l]->id == atmp->hla->id){
		      hits[k] = true;
		      tmprank[k] = i;
		      hitsk[k] = s;
		      break;
		    }
		  }
		}
		if(hits[k])
		  break;
	      }
	      if(hits[k])
		break;
	    }
	  }
	}
      }
      
      if(hits[0] || hits[1]){
	if(hits[0])
	  afin[1-hitsk[0]] = true;
	else
	  afin[1-hitsk[1]] = true;
	hit = true;	
	for(i=0;i<rtree->ranksum;i++){	  
	  if(hits[0]){
	    if(rtree->rankid[0][i] == tmprank[0]){
	      hit = false;
	      break;
	    }
	  }
	  else{
	    if(rtree->rankid[1][i] == tmprank[1]){
	      hit = false;
	      break;
	    }
	  }
	}     
	if(hit){
	  rtree->rankid[0][rtree->ranksum] = tmprank[0];
	  rtree->rankid[1][rtree->ranksum] = tmprank[1];
	  if(hits[0]){
	    anew[0][rtree->ranksum] = hitsk[0];
	    anew[1][rtree->ranksum] = 1-hitsk[0];
	  }
	  if(hits[1]){
	    anew[0][rtree->ranksum] = 1-hitsk[1];
	    anew[1][rtree->ranksum] = hitsk[1];
	  }
	  rtree->ranksum++;      
	}
      }      
    }
  }
    
  cout << "Pair Num:" << rtree->paircnt << endl;
  cout << "Rank Num:" << rtree->ranksum << endl;

  bool reuse = false;//[2];
  //for(i=0;i<2;i++)
  // reuse[i] = false;
  int id1,id2;

  int *tmpbucnt;
  if(rtree->ranksum > 0){
    tmpbucnt = new int[rtree->ranksum];
    for(i=0;i<rtree->ranksum;i++)
      tmpbucnt[i] = usedcnt;
  }
  else{    
    rtree->ranksum = 1;
    tmpbucnt = new int[rtree->ranksum];
    tmpbucnt[0] = rtree->bestucnt;
    reuse = true;
    anew[0][0] = -1;
    anew[1][0] = -1;
  }    
  int ccnt=rtree->ranksum;
  bool *complist = new bool[ccnt];
  bool win[2];
     
  rtree->next = new RANK_TREE*[rtree->ranksum];    
  for(i=0;i<rtree->ranksum;i++){
    rtree->next[i] = NULL;
  }  
         
  RANK_TREE *rtmp;
  for(i=0;i<rtree->ranksum;i++){
    cout << "Number:" << i << endl;
    rtree->next[i] = new RANK_TREE();
    rtmp = rtree->next[i];
    rtmp->comp = rtree->comp;
    rtmp->next = NULL;
 
    int *tmpused = new int[lcnt];
    int *tmpfin = new int[lcnt];

    int fincnt=0;
    for(j=0;j<lcnt;j++){
      if(rtree->finl[j])
	fincnt++;
    }
 
    if(reuse){
      for(j=0;j<lcnt;j++){
	if(rtree->finl[j]==1){
	  tmpfin[j] = 1;
	}
	else{
	  tmpfin[j] = 0;
	}
	if(rtree->used[j]==1)
	  tmpfin[j] = 1;
	tmpused[j] = 0;
      }	
    }
    else{
      for(j=0;j<lcnt;j++){
	tmpfin[j] = rtree->finl[j];
	if(rtree->used[j]==1)
	  tmpfin[j] = 1;
	tmpused[j] = 0;
      }
    }
    
    for(k=0;k<2;k++){
      if(rtree->alist[k]->all){
	for(j=0;j<hcnt;j++){
	  htmp = hh[j];
	  while(htmp->next){
	    htmp = htmp->next;
	    for(l=0;l<lcnt;l++){
	      if(htmp->hit[l]){
		tmpused[l] = 1;
	      }
	    }
	  }
	}
      }
      else{
	atmp = rtree->alist[k];
	while(atmp->next){
	  atmp = atmp->next;
	  for(l=0;l<lcnt;l++){
	    if(atmp->hla->hit[l]){
	      tmpused[l] = 1;
	    }            
	  }
	}
      }
    }
  
    maxl = 0;
    for(j=0;j<lcnt;j++){
      if(tmpused[j])
	maxl++;
    }

    for(k=0;k<2;k++){      
      rtmp->alist[k] = new ALIST();
      rtmp->alistcnt[k] = 0;
      rtmp->alist[k]->all = false;
      
      atmp = rtmp->alist[k];
      atmp->next = NULL;
      
      if(reuse){
	if(afin[k]){
	  atmp2 = rtree->alist[k];
	  while(atmp2->next){
	    atmp2 = atmp2->next;
	    atmp->next = new ALIST();
	    atmp = atmp->next;
	    atmp->next = NULL;
	    atmp->hla = atmp2->hla;
	    rtmp->alistcnt[k]++;
	  }
	  continue;
	}

	if(rtree->alist[k]->all){
	  atmp->all = true;
	  for(j=0;j<hcnt;j++){
	    htmp = hh[j];
	    while(htmp->next){
	      htmp = htmp->next;
	      bool finhit = true;
	      for(l=0;l<lcnt;l++){
		if(htmp->hit[l] && !tmpfin[l]){
		  finhit = false;
		  break;
		}
	      }
	      if(!finhit){
		rtmp->alistcnt[k]++;
		for(l=0;l<lcnt;l++){
		  if(!htmp->hit[l])
		    tmpused[l] = 0;
		}
	      }
	    }
	  }
	}
	else{	   
	  cout << "Reuse check" << endl;	
	  for(l=0;l<rtree->paircnt;l++){
	    for(s=0;s<2;s++){
	      cout << rtree->bestpair[l][s] << " ";
	    }
	    cout << endl;
	  }

	  atmp2 = rtree->alist[k];
	  while(atmp2->next){
	    atmp2 = atmp2->next;
	    //
	    for(j=0;j<rtree->hithla;j++)
	      if(rtree->hrank[j]->id == atmp2->hla->id)
		break;    
	    if(j < rtree->hithla){
	      hit = false;
	      for(l=0;l<rtree->paircnt;l++){
		for(s=0;s<2;s++){
		  if(rtree->bestpair[l][s] == rtree->sameid[j]){
		    hit = true;
		    break;
		  }		  
		}
		if(hit)
		  break;
	      }
		
	      if(hit){
		//	      
		bool finhit = true;
		for(l=0;l<lcnt;l++){
		  if(atmp2->hla->hit[l] && !tmpfin[l]){
		    finhit = false;
		    break;
		  }
		}
		if(!finhit){
		  cout << "Reuse:" << atmp2->hla->aname << endl;
		  atmp->next = new ALIST();
		  atmp = atmp->next;
		  atmp->next = NULL;
		  atmp->hla = atmp2->hla;
		  rtmp->alistcnt[k]++;
		  for(l=0;l<lcnt;l++){
		    if(!atmp2->hla->hit[l])
		      tmpused[l] = 0;
		  }
		}
	      }//
	    }
	  }

	  if(rtmp->alistcnt[k] == 0){
	    atmp2 = rtree->alist[k];
	    while(atmp2->next){
	      atmp2 = atmp2->next;
	      bool finhit = true;
	      for(l=0;l<lcnt;l++){
		if(atmp2->hla->hit[l] && !tmpfin[l]){
		  finhit = false;
		  break;
		}
	      }
	      if(!finhit){
		cout << "Reuse:" << atmp2->hla->aname << endl;
		atmp->next = new ALIST();
		atmp = atmp->next;
		atmp->next = NULL;
		atmp->hla = atmp2->hla;
		rtmp->alistcnt[k]++;
		for(l=0;l<lcnt;l++){
		  if(!atmp2->hla->hit[l])
		    tmpused[l] = 0;
		}
	      }	   			
	    }
	  }
	}
      }
      else{
	if(anew[k][i] >= 0){
	  if(afin[anew[k][i]]){
	    atmp2 = rtree->alist[anew[k][i]];
	    while(atmp2->next){
	      atmp2 = atmp2->next;
	      atmp->next = new ALIST();
	      atmp = atmp->next;
	      atmp->next = NULL;
	      atmp->hla = atmp2->hla;
	      rtmp->alistcnt[k]++;
	    }
	    continue;
	  }
	}
	
	if(anew[k][i]==-1 && rtree->rankid[k][i] >= 0 && !rtree->alist[1]->next){
	  for(j=0;j<rtree->hithla;j++){	    
	    if(rtree->sameid[j] == rtree->rankid[k][i]){
	      atmp->next = new ALIST();
	      atmp = atmp->next;
	      atmp->next = NULL;
	      atmp->hla = rtree->hrank[j];
	      cout << "New Add:" << atmp->hla->aname << endl;
	      rtmp->alistcnt[k]++;
	      for(l=0;l<lcnt;l++){
		if(!rtree->hrank[j]->hit[l]){
		  tmpused[l] = 0;
		}
	      } 
	    }
	  }
	}
	else{
	  for(j=0;j<rtree->hithla;j++){
	    if(rtree->sameid[j] == rtree->rankid[k][i]){
	      atmp2 = rtree->alist[anew[k][i]];	    
	      while(atmp2->next){
		atmp2 = atmp2->next;
		if(rtree->hrank[j]->id == atmp2->hla->id){
		  break;
	      }
	      }
	      if(atmp2->all || rtree->hrank[j]->id == atmp2->hla->id){	      	
		atmp->next = new ALIST();
		atmp = atmp->next;
		atmp->next = NULL;
		atmp->hla = rtree->hrank[j];
		rtmp->alistcnt[k]++;
		for(l=0;l<lcnt;l++){
		  if(!rtree->hrank[j]->hit[l]){
		    tmpused[l] = 0;
		  }
		} 
	      }
	    }
	  }
	}
      }
    }
  
    int usecnt = 0;
 
    for(j=0;j<lcnt;j++){
      if(tmpused[j] == 1)
	usecnt++;
    }
  
    if(usecnt == usedcnt){
      int *tmpused2[2];
      int usedcnt2[2];
      for(k=0;k<2;k++){
	tmpused2[k] = new int[lcnt];
	for(l=0;l<lcnt;l++)
	  tmpused2[k][l] = 0;
	atmp = rtmp->alist[k];
	while(atmp->next){
	  atmp = atmp->next;
	  for(l=0;l<lcnt;l++){
	    if(atmp->hla->hit[l]){
	      tmpused2[k][l] = 1;
	    }            
	  }
	}
      }
      maxl = 0;
      for(k=0;k<2;k++){
	j = 0;
	for(l=0;l<lcnt;l++){
	  if(tmpused2[k][l])
	    j++;	  
	}          
	if(maxl < j)
	  maxl = j;
      }      
	
      for(k=0;k<2;k++){
	if(anew[k][i] >= 0){
	  if(afin[anew[k][i]]){
	    usedcnt2[k] = usecnt;
	    continue;
	  }
	}
	atmp = rtmp->alist[k];
	while(atmp->next){
	  atmp = atmp->next;
	  j = 0;
	  for(l=0;l<lcnt;l++){
	    if(atmp->hla->hit[l])
	      j++;
	  }
	  if(j > usecnt){
	    for(l=0;l<lcnt;l++){
	      if(!atmp->hla->hit[l])
		tmpused2[k][l] = 0;
	    }
	  }
	}
	usedcnt2[k] = 0;
	for(l=0;l<lcnt;l++)
	  if(tmpused2[k][l] == 1)
	    usedcnt2[k]++;
      }
      
      cout << "Maxl\t" << maxl << endl;
      cout << "Usedcnt\t" << usedcnt2[0] << "\t" << usedcnt2[1] << endl;

      if(usedcnt2[0] <= usecnt && usedcnt2[1] <= usecnt){
	for(j=0;j<lcnt;j++)
	  tmpused[j] = 0;
	for(k=0;k<2;k++){
	  rtmp->alistcnt[k] = 0;
	  atmp = rtmp->alist[k];
	  while(atmp->next){
	    atmp = atmp->next;
	    s=0;
	    for(l=0;l<lcnt;l++){
	      if(atmp->hla->hit[l]){
		s++;
		tmpused[l] = 1;
	      }
	    }
	    if(s==maxl)
	      rtmp->alistcnt[k]++;
	  }
	}	
      }
      else{
	if(usedcnt2[0] > usecnt && usedcnt2[1] <= usecnt){
	  for(j=0;j<lcnt;j++)
	    tmpused[j] = tmpused2[0][j];
	  usecnt = usedcnt2[0];
	}
	else if(usedcnt2[1] > usecnt && usedcnt2[0] <= usecnt){
	  for(j=0;j<lcnt;j++)
	    tmpused[j] = tmpused2[1][j];
	  usecnt = usedcnt2[1];
	}
	else{
	  if(usedcnt2[0] < usedcnt2[1]){
	    for(j=0;j<lcnt;j++)
	    tmpused[j] = tmpused2[0][j];
	    usecnt = usedcnt2[0];
	  }
	  else{
	    for(j=0;j<lcnt;j++)
	      tmpused[j] = tmpused2[1][j];
	    usecnt = usedcnt2[1];
	  }	  
	}
      }
      cout << "Expand Use:";
      usecnt = 0;
      for(j=0;j<lcnt;j++){
	cout << tmpused[j];
	if(tmpused[j])
	  usecnt++;
      }
      cout << " " << usecnt << endl;

      for(k=0;k<2;k++)
	delete[] tmpused2[k];
    }   

    bool nextcheck = true;
    if(reuse){
      if(rtmp->alistcnt[0] == 0 && rtmp->alistcnt[1] == 0){
	for(k=0;k<2;k++){
	  atmp = rtmp->alist[k];
	  rtmp->alist[k] = rtree->alist[k];
	  rtmp->alistcnt[k] = rtree->alistcnt[k];
	  while(atmp->next){
	    atmp2 = atmp;
	    atmp = atmp->next;
	    delete atmp2;
	  }
	  delete atmp;
	}
	nextcheck = false;
      }
      else{
	for(k=0;k<2;k++){
	  if(rtmp->alistcnt[k] == 0){
	    atmp = rtmp->alist[k];
	    rtmp->alist[k] = rtree->alist[k];
	    rtmp->alistcnt[k] = rtree->alistcnt[k];
	    while(atmp->next){
	      atmp2 = atmp;
	      atmp = atmp->next;
	      delete atmp2;
	    }
	    delete atmp;
	  }
	}
      }
    }
    else{
      //if((rtmp->alistcnt[0] == 0 && rtmp->alistcnt[1] == 0) || 
      // (rtree->rankid[1][i] >= 0 && rtree->alistcnt[1] > 0 && rtmp->alistcnt[1] == 0))
      if(rtmp->alistcnt[0] <= 1 && rtmp->alistcnt[1] <= 1){
	nextcheck = false;
      }  
    }

    rtmp->used = new int[lcnt];
    rtmp->finl = new int[lcnt];
    rtmp->depth = rtree->depth+1;
    rtmp->fin = false;
    rtmp->bestucnt = tmpbucnt[i];
     
    for(j=0;j<lcnt;j++){
      rtmp->used[j] = tmpused[j];
      rtmp->finl[j] = tmpfin[j];
    }
    delete[] tmpused;
    delete[] tmpfin;
      
    int colcnt = 0;
    bool same6[2];
    for(k=0;k<2;k++){
      same6[k] = false;
      if(rtmp->alist[k]->next){
	atmp = rtmp->alist[k]->next;
	n = 0;
	GetWordast(n,atmp->hla->aname);      
	word = "";
	do{
	  word += GetWordcol(n,atmp->hla->aname);
	  colcnt++;
	  if(colcnt == 3)
	    break;
	}while(n < atmp->hla->aname.size());
	cout << k << " 6digit:" << word << " " << colcnt << endl;
	if(colcnt == 3){
	  same6[k] = true;       
	  while(atmp->next){
	    atmp = atmp->next;
	    word2 = "";
	    n = 0;
	    GetWordast(n,atmp->hla->aname);
	    for(j=0;j<3;j++){
	      word2 += GetWordcol(n,atmp->hla->aname);
	    }
	    cout << word2 << endl;
	    if(word != word2){
	      same6[k] = false;
	      break;
	    }
	  }
	}
      }
    }
    
    if(nextcheck && usecnt > 0 && fincnt < maxl){
      //if(usecnt == maxl){
      //rtmp->covth = 1.0;
      //}
      //else{
      rtmp->covth = covth;
      //}
      
      rtmp->hrank = new HLA_HASH*[hlacnt];
      rtmp->sameid = new int[hlacnt];   
      rtmp->hlabest = new HLA_HASH*[BESTMAX];
      rtmp->diffcnt = new double*[BESTMAX];
      rtmp->diffcnt2 = new double*[BESTMAX];
      rtmp->mm = new int[BESTMAX];
      rtmp->pmm = new int[BESTMAX];
      rtmp->rsum = new int[BESTMAX];
      rtmp->bestcnt = 0;
      for(j=0;j<BESTMAX;j++){
	rtmp->diffcnt[j] = new double[BESTMAX];
	rtmp->diffcnt2[j] = new double[BESTMAX];
      }	
      Create_Rank_Tree(rtmp,hh,lcnt,hcnt,hlacnt,lmode,maxlm,lset);
    }
    else{
      for(k=0;k<2;k++){
	cout << "Allele " << k+1 << endl;
	atmp = rtmp->alist[k];
	while(atmp->next){
	  atmp = atmp->next;
	  cout << atmp->hla->aname << endl;
	}
      }
      cout << "Finish." << endl << endl;
      rtmp->fin = true;
    }      
  }
  delete[] tmpbucnt;
  for(i=0;i<2;i++)
    delete[] anew[i];
  
}

