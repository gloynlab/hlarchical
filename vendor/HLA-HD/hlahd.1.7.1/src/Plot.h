void Plot_Reads(ofstream &out,HLA_HASH *htmp,List_Set *lset,int lcnt){
  int i,j,k,l,l2,ii;
  MPED_FQ *mptmp,*mptmp2;
  IE_PAIR_MPED_FQ *iepair,*iepairtmp;
  iepair = new IE_PAIR_MPED_FQ();  
  iepair->next = NULL;
  iepairtmp = iepair;

  bool hit;
  int cnt[3];
  for(i=0;i<3;i++)
    cnt[i] = 0;
  
  for(l=0;l<lcnt;l++){
    if(!htmp->hit[l])
      continue;
    mptmp = htmp->mp[0][l];
    while(mptmp->next){
      mptmp = mptmp->next;
      hit = false;
      for(l2=0;l2<lcnt;l2++){
	if(!htmp->hit[l2])
	  continue;
	
	mptmp2 = htmp->mp[1][l2];
	while(mptmp2->next){
	  mptmp2 = mptmp2->next;
	  if(mptmp->fq == mptmp2->fq){
	    iepairtmp->next = new IE_PAIR_MPED_FQ();
	    iepairtmp = iepairtmp->next;
	    iepairtmp->next = NULL;
	    iepairtmp->mp[0] = mptmp;
	    iepairtmp->mp[1] = mptmp2;
	    iepairtmp->lcnt[0] = l;
	    iepairtmp->lcnt[1] = l2;
	    cnt[1]++;
	    hit = true;
	    break;
	  }
	}
	if(hit)
	  break;
      }
      if(!hit){
	iepairtmp->next = new IE_PAIR_MPED_FQ();
	iepairtmp = iepairtmp->next;
	iepairtmp->next = NULL;
	iepairtmp->mp[0] = mptmp;
	iepairtmp->mp[1] = NULL;
	iepairtmp->lcnt[0] = l;
	iepairtmp->lcnt[1] = -1;
	cnt[0]++;
	hit = true;
      }
    }

    mptmp = htmp->mp[1][l];
    while(mptmp->next){
      mptmp = mptmp->next;
      hit = false;
      for(l2=0;l2<lcnt;l2++){
	if(!htmp->hit[l2])
	  continue;
	mptmp2 = htmp->mp[0][l2];
	while(mptmp2->next){
	  mptmp2 = mptmp2->next;
	  if(mptmp->fq == mptmp2->fq){
	    hit = true;
	    break;
	  }
	}
	if(hit)
	  break;
      }
      if(!hit){
	iepairtmp->next = new IE_PAIR_MPED_FQ();
	iepairtmp = iepairtmp->next;
	iepairtmp->next = NULL;
	iepairtmp->mp[1] = mptmp;
	iepairtmp->mp[0] = NULL;
	iepairtmp->lcnt[1] = l;
	iepairtmp->lcnt[0] = -1;
	hit = true;
	cnt[2]++;
      }
    }
  }

  int allcnt = cnt[0]+cnt[1]+cnt[2];
  out << htmp->aname << "\t" << allcnt << endl;
  iepairtmp = iepair;
  out << "R1 only\t" << cnt[0] << endl;
  int nlen = 0;
  for(i=0;i<lcnt;i++){
    if(htmp->nlen[i] > 0){
      nlen = htmp->nlen[i];
      break;
    }
  }
  
  while(iepairtmp->next){
    iepairtmp = iepairtmp->next;
    if(!iepairtmp->mp[1]){
      out << iepairtmp->mp[0]->fq->id;
      out << "\t" << lset->exname[iepairtmp->lcnt[0]];
      out << "\t" << iepairtmp->mp[0]->s - nlen + 1 << "\t" << iepairtmp->mp[0]->e - nlen + 1; //1,Sep,2017
      out << "\t" << iepairtmp->mp[0]->fq->w[0] << endl;
    }        
  }
  iepairtmp = iepair;
  out << "R2 only\t" << cnt[2] << endl;
  while(iepairtmp->next){
    iepairtmp = iepairtmp->next;
    if(!iepairtmp->mp[0]){
      out << iepairtmp->mp[1]->fq->id;
      out << "\t" << lset->exname[iepairtmp->lcnt[1]];
      out << "\t" << iepairtmp->mp[1]->s - nlen + 1 << "\t" << iepairtmp->mp[1]->e - nlen + 1; //1,Sep,2017
      out << "\t" << iepairtmp->mp[1]->fq->w[1] << endl;
    }        
  }

  iepairtmp = iepair;
  out << "Pair\t" << cnt[1] << endl;
  while(iepairtmp->next){
    iepairtmp = iepairtmp->next;
    if(iepairtmp->mp[0] && iepairtmp->mp[1]){
      out << iepairtmp->mp[1]->fq->id;
      out << "\t" << lset->exname[iepairtmp->lcnt[0]];
      out << "\t" << iepairtmp->mp[0]->s - nlen + 1 << "\t" << iepairtmp->mp[0]->e - nlen + 1; //1,Sep,2017     
      out << "\t" << lset->exname[iepairtmp->lcnt[1]];
      out << "\t" << iepairtmp->mp[1]->s - nlen + 1 << "\t" << iepairtmp->mp[1]->e - nlen + 1;
      //out << "\t" << iepairtmp->mp[0]->fq->w[0] << "\t" << iepairtmp->mp[1]->fq->w[1] << endl;
      out << "\t" << iepairtmp->mp[0]->fq->w[0] << endl; //1,Sep,2017
    }        
  }
}
