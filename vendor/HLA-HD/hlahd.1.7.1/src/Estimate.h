void All_Estimation(HLA_HASH *htmp1,HLA_HASH *htmp2,HLA_HASH **hh,List_Set *lset,int lcnt,int hlacnt){
  int i,j,k,l,s,ii;
  
  HLA_HASH **hrank = new HLA_HASH*[hlacnt];
  HLA_HASH *htmp;
  bool search[2];
  RANK_TREE *rtmp;

  for(l=0;l<lcnt;l++){
    if(htmp1->hit[l])
      search[0] = false;    
    if(htmp2){
      if(htmp2->hit[l])
	search[1] = false;
    }
    else
      search[1] = false;    
  }

}
