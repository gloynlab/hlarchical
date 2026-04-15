void Re_Select_Best(RANK_TREE *rtree,int lcnt){
  int i,j;
  double tmpco,maxco;
  int i0,i1;

  maxco = 0;
  for(i=0;i<rtree->paircnt;i++){
    i0 = rtree->bestpair[i][0];
    i1 = rtree->bestpair[i][1];
    
    if(i1 >= 0)
      tmpco = (double)(rtree->readcnt[i0] + rtree->diffcnt[i1][i0] + rtree->readcnt2[i0] + rtree->diffcnt2[i1][i0] + rtree->readcnt[i1] + rtree->diffcnt[i0][i1] + rtree->readcnt2[i1] + rtree->diffcnt2[i0][i1]);
    else
      tmpco = (double)(rtree->readcnt[i0] + rtree->readcnt2[i0]);
	
    if(tmpco > maxco)
      maxco = tmpco;    
  }
  
  int tmpbestcnt = 0;
  for(i=0;i<rtree->paircnt;i++){
    i0 = rtree->bestpair[i][0];
    i1 = rtree->bestpair[i][1];

    if(i1 >= 0)
      tmpco = (double)(rtree->readcnt[i0] + rtree->diffcnt[i1][i0] + rtree->readcnt2[i0] + rtree->diffcnt2[i1][i0] + rtree->readcnt[i1] + rtree->diffcnt[i0][i1] + rtree->readcnt2[i1] + rtree->diffcnt2[i0][i1]);
    else
      tmpco = (double)(rtree->readcnt[i0] + rtree->readcnt2[i0]);

    if(fabs(tmpco - maxco) < 0.0000001){
      rtree->bestpair[tmpbestcnt][0] = rtree->bestpair[i][0];
      rtree->bestpair[tmpbestcnt][1] = rtree->bestpair[i][1];
      tmpbestcnt++;
    }
  }
  rtree->paircnt = tmpbestcnt;
  

}
