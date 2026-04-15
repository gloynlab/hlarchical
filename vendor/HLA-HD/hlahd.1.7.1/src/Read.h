void Search_Hash_Position(int lcnt,List_Set *lset){
  int i,j,k,l,s,n;
  int ii;
  ifstream ins;
  string idword,sline;
  char sw;
  
  int **pcnt,*psum;
  pcnt = new int*[hsize2*3];
  psum = new int[hsize2*3];
  for(i=0;i<hsize2*3;i++){
    psum[i] = 0;
    pcnt[i] = new int[10];
    for(j=0;j<10;j++)
      pcnt[i][j] = 0;
  }

  for(i=0;i<hsize2;i++)
    hpos[i] = -1;

  for(l=0;l<lcnt;l++){
    for(ii=0;ii<2;ii++){
      ins.open(lset->exfile[ii][l].c_str(),ios::in);
      if(!ins){
	cout << "Couldn't open sam file " << lset->exfile[ii][l] << "." << endl;
	exit(0);
      } 
      
      while(getline(ins,sline)){
	n = 0;
	idword = GetWordtab(n,sline);
	if(idword[0] == '@')
	  continue;
	j = idword.size();  
	
	for(i=0;i<hsize2*3;i++){
	  if(i>=j){
	    psum[i]++;
	    pcnt[i][0]++;
	    break;
	  }
	  sw = idword[j-i-1];
	  if ( sw < '0' || sw > '9' ) {
	    s = 0;
	  } else {	
	    s = (int)(sw - '0');
	  }
	  psum[i]++;
	  pcnt[i][s]++;
	}
      }
      ins.close();
    }
  }

  double pmax;
  double p;
  int pid;
  bool hit;
  
  for(l=0;l<hsize2;l++){
    pmax = -1.0;
    for(j=0;j<hsize2*3;j++){      
      hit = false;
      for(k=0;k<l;k++){
	if(hpos[k]==j){
	  hit = true;
	  break;
	}
      }
      if(hit)
	continue;
      
      p = 1.0;
      for(k=0;k<10;k++){
	p = p*(1.0-(double)pcnt[j][k]/psum[j]);
      }
      if(p > pmax){
	pmax = p;
	pid = j;
      }
    }
    hpos[l] = pid;
  }
  
  cout << "Hash position is ";
  for(i=0;i<hsize2;i++){
    cout << " " << hpos[i];
  }
  cout << endl;
  
  delete[] psum;
  for(i=0;i<hsize2*2;i++){
    delete[] pcnt[i];
  }  
  
}


void Read_All_Map(FASTQ_HASH **sh,int hcnt,int hcnt2){
  int i,ii,j,k,l,s,n,n2,ts;
  int fr;  
  int XN,XM,XG;
  bool N0,NS,NE;
  int samcnt;
  int locnt;
  ifstream inr,ins;
  char sw;
  string idword,word,word2,word3,word4,words[10],sline;
  int hid,mid,mid2,hitcnt;
  bool hit;
  long int start;
  FASTQ_HASH *stmp;

  int *ls = new int[hsize];
  ls[0] = 1;
  for(i=1;i<hsize;i++)
    ls[i] = 10*ls[i-1];

  int *ls2 = new int[hsize2];
  ls2[0] = 1;
  for(i=1;i<hsize2;i++)
    ls2[i] = 10*ls2[i-1];
        
  GENE_LIST *gtop,*gtmp;
  gtop = new GENE_LIST();
  gtop->next = NULL;
  int gcnt = 0,gecnt = 0;
  int *gfreq;

  bool excheck[2],incheck[2];
  for(i=0;i<2;i++){
    excheck[i] = false;
    incheck[i] = false;
  }
  string sename[2],siname[2];
  inr.open(loname.c_str(),ios::in);
  if(!inr){
    cout << "Couldn't open " << loname << endl;
    exit(0);
  }

  bool convert = false;
  string convname;
  while(getline(inr,sline)){
    n = 0;
    word = GetWordtab(n,sline);
    if(word == "convert"){
      convert = true;
      convname = GetWordtab(n,sline);
    }
    else if(word == "exon"){
      i = atoi(GetWordtab(n,sline).c_str());
      if(i < 1 || i > 2){
	cout << "Wrong line:" << endl;
	cout << sline << endl;
	exit(0);
      }
      excheck[i-1] = true;
      sename[i-1] = GetWordtab(n,sline);
    }
    else if(word == "intron"){
      i = atoi(GetWordtab(n,sline).c_str());
      if(i < 1 || i > 2){
	cout << "Wrong line:" << endl;
	cout << sline << endl;
	exit(0);
      }
      incheck[i-1] = true;
      siname[i-1] = GetWordtab(n,sline);
    }
    else{
      cout << "Wrong list word:" << word << endl;
      exit(0);
    }
  }
  inr.close();
  
  for(i=0;i<2;i++){
    if(!excheck[i]){
      cout << "All exon " << i+1 << " file is not recorded in list file." << endl;
      exit(0);
    }
    if(!incheck[i]){
      cout << "All intron " << i+1 << " file is not recorded in list file." << endl;
      exit(0);
    }
  }

  struct CONV_HASH{
    string name;
    int cnt;
    CONV_HASH *next;
  };
  struct CONV_LIST{
    string gname;   
    string einame; 
    CONV_HASH **chash;
    CONV_LIST *next;
  };
  CONV_LIST *clist,*cltmp;
  CONV_HASH *chtmp;
  clist = new CONV_LIST();
  clist->next = NULL;
  clist->gname = "";
  
  HLA_HASH **hh,*htmp;    

  if(convert){
    hh = new HLA_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      hh[i] = new HLA_HASH();
      hh[i]->next = NULL;
      hh[i]->id = "";
    }

    ifstream inc;
    inc.open(convname.c_str(),ios::in);
    if(!inc){
      cout << "Couldn't open convert file." << endl;
      exit(0);
    }
    while(getline(inc,sline)){
      n = 0;
      word = GetWordtab(n,sline);
      n2 = 0;
      word2 = GetWordcol(n2,word);
      word3 = GetWordtab(n2,word);
  
      hid = 0;
      j = word3.size();
      k = 0;
      i = 0;
      while(1){
	if(j-i-1 < 0)
	  break;
	sw = word3[j-i-1];
	if ( sw < '0' || sw > '9' ) {
	  i++;
	  continue;
	} else {	
	  s = (int)(sw - '0');
	}
	hid += ls[k]*s;
	k++;
	if(k == hsize)
	  break;
	i++;
      }
      
      cltmp = clist;
      n2 = 0;
      word4 = GetWordUbar(n2,word3);
      if(word4.size() >= 6){
	if(word4.substr(0,4) != "Exon" && word4.substr(0,6) != "Intron"){
	  word4 = "Gene";
	}
      }
      else if(word4.size() == 5){
	if(word4.substr(0,4) != "Exon")
	  word4 = "Gene";
      }
      else{
	word4 = "Gene";
      }

      while(cltmp->next){
	cltmp = cltmp->next;
	if(cltmp->gname == word2 && cltmp->einame == word4)
	  break;
      }      
      if(cltmp->gname != word2 || cltmp->einame != word4){
	cltmp->next = new CONV_LIST();
	cltmp = cltmp->next;
	cltmp->next = NULL;
	cltmp->gname = word2;
	cltmp->einame = word4;
	cltmp->chash = new CONV_HASH*[hcnt];
	for(i=0;i<hcnt;i++){
	  cltmp->chash[i] = new CONV_HASH();
	  cltmp->chash[i]->next = NULL;
	}
      }
      chtmp = cltmp->chash[hid];
      while(chtmp->next){
	chtmp = chtmp->next;
      }
      chtmp->next = new CONV_HASH();
      chtmp = chtmp->next;
      chtmp->next = NULL;
      chtmp->name = word3;
      word = GetWordtab(n,sline);
      chtmp->cnt = 0;
      n = 0;
      do{
	word2 = GetWordcam(n,word);
	n2 = 0;
	word2 = GetWorddot(n2,word2);
	chtmp->cnt++;

	hid = 0;
	j = word2.size();
	k = 0;
	i = 0;
	while(1){
	  if(j-i-1 < 0)
	    break;
	  sw = word2[j-i-1];
	  if ( sw < '0' || sw > '9' ) {
	    i++;
	    continue;
	  } else {	
	    s = (int)(sw - '0');
	  }
	  hid += ls[k]*s;
	  k++;
	  if(k == hsize)
	    break;
	  i++;
	}
	htmp = hh[hid];
	while(htmp->next){
	  htmp = htmp->next;
	  if(htmp->id == word2 && htmp->einame == cltmp->einame)
	    break;
	}
	if(htmp->id != word2 || htmp->einame != cltmp->einame){
	  htmp->next = new HLA_HASH();
	  htmp = htmp->next;
	  htmp->next = NULL;
	  htmp->id = word2;	  
	  htmp->aname = cltmp->gname;
	  htmp->einame = cltmp->einame;
	}
      }while(n < word.size());  
    }
    inc.close();
  }

  
  long int scnt;
  for(ii=0;ii<2;ii++){ 
    ins.open(sename[ii].c_str(),ios::in);
    if(!ins){
      cout << "Couldn't open sam file " << sename[ii] << "." << endl;
      exit(0);
    } 
    
    if(ii==0){
      gtmp = gtop;
      
      while(getline(ins,sline)){
	n = 0;
	if(sline[0] != '@')
	  break;
	word = GetWordtab(n,sline);
	if(word != "@SQ")
	  continue;
	word = GetWordtab(n,sline);
	n = 0;
	for(i=0;i<2;i++)
	  word2 = GetWordcol(n,word);
	word3 = GetWordUbar(n,word);
	if(word3.size() >= 6){
	  if(word3.substr(0,4) != "Exon" && word3.substr(0,6) != "Intron"){
	    word3 = "Gene";
	  }
	}
	else if(word3.size() == 5){
	  if(word3.substr(0,4) != "Exon")
	    word3 = "Gene";
	}
	else
	  word3 = "Gene";

	gtmp = gtop;
	while(gtmp->next){
	  gtmp = gtmp->next;
	  if(gtmp->name == word2 && gtmp->einame == word3)
	    break;
	}
	if(gtmp->name == word2 && gtmp->einame == word3){
	  gtmp->freq++;
	}
	else{
	  gtmp->next = new GENE_LIST();
	  gtmp = gtmp->next;
	  gtmp->next = NULL;
	  gtmp->name = word2;
	  gtmp->einame = word3;
	  gtmp->freq = 1;
	  gtmp->id = gecnt;
	  gecnt++;
	}
      }
    }
    
                 
    bool sread = false;
    scnt=0;
    while(1){
      if(sread){
	if(!getline(ins,sline)){
	  break;
	}
      }
      else{
	sread = true;
      }
      if(sline.size() == 0){
	continue;
      }
      scnt++;

      n = 0;
 
      idword = GetWordtab(n,sline);
      if(idword[0] == '@')
	continue;
      j = idword.size();  
      hid = 0;
      for(i=0;i<hsize2;i++){
	if(hpos[i]>=j){
	  continue;
	}
	sw = idword[j-hpos[i]-1];
	if ( sw < '0' || sw > '9' ) {
	  s = 0;
	} else {	
	  s = (int)(sw - '0');
	}
	hid += ls2[i]*s;
      }

      mid = atoi(GetWordtab(n,sline).c_str());
	    
      fr = ii;	  
      stmp = sh[hid];
	    
      hit = false;
      while(stmp->next){
	stmp = stmp->next;
	if(stmp->id == idword){
	  hit = true;
	  break;
	}
      }

      if(!hit)
	continue;

      word2 = GetWordtab(n,sline);    
      start = atoi(GetWordtab(n,sline).c_str());
      GetWordtab(n,sline);
      if(GetWordtab(n,sline) == "*")
	continue;
	    
      for(i=0;i<3;i++)
	GetWordtab(n,sline);
      ts = GetWordtab(n,sline).size();
	    	
      if(ts < MINLENGTH)
	continue;

      hit = false;
      do{
	word = GetWordtab(n,sline);
	if(word.substr(0,2) == "XN"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  XN = atoi(GetWordtab(n2,word).c_str());
	}
	else if(word.substr(0,2) == "NM"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  hitcnt = atoi(GetWordtab(n2,word).c_str());
	}
	else if(word.substr(0,2) == "MD"){
	  if(word[5] == '0' && word[6] == 'N')
	    N0 = false;
	  else
	    N0 = true;	  
	}
      }while(n < sline.size());

      if(hitcnt <= EMTH  || XN + EMTH >= hitcnt)
	hit = true;
      if(ts-hitcnt+EMTH < MINMATCH)
	hit = false;
      if(!hit)
	continue;

      stmp->ts[fr] = ts;
	
      if(stmp->freq[0] == NULL){
	for(i=0;i<2;i++){
	  stmp->freq[i] = new int[gecnt*2];
	  for(j=0;j<gecnt*2;j++){
	    stmp->freq[i][j] = 0;
	  }
	}
      }
   
      n2 = 0;
      word3 = word2;
      word2 = GetWordcol(n2,word2);      
      word4 = "Gene";

      if(convert){
	word3 = GetWordtab(n2,word3);
	n2 = 0;
	word4 = GetWordUbar(n2,word3);
	if(word4.size() >= 6){
	  if(word4.substr(0,4) != "Exon" && word4.substr(0,6) != "Intron"){
	    word4 = "Gene";
	  }
	}
	else if(word4.size() == 5){
	  if(word4.substr(0,4) != "Exon")
	    word4 = "Gene";
	}
	else
	  word4 = "Gene";

	hid = 0;
	j = word3.size();
	k = 0;
	i = 0;
	while(1){
	  if(j-i-1 < 0)
	    break;
	  sw = word3[j-i-1];
	  if ( sw < '0' || sw > '9' ) {
	    i++;
	    continue;
	  } else {	
	    s = (int)(sw - '0');
	  }
	  hid += ls[k]*s;
	  k++;
	  if(k == hsize)
	    break;
	  i++;
	}

	cltmp = clist;
	while(cltmp->next){
	  cltmp = cltmp->next;
	  if(cltmp->gname == word2 && cltmp->einame == word4)
	    break;
	}   

	if(cltmp->gname != word2 || cltmp->einame != word4){	 
	  cout << "Convert list and All exon file are not matched 1." << endl;
	  exit(0);
	}
	chtmp = cltmp->chash[hid];
	while(chtmp->next){
	  chtmp = chtmp->next;
	  if(chtmp->name == word3)
	    break;
	}
	if(chtmp->name != word3){
	  cout << "Convert list and All exon file are not matched 2." << endl;
	  exit(0);
	}
	k = chtmp->cnt;
      }
      else{
	k = 1;
      }
   
      gtmp = gtop;
      while(gtmp->next){
	gtmp = gtmp->next;
	if(gtmp->name == word2 && gtmp->einame == word4){	  	  
	  stmp->freq[fr][gtmp->id] += k;
	}	    
      }
    }                    
    ins.close();
  }

  for(ii=0;ii<2;ii++){
    ins.open(siname[ii].c_str(),ios::in);
    if(!ins){
      cout << "Intron mapped file couldn't be opend." << endl;
      exit(0);
    } 
	
    if(ii==0){
      gtmp = gtop;
        
      while(getline(ins,sline)){
	n = 0;
	if(sline[0] != '@')
	  break;
	word = GetWordtab(n,sline);
	if(word != "@SQ")
	  continue;
	word = GetWordtab(n,sline);
	n = 0;
	for(i=0;i<2;i++)
	  word2 = GetWordcol(n,word);
	word3 = GetWordUbar(n,word);
	if(word3.size() >= 6){
	  if(word3.substr(0,4) != "Exon" && word3.substr(0,6) != "Intron"){
	    word3 = "Gene";
	  }
	}
	else if(word3.size() == 5){
	  if(word3.substr(0,4) != "Exon")
	    word3 = "Gene";
	}
	else
	  word3 = "Gene";

	gtmp = gtop;
	while(gtmp->next){
	  gtmp = gtmp->next;
	  if(gtmp->name == word2 && gtmp->einame == word3)
	    break;
	}
	if(gtmp->name == word2 && gtmp->einame == word3){
	  gtmp->freq++;
	}
	else{
	  gtmp->next = new GENE_LIST();
	  gtmp = gtmp->next;
	  gtmp->next = NULL;
	  gtmp->name = word2;
	  gtmp->einame = word3;
	  gtmp->freq = 1;
	  gtmp->id = gecnt;
	  gecnt++;
	}
      }
    }  

    bool sread = false;
    scnt = 0;
    while(1){
      if(sread){
	if(!getline(ins,sline)){
	  break;
	}
      }
      else{
	sread = true;
      }
      if(sline.size() == 0){
	continue;
      }
	  
      n = 0;
  
      idword = GetWordtab(n,sline);
      if(idword[0] == '@')
	continue;
      j = idword.size();  
      hid = 0;
      for(i=0;i<hsize2;i++){
	if(hpos[i]>=j){
	  continue;
	}
	sw = idword[j-hpos[i]-1];
	if ( sw < '0' || sw > '9' ) {
	  s = 0;
	} else {	
	  s = (int)(sw - '0');
	}
	hid += ls2[i]*s;
      }
	  
      mid = atoi(GetWordtab(n,sline).c_str());	  
      fr = ii;	  
      stmp = sh[hid];
	  
      hit = false;
      while(stmp->next){
	stmp = stmp->next;
	if(stmp->id == idword){
	  hit = true;
	  break;
	}
      }

      if(!hit)
	continue;

      word2 = GetWordtab(n,sline);    
      start = atoi(GetWordtab(n,sline).c_str());
      GetWordtab(n,sline);
      if(GetWordtab(n,sline) == "*")
	continue;
	  
      for(i=0;i<3;i++)
	GetWordtab(n,sline);
      ts = GetWordtab(n,sline).size();
	  
      if(ts < MINLENGTH)
	continue;

      hit = false;
      do{
	word = GetWordtab(n,sline);
	if(word.substr(0,2) == "XN"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  XN = atoi(GetWordtab(n2,word).c_str());
	}
	else if(word.substr(0,2) == "NM"){
	  n2 = 0;
	  for(i=0;i<2;i++)
	    GetWordcol(n2,word);
	  hitcnt = atoi(GetWordtab(n2,word).c_str());
	}
      }while(n < sline.size());
	  
      if(hitcnt <= INMTH || XN + INMTH >= hitcnt)
	hit = true;

      if(ts-hitcnt+INMTH < MINMATCH)
	hit = false;

      if(!hit)
	continue;
	  
      stmp->ts[fr] = ts;
      if(stmp->freq[0] == NULL){
	for(i=0;i<2;i++){
	  stmp->freq[i] = new int[gecnt];
	  for(j=0;j<gecnt;j++){
	    stmp->freq[i][j] = 0;
	  }
	}
      }
   	       
      n2 = 0;
      word3 = word2;
      word2 = GetWordcol(n2,word2);
      word4 = "Gene";

      if(convert){
	word3 = GetWordtab(n2,word3);
	n2 = 0;
	word4 = GetWordUbar(n2,word3);
	if(word4.size() >= 6){
	  if(word4.substr(0,4) != "Exon" && word4.substr(0,6) != "Intron"){
	    word4 = "Gene";
	  }
	} 
	else if(word4.size() == 5){
	  if(word4.substr(0,4) != "Exon")
	    word4 = "Gene";
	}        
	else
	  word4 = "Gene";
	hid = 0;
	j = word3.size();
	k = 0;
	i = 0;
	while(1){
	  if(j-i-1 < 0)
	    break;
	  sw = word3[j-i-1];
	  if ( sw < '0' || sw > '9' ) {
	    i++;
	    continue;
	  } else {	
	    s = (int)(sw - '0');
	  }
	  hid += ls[k]*s;
	  k++;
	  if(k == hsize)
	    break;
	  i++;
	}
	cltmp = clist;
	while(cltmp->next){
	  cltmp = cltmp->next;
	  if(cltmp->gname == word2 && cltmp->einame == word4)
	    break;
	}      

	if(cltmp->gname != word2 || cltmp->einame != word4){
	  cout << "Convert list and All exon file are not matched 3." << endl;
	  exit(0);
	}
	chtmp = cltmp->chash[hid];
	while(chtmp->next){
	  chtmp = chtmp->next;
	  if(chtmp->name == word3)
	    break;
	}
	if(chtmp->name != word3){
	  cout << "Convert list and All exon file are not matched 4." << endl;
	  exit(0);
	}
	k = chtmp->cnt;
      }
      else{
	k = 1;
      }

      gtmp = gtop;
      while(gtmp->next){
	gtmp = gtmp->next;
	if(gtmp->name == word2 && gtmp->einame == word4){
	  stmp->freq[fr][gtmp->id] += k;
	  break;
	}
      }

    }                    
    ins.close();		
  }
   
  
  gfreq = new int[gecnt];
  if(convert){
    gtmp = gtop;
    while(gtmp->next){
      gtmp = gtmp->next;   
      gtmp->freq = 0;
      cltmp = clist;
      while(cltmp->next){
	cltmp = cltmp->next;
	if(cltmp->gname == gtmp->name && cltmp->einame == gtmp->einame)
	  break;
      }      
      
      for(i=0;i<hcnt;i++){
	htmp = hh[i];
	while(htmp->next){
	  htmp = htmp->next;
	  if(htmp->aname == gtmp->name && htmp->einame == gtmp->einame)
	    gtmp->freq++;
	}
      }
      gfreq[gtmp->id] = gtmp->freq;
    }
  }
  else{
    gtmp = gtop;
    while(gtmp->next){
      gtmp = gtmp->next;
      gfreq[gtmp->id] = gtmp->freq;
    }
  }

  string *gnames = new string[gecnt];
  int *gids = new int[gecnt];
  int gid;
  i = 0;
  gtmp = gtop;
  while(gtmp->next){
    gtmp = gtmp->next;    
    hit = false;
    for(j=0;j<i;j++){
      if(gnames[j] == gtmp->name){
	hit = true;
	gids[gtmp->id] = j;
	break;
      }
    }
    if(!hit){
      gnames[i] = gtmp->name;
      if(gnames[i] == gname)
	gid = i;
      gids[gtmp->id] = i;
      i++;  
    }
   }
  gcnt = i;

  double wsum1[2],wsum2[2];
  bool w0,w1;
  double *gfreqs[2],*gfreqs2[2],*ghits[2];
  for(i=0;i<2;i++){
    gfreqs[i] = new double[gcnt];
    gfreqs2[i] = new double[gcnt];
    ghits[i] = new double[gcnt];
  }

  double tmpw;
  GENE_LIST *gltmp;
  for(i=0;i<hcnt2;i++){
    stmp = sh[i];
    while(stmp->next){
      stmp = stmp->next; 

      if(stmp->w[0] < 0){
	for(j=0;j<2;j++){
	  stmp->w[j] = 0.0;
	}
	continue;
      }
    
      if(!stmp->freq[0]){
	stmp->w[0] = 1.0;
	stmp->w[1] = 1.0;
      }
      else{	
	for(j=0;j<2;j++){
	  for(k=0;k<gcnt;k++){
	    gfreqs[j][k] = 1.0;
	    ghits[j][k] = 0;
	  }
	  for(k=0;k<gecnt;k++){
	    if(stmp->freq[j][k] > 0){
	      gfreqs[j][gids[k]] *= (double)stmp->freq[j][k]/gfreq[k];
	      ghits[j][gids[k]]++;

	    }
	  }
	}
	    
	for(j=0;j<2;j++){
	  stmp->w[j] = 1.0;
	  wsum1[j] = 0.0; wsum2[j] = 0.0;

	  w0 = false;
	  for(k=0;k<gcnt;k++){
	    if(k == gid)
	      continue;
	    if(gfreqs[1-j][gid] > 0 && gfreqs[1-j][k] == 0)
	      continue;
	    else if(gfreqs[1-j][gid] == 0 && gfreqs[1-j][k] > 0){
	      w0 = true;
	      break;
	    }
	    else if(gfreqs[j][k] > 0 && ghits[j][k] > 0){
	      wsum2[j] += (double)gfreqs[j][k];
	    }
	  }
	  
	  if(gfreqs[j][gid] > 0 && ghits[j][gid] > 0){
	    wsum1[j] = (double)gfreqs[j][gid];
	    	    
	    if(w0){
	      stmp->w[j] = 0.0;
	    }
	    else{		   
	      stmp->w[j] = wsum1[j]/(wsum1[j]+wsum2[j]);
	    }	    
	  }
	  else{
	    stmp->w[j] = 0.0;
	  }	
	}
	if((wsum1[0] > 0 && wsum1[1] > 0) && (wsum2[0] == 0 || wsum2[1] == 0)){
	  stmp->w[0] = 1.0;
	  stmp->w[1] = 1.0;
	}
	else if((wsum2[0] > 0 && wsum2[1] > 0) && (wsum1[0] == 0 || wsum1[1] == 0)){
	  stmp->w[0] = 0.0;
	  stmp->w[1] = 0.0;
	}
	
	if(wsum1[0] > 0 && wsum1[1] > 0){
	  if(stmp->w[0] < stmp->w[1]){
	    stmp->w[1] = stmp->w[0];
	  }
	  else{
	   stmp->w[0] = stmp->w[1];
	  }	  
	}
	
      }

      for(j=0;j<2;j++)
	if(stmp->w[j] < MIN_WEIGHT)
	  stmp->w[j] = 0.0;

    }
  }
  
 
}

int Read_List_File(List_Set *lset){
  int MAX_EXON = 100000;
  int i,j,k,n;
  int lcnt = 0;
  int ilcnt = 0;
  int faecnt = 0;
  int faicnt = 0;
  int efcnt[2],ifcnt[2];
  for(i=0;i<2;i++){
    efcnt[i] = 0;
    ifcnt[i] = 0;
  }
    
  string word,words[10],sline;

  string *tmpes = new string[MAX_EXON]; 
  int *used = new int[MAX_EXON];
  ifstream inl;
  bool icheck;
  icheck = false;

  inl.open(lname.c_str(),ios::in);
  while(getline(inl,sline)){
    n = 0;
    word = GetWordtab(n,sline);

    if(word == "convert"){
      convert_read = true;
      cvname = GetWordtab(n,sline);
    }
    else if(word == "exon"){
      if(icheck){
	cout << "Exon list must be written at first." << endl;
	exit(0);
      }
      if(lcnt == MAX_EXON){
	cout << "Too many exons." << endl;
	exit(0);	
      }
      tmpes[lcnt] = GetWordtab(n,sline);
      used[lcnt] = atoi(GetWordtab(n,sline).c_str());
      if(used[lcnt] < 0 || used[lcnt] > 1){
	cout << "Wrong line. 1" << endl;
	cout << sline << endl;
	exit(0);
      }
      lcnt++;
    }
    else if(word == "intron"){
      if(!icheck){
	lset->lcnt = lcnt;
	if(lcnt == 0){
	  cout << "Exon is not recorded." << endl;
	  exit(0);
	}
	if(lcnt >= 2){
	  lset->inname = new string[lcnt-1];
	  lset->lpos = new int[lcnt-1];
	  lset->rpos = new int[lcnt-1];
	  lset->faifile = new string[lcnt-1];
	  lset->exidl = new int[lcnt-1];
	  lset->exidr = new int[lcnt-1];
	  for(i=0;i<lcnt-1;i++){
	    lset->exidl[i] = -1;
	    lset->exidr[i] = -1;
	  }
	}
	lset->intidl = new int[lcnt];
	lset->intidr = new int[lcnt];
	lset->exname = new string[lcnt];
	lset->inname = new string[lcnt];
	lset->faefile = new string[lcnt];
	lset->used = new int[lcnt];	
	for(i=0;i<lcnt;i++){
	  lset->intidl[i] = -1;
	  lset->intidr[i] = -1;
	}

	for(i=0;i<2;i++){
	  lset->exfile[i] = new string[lcnt];
	  if(lcnt >= 2)
	    lset->infile[i] = new string[lcnt-1];
	}
	icheck = true;
      }
      for(i=0;i<3;i++)
	words[i] = GetWordtab(n,sline);
      for(i=0;i<lcnt;i++){
	if(words[1] == tmpes[i]){
	  lset->intidr[i] = ilcnt;
	  lset->exidl[ilcnt] = i;
	  break;
	}
      }
      if(i==lcnt){
	cout << words[1] << " is not recorded in list file." << endl;
	exit(0);
      }
      for(i=0;i<lcnt;i++){
	if(words[2] == tmpes[i]){
	  lset->intidl[i] = ilcnt;
	  lset->exidr[ilcnt] = i;
	  break;
	}
      }
      if(i==lcnt){
	cout << words[1] << " is not recorded in list file." << endl;
	exit(0);
      }
      lset->inname[ilcnt] = words[0];
      ilcnt++;
    }
    else if(word == "fasta"){
      if(!icheck){
	lset->lcnt = lcnt;
	if(lcnt == 0){
	  cout << "Exon is not recorded." << endl;
	  exit(0);
	}
	if(lcnt >= 2){
	  lset->inname = new string[lcnt-1];
	  lset->lpos = new int[lcnt-1];
	  lset->rpos = new int[lcnt-1];
	  lset->faifile = new string[lcnt-1];
	  lset->exidl = new int[lcnt-1];
	  lset->exidr = new int[lcnt-1];
	  for(i=0;i<lcnt-1;i++){
	    lset->exidl[i] = -1;
	    lset->exidr[i] = -1;
	  }
	}
	lset->intidl = new int[lcnt];
	lset->intidr = new int[lcnt];
	lset->exname = new string[lcnt];
	lset->inname = new string[lcnt];
	lset->faefile = new string[lcnt];
	lset->used = new int[lcnt];	
	for(i=0;i<lcnt;i++){
	  lset->intidl[i] = -1;
	  lset->intidr[i] = -1;
	}

	for(i=0;i<2;i++){
	  lset->exfile[i] = new string[lcnt];
	  if(lcnt >= 2)
	    lset->infile[i] = new string[lcnt-1];

	}
	icheck = true;
      }

      word = GetWordtab(n,sline);
      for(i=0;i<lcnt;i++){
	if(word == tmpes[i]){
	  break;
	}
      }
      if(i < lcnt){
	lset->faefile[i] = GetWordtab(n,sline);
	faecnt++;
      }
      else{
	for(i=0;i<lcnt-1;i++){
	  if(word == lset->inname[i]){
	    break;
	  }
	}
	if(i < lcnt-1){
	  lset->lpos[i] = atoi(GetWordtab(n,sline).c_str());
	  lset->rpos[i] = atoi(GetWordtab(n,sline).c_str());
	  lset->faifile[i] = GetWordtab(n,sline);
	  faicnt++;
	}
	else{
	  cout << "Wrong line. 5" << endl;
	  cout << sline << endl;
	  exit(0);
	}
      }       
    }
    else{
      for(i=0;i<lcnt;i++){
	if(word == tmpes[i]){
	  break;
	}
      }
      if(i < lcnt){
	j = atoi(GetWordtab(n,sline).c_str());
	if(j < 0 || j > 2){
	  cout << "Wrong line. 2" << endl;
	  cout << sline << endl;
	  exit(0);
	}
	lset->exfile[j-1][i] = GetWordtab(n,sline);
	efcnt[j-1]++;
      }
      else{
	for(i=0;i<lcnt-1;i++){
	  if(word == lset->inname[i]){
	    break;
	  }
	}
	if(i < lcnt-1){
	  for(k=0;k<2;k++)
	    words[k] = GetWordtab(n,sline);	  
	  j = atoi(words[0].c_str());
	  if(j < 0 || j > 2){
	    cout << "Wrong line. 3" << endl;
	    cout << sline << endl;
	    exit(0);
	  }
	  lset->infile[j-1][i] = words[1];
	  ifcnt[j-1]++;
	}
	else{
	  cout << lcnt << " " << ilcnt << endl;
	  cout << "Wrong line. 4" << endl;
	  cout << sline << endl;
	  exit(0);
	}
      }    
    }
  }
  if(ilcnt != lcnt-1 || efcnt[0] != lcnt || efcnt[1] != lcnt
     || ifcnt[0] != lcnt-1 || ifcnt[1] != lcnt-1
     || faecnt != lcnt || faicnt != lcnt-1){
    cout << "All exon and intron files must be recorded in list file." << endl;
    exit(0);
  }

  for(i=0;i<lcnt;i++){
    lset->exname[i] = tmpes[i];
    lset->used[i] = used[i];
  }

  for(i=0;i<lcnt;i++){
    cout << lset->exname[i];
    if(lset->intidl[i] >= 0){
      cout << "\t" << lset->inname[lset->intidl[i]];
    }
    else
      cout << "\tNULL";
    if(lset->intidr[i] >= 0){
      cout << "\t" << lset->inname[lset->intidr[i]];
    }
    else
      cout << "\tNULL";
    cout << "\t" << lset->faefile[i] << "\t" << lset->exfile[0][i] 
	 << "\t" << lset->exfile[1][i] << endl;
  }
  for(i=0;i<lcnt-1;i++){
    cout << lset->inname[i];
    if(lset->exidl[i] >= 0)
      cout << "\t" <<  lset->exname[lset->exidl[i]];
    else
      cout << "\tNULL";
    if(lset->exidr[i] >= 0)
      cout << "\t" << lset->exname[lset->exidr[i]];
    else 
      cout << "\tNULL";    
    cout << "\t" << lset->lpos[i] << "\t" 
	 << lset->rpos[i] << "\t"
	 << lset->faifile[i] << "\t" << lset->infile[0][i] << "\t" 
	 << lset->infile[1][i] << endl; 
  }
  
  delete[] tmpes;
  delete[] used;
  return lcnt;
};

int Read_Exon_Map(FASTQ_HASH **sh,HLA_HASH **hh,int lcnt,int hcnt,int hcnt2,List_Set *lset){
  int i,ii,j,k,l,s,n,n2,ts;
  int fr;  
  int XN,XM;
  int D,I;
  string MD;
  bool N0,NS,NE;
  int locnt;
  ifstream inr,ins;
  char sw;
  string idword,word,word2,word3,word4,words[10],sline;
  int hid,mid,mid2,hitcnt;
  bool dir,hit,ehit;
  long int start;
  string reads;
  FASTQ_HASH *stmp;
  GENE_LIST *gltmp;
  HLA_HASH *htmp,*htmp2;
  int hlacnt = 0;
  ifstream inf;
  MPED_FQ *mptmp,*mptmp2,*mptpm3;
  string fname; 
  string code;
  
  int *ls = new int[hsize];
  ls[0] = 1;
  for(i=1;i<hsize;i++)
    ls[i] = 10*ls[i-1];  
  int *ls2 = new int[hsize2];
  ls2[0] = 1;
  for(i=1;i<hsize2;i++)
    ls2[i] = 10*ls2[i-1];  

  struct CONV_HASH{
    string id;
    int cnt;
    string *name;
    string *al;
    int *hid;
    CONV_HASH *next;
  }; 
  CONV_HASH ***che,***chi;
  CONV_HASH *chtmp;
  che = new CONV_HASH**[lcnt];
  for(l=0;l<lcnt;l++){
    che[l] = new CONV_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      che[l][i] = new CONV_HASH();
      che[l][i]->next = NULL;
    }
  }
  chi = new CONV_HASH**[lcnt-1];
  for(l=0;l<lcnt-1;l++){
    chi[l] = new CONV_HASH*[hcnt];
    for(i=0;i<hcnt;i++){
      chi[l][i] = new CONV_HASH();
      chi[l][i]->next = NULL;
    }
  }

  if(convert_read){
    ifstream inc;
    bool exon;
    inc.open(cvname.c_str(),ios::in);
    if(!inc){
      cout << "Couldn't open convert file." << endl;
      exit(0);
    }
    while(getline(inc,sline)){
      n = 0;
      word = GetWordtab(n,sline);
      n2 = 0;
      word2 = GetWordcol(n2,word);
      if(word2 != gname)
	continue;
      word3 = GetWordUbar(n2,word);

      if(word3.substr(0,4)=="exon" || word3.substr(0,4)=="Exon"){
	word3 = "exon" + word3.substr(4,word3.size()-4);
      }
      else if(word3.substr(0,6)=="intron" || word3.substr(0,6)=="Intron"){
	word3 = "intron" + word3.substr(6,word3.size()-6);
      }
      else if(word3.substr(0,4)=="3utr" || word3.substr(0,4)=="5utr")
	continue;

      for(l=0;l<lcnt;l++){
	if(word3 == lset->exname[l]){
	  exon=true;
	  break;
	}
	if(word3 == lset->inname[l]){
	  exon=false;
	  break;
	}
      }
      if(l==lcnt){
	cout << sline << endl;
	cout << "Wrong convert list." << endl;
	exit(0);
      }
      word3 = GetWordtab(n2,word);
      hid = 0;
      j = word3.size();
      k = 0;
      i = 0;
      while(1){
	if(j-i-1 < 0)
	  break;
	sw = word3[j-i-1];
	if ( sw < '0' || sw > '9' ) {
	  i++;
	  continue;
	} else {	
	  s = (int)(sw - '0');
	}
	hid += ls[k]*s;
	k++;
	if(k == hsize)
	  break;
	i++;
      }

      if(exon)
	chtmp = che[l][hid];
      else
	chtmp = chi[l][hid];
      while(chtmp->next){
	chtmp = chtmp->next;
      }
      chtmp->next = new CONV_HASH();
      chtmp = chtmp->next;
      chtmp->next = NULL;
      chtmp->id = word3;
      
      word = GetWordtab(n,sline);
      n2 = 0;
      chtmp->cnt = 0;
      do{
	GetWordcam(n2,word);
	chtmp->cnt++;
      }while(n2 < word.size());
      n2 = 0;
      chtmp->name = new string[chtmp->cnt];
      chtmp->al = new string[chtmp->cnt];
      chtmp->hid = new int[chtmp->cnt];
      for(l=0;l<chtmp->cnt;l++){
	chtmp->name[l] = GetWorddot(n2,word);
	chtmp->al[l] = GetWordcam(n2,word);
	hid = 0;
	j = chtmp->name[l].size();
	k = 0;
	i = 0;
	while(1){
	  if(j-i-1 < 0)
	    break;
	  sw = chtmp->name[l][j-i-1];
	  if ( sw < '0' || sw > '9' ) {
	    i++;
	    continue;
	  } else {	
	    s = (int)(sw - '0');
	  }
	  hid += ls[k]*s;
	  k++;
	  if(k == hsize)
	    break;
	  i++;
	}
	chtmp->hid[l] = hid;
      }
    }
  }

  int *tmphid;
  int tmphcnt;
  int tmpnlen;
  int tmplen;
  char *tmpcode;
  string *tmpname;
  string *tmpal;

  for(l=0;l<lcnt;l++){ 
    inf.open(lset->faefile[l].c_str(),ios::in);
    cout << lset->faefile[l] << endl;
    if(inf){
      bool sread = true;
      do{
	if(sread){
	  if(!getline(inf,sline)){
	    break;
	  }	    
	}
	else{
	  sread = true;
	}
	if(sline[0] == '>'){
	  n = 1;
	  word = GetWordsp(n,sline);	 
	  if(convert_read){
	    n2 = 0;
	    word2 = GetWordcol(n2,word);
	    if(word2 != gname){
	      cout << "Wrong sam file 1." << endl;
	      exit(0);
	    }
	    word3 = GetWordUbar(n2,word);
	    word3 = "exon" + word3.substr(4,word3.size()-4);
	    if(word3 != lset->exname[l]){
	      cout << "Wrong sam file 2." << endl;
	      exit(0);
	    }
	    word3 = GetWordtab(n2,word);
	    hid = 0;
	    j = word3.size();
	    k = 0;
	    i = 0;
	    while(1){
	      if(j-i-1 < 0)
		break;
	      sw = word3[j-i-1];
	      if ( sw < '0' || sw > '9' ) {
		i++;
		continue;
	      } else {	
		s = (int)(sw - '0');
	      }
	      hid += ls[k]*s;
	      k++;
	      if(k == hsize)
		break;
	      i++;
	    }	  
	    chtmp = che[l][hid];
	    while(chtmp->next){
	      chtmp = chtmp->next;
	      if(chtmp->id == word3)
		break;
	    }
	    
	    tmphid = new int[chtmp->cnt];
	    tmpname = new string[chtmp->cnt];
	    tmpal = new string[chtmp->cnt];
	    for(j=0;j<chtmp->cnt;j++){
	      tmphid[j] = chtmp->hid[j];
	      tmpname[j] = chtmp->name[j];
	      tmpal[j] = chtmp->al[j];	      
	    }
	    tmphcnt = chtmp->cnt;
	  }
	  else{
	    word2 = GetWordsp(n,sline);
	    hid = 0;
	    j = word.size();
	    k = 0;
	    i = 0;
	    while(1){
	      if(j-i-1 < 0)
		break;
	      sw = word[j-i-1];
	      if ( sw < '0' || sw > '9' ) {
		i++;
		continue;
	      } else {	
		s = (int)(sw - '0');
	      }
	      hid += ls[k]*s;
	      k++;
	      if(k == hsize)
		break;
	      i++;
	    }
	    tmphid = new int[1];
	    tmpname = new string[1];
	    tmpal = new string[1];
	    tmphid[0] = hid;
	    tmpname[0] = word;
	    tmpal[0] = word2;
	    tmphcnt = 1;
	  }

	  word = GetWordsp(n,sline);
	  tmplen = atoi(word.substr(0,word.size()-3).c_str());
	  word2 = GetWordsp(n,sline);
	  tmpnlen = 0;
	  if(word2.substr(0,5)=="Nlen="){	      
	    tmpnlen = atoi(word2.substr(5,word2.size()-8).c_str());
	  } 
	  tmpcode = new char[tmplen+2*tmpnlen];

	  k = 0;
	  while(getline(inf,sline)){
	    if(sline[0] == '>'){
	      sread = false;
	      break;
	    }
	    for(i=0;i<sline.size();i++){
	      tmpcode[k] = UpperC(sline[i]);
	      k++;
	    }
	  }

	  for(s=0;s<tmphcnt;s++){ 
	    htmp = hh[tmphid[s]];
	    while(htmp->next){
	      htmp = htmp->next;
	      if(htmp->id == tmpname[s])
		break;
	    }
	    
	    if(htmp->id != tmpname[s]){
	      htmp->next = new HLA_HASH();
	      htmp = htmp->next;
	      htmp->next = NULL;
	      htmp->id = tmpname[s];
	      htmp->aname = tmpal[s];
	      htmp->len = new int[lcnt];
	      htmp->es = new int[lcnt];
	      htmp->ee = new int[lcnt];
	      htmp->hit = new bool[lcnt];
	      htmp->edge = new bool[lcnt];
	      htmp->hitcnt = new bool*[lcnt];
	      htmp->hitcntp = new bool*[lcnt];
	      htmp->nmcnt = new int[lcnt];
	      htmp->nmcntp = new int[lcnt];
	      htmp->rsum = new int[lcnt];
	      htmp->exvcnt = new double[lcnt];
	      htmp->exv2cnt = new double[lcnt];
	      htmp->nlen = new int[lcnt];
	      htmp->mm = new int[lcnt];
	      htmp->mmp = new int[lcnt];
	      htmp->code = new char*[lcnt];
	      htmp->pcnt = 1;
	      htmp->plot = true;	      
	      htmp->samee = new HLA_HASH*[lcnt];

	      for(i=0;i<2;i++){
		htmp->mp[i] = new MPED_FQ*[lcnt];		
		htmp->mptmp[i] = new MPED_FQ*[lcnt];
	      }
	      for(i=0;i<lcnt;i++){
		htmp->hit[i] = false;	      
		htmp->edge[i] = false;
		htmp->nlen[i] = -1;
	      }

	      //Intron 
	      htmp->ilen = new int[lcnt-1];
	      htmp->is = new int[lcnt-1];
	      htmp->ie = new int[lcnt-1];
	      htmp->ihit = new bool[lcnt-1];
	      htmp->ihitcnt = new bool*[lcnt-1];
	      htmp->ihitcntp = new bool*[lcnt-1];
	      htmp->inmcnt = new int[lcnt-1];
	      htmp->inmcntp = new int[lcnt-1];
	      htmp->irsum = new int[lcnt-1];
	      htmp->invcnt = new double[lcnt-1];
	      htmp->inlen = new int[lcnt-1];
	      htmp->imm = new int[lcnt-1];
	      htmp->immp = new int[lcnt-1];
	      htmp->imiss = new int[lcnt-1];
	      htmp->icode = new char*[lcnt-1];	    
	      htmp->esticode = new string[lcnt-1];
	      htmp->estD = new int[lcnt-1];
	      htmp->estI = new int[lcnt-1];
	      htmp->estis = new int[lcnt-1];
	      htmp->estie = new int[lcnt-1];
	      htmp->samei = new HLA_HASH*[lcnt-1];

	      for(i=0;i<2;i++){
		htmp->imp[i] = new MPED_FQ*[lcnt-1];		
		htmp->imptmp[i] = new MPED_FQ*[lcnt-1];
	      }
	      for(i=0;i<lcnt-1;i++){
		htmp->ihit[i] = false;
		htmp->imiss[i] = 0;
		htmp->esticode[i] = "";	      
	      }

	      hlacnt++;
	    } 
	    htmp->hit[l] = true;
	    htmp->len[l] = tmplen; 
	    htmp->nlen[l] = tmpnlen;
	    for(i=0;i<2;i++){
	      htmp->mp[i][l] = new MPED_FQ();
	      htmp->mp[i][l]->next = NULL;
	      htmp->mptmp[i][l] = htmp->mp[i][l];
	      htmp->mp[i][l]->fq = NULL;
	    } 
	    htmp->es[l] = htmp->nlen[l];
	    htmp->ee[l] = htmp->len[l]+htmp->nlen[l]-1;
	    htmp->len[l] += 2*htmp->nlen[l];
	    htmp->hitcnt[l] = new bool[htmp->len[l]];
	    htmp->hitcntp[l] = new bool[htmp->len[l]];

	    for(i=0;i<htmp->len[l];i++){
	      htmp->hitcnt[l][i] = false;
	      htmp->hitcntp[l][i] = false;
	    }	   
	    htmp->code[l] = new char[htmp->len[l]];
	    for(i=0;i<htmp->len[l];i++){
	      htmp->code[l][i] = tmpcode[i];
	    }
	    if(s == 0){
	      htmp->samee[l] = NULL;
	      htmp2 = htmp;
	    }
	    else
	      htmp->samee[l] = htmp2;
	  }

	  delete[] tmphid;
	  delete[] tmpname;
	  delete[] tmpal;
	  delete[] tmpcode;
	}
      }while(1);
      inf.close();
    }
    else{
      cout << "Fasta file couldn't be opened." << endl;
      exit(0);
    }
    
    cout << lset->exname[l] << endl;
    for(ii=0;ii<2;ii++){
      ins.open(lset->exfile[ii][l].c_str(),ios::in);
      if(!ins){
	cout << "Couldn't open sam file " << lset->exfile[ii][l] << "." << endl;
	exit(0);
      } 

      while(getline(ins,sline)){
	n = 0;

	idword = GetWordtab(n,sline);
	if(idword[0] == '@')
	  continue;
	j = idword.size();  
	hid = 0;
	for(i=0;i<hsize2;i++){
	  if(hpos[i]>=j){
	    continue;
	  }
	  sw = idword[j-hpos[i]-1];
	  if ( sw < '0' || sw > '9' ) {
	    s = 0;
	  } else {	
	    s = (int)(sw - '0');
	  }
	  hid += ls2[i]*s;
	}
		
	mid = atoi(GetWordtab(n,sline).c_str());

	fr = ii;	  
	stmp = sh[hid];
	
	word2 = GetWordtab(n,sline);    
	start = atoi(GetWordtab(n,sline).c_str());
	GetWordtab(n,sline);
	if(GetWordtab(n,sline) == "*")
	  continue;
	
	for(i=0;i<3;i++)
	  GetWordtab(n,sline);
	reads = GetWordtab(n,sline);

	hit = false;
	NS = false;
	NE = false;
	XN=0;
	do{
	  word = GetWordtab(n,sline);
	  if(word.substr(0,2) == "XN"){
	    n2 = 0;
	    for(i=0;i<2;i++)
	      GetWordcol(n2,word);
	    XN = atoi(GetWordtab(n2,word).c_str());
	  }	  
	  else if(word.substr(0,2) == "NM"){
	    n2 = 0;
	    for(i=0;i<2;i++)
	      GetWordcol(n2,word);
	    hitcnt = atoi(GetWordtab(n2,word).c_str());
	  }
	  else if(word.substr(0,2) == "MD"){
	    if(word[5] == '0' && word[6] == 'N')
	      NS = true;
	    if(word[word.size()-1] == '0' && word[word.size()-2] == 'N')
	      NE = true;
	    MD = word.substr(5,word.size()-5);	  
	  }
	}while(n < sline.size());
	if(hitcnt == 0 || XN == hitcnt)
	  hit = true;
	if(!hit)
	  continue;

	reads = UpperS(reads);
	ts = reads.size();

	if(ts < MINLENGTH)
	  continue;

	if(ts-hitcnt < MINMATCH)
	  continue;

	hit = false;
	while(stmp->next){
	  stmp = stmp->next;
	  if(stmp->id == idword){
	    hit = true;
	    break;
	  }
	}
	
	if(!hit){
	  stmp->next = new FASTQ_HASH();
	  stmp = stmp->next;
	  stmp->next = NULL;
	  stmp->id = idword;
	  for(i=0;i<2;i++){
	    stmp->w[i] = 1.0;
	    stmp->edge[i] = false;
	  }
	  stmp->paired = false;	 
	  stmp->ipaired = false;	  
	  for(i=0;i<2;i++){
	    stmp->freq[i] = NULL;
	    stmp->minhit[i] = EMTH+INMTH+1;	  
	    stmp->gl[i] = new GENE_LIST();
	    stmp->gl[i]->next = NULL;
	    stmp->ehit[i] = false;
	  }
	  stmp->inhit = false;	  
	}
	
	stmp->ehit[fr] = true;
	stmp->ts[fr] = ts;
	stmp->code[fr] = reads;
	if(hitcnt-XN < stmp->minhit[fr])
	  stmp->minhit[fr] = hitcnt-XN;

	if(mid == 0 || mid == 256)
	  dir = true;
	else if(mid == 16 || mid == 272){
	  dir = false;	  
	}
	else{
	  cout << "Wrong sam ID:" << mid << endl;
	  exit(0);
	}
	
	if(convert_read){
	  n2 = 0;
	  word3 = word2;
	  word2 = GetWordcol(n2,word2);
	  if(word2 != gname){
	    cout << "Wrong sam file 1." << endl;
	    exit(0);
	  }
	  word2 = word3;
	  word3 = GetWordUbar(n2,word3);
	  word3 = "exon" + word3.substr(4,word3.size()-4);

	  if(word3 != lset->exname[l]){
	    cout << "Wrong sam file 2." << endl;
	    exit(0);
	  }
	  word3 = GetWordtab(n2,word2);
	  hid = 0;
	  j = word3.size();
	  k = 0;
	  i = 0;
	  while(1){
	    if(j-i-1 < 0)
	      break;
	    sw = word3[j-i-1];
	    if ( sw < '0' || sw > '9' ) {
	      i++;
	      continue;
	    } else {	
	      s = (int)(sw - '0');
	    }
	    hid += ls[k]*s;
	    k++;
	    if(k == hsize)
	      break;
	    i++;
	  }	  
	  chtmp = che[l][hid];
	  while(chtmp->next){
	    chtmp = chtmp->next;
	    if(chtmp->id == word3)
	      break;
	  }

	  for(j=0;j<chtmp->cnt;j++){
	    htmp = hh[chtmp->hid[j]];
	    while(htmp->next){
	      htmp = htmp->next;
	      if(htmp->id == chtmp->name[j]){
		break;
	      }
	    }
	    if(htmp->id != chtmp->name[j]){
	      cout << "Fasta file is not matched to sam data." << endl;
	      exit(0);	
	    }
	    
	    htmp->mptmp[fr][l]->next = new MPED_FQ();
	    htmp->mptmp[fr][l] = htmp->mptmp[fr][l]->next;
	    mptmp = htmp->mptmp[fr][l];
	    mptmp->next = NULL;
	    mptmp->fq = stmp;
	    mptmp->paired = false;
	    mptmp->opaired[0] = false;
	    mptmp->opaired[1] = false;
	    mptmp->pair = NULL;
	    mptmp->NS = 0;
	    mptmp->NE = 0;

	    if(XN > 0){
	      if(NS && NE){
		for(i=0;i+1<MD.size();i+=2){
		  if(MD[i] == '0' && MD[i+1] == 'N')
		    mptmp->NS++;
		  else
		    break;
		}
		for(i=MD.size()-1;i-1>=0;i-=2){
		  if(MD[i-1] == 'N' && MD[i] == '0')
		    mptmp->NE++;
		  else
		    break;
		}
		mptmp->s = htmp->es[l];
		mptmp->e = htmp->ee[l];
	      }
	      else if(NS){
		mptmp->s = htmp->es[l];
		mptmp->e = start+ts-2;
		mptmp->NS = XN;
	      }
	      else if(NE){
		mptmp->s = start-1;
		mptmp->e = htmp->ee[l];
		mptmp->NE = XN;
	      }
	    }
	    else{
	      mptmp->s = start-1;
	      mptmp->e = start+ts-2;	
	    }
	    
	  }
	}
	else{
	  hid = 0;
	  j = word2.size();
	  k = 0;
	  i = 0;
	  while(1){
	    if(j-i-1 < 0)
	      break;
	    sw = word2[j-i-1];
	    if ( sw < '0' || sw > '9' ) {
	      i++;
	      continue;
	    } 
	    else {	
	      s = (int)(sw - '0');
	    }
	    hid += ls[k]*s;
	    k++;
	    if(k == hsize)
	      break;
	    i++;
	  }
	
	  htmp = hh[hid];
	  while(htmp->next){
	    htmp = htmp->next;
	    if(htmp->id == word2){
	      break;
	    }
	  }
	  if(htmp->id != word2){
	    cout << "Fasta file is not matched to sam data." << endl;
	    exit(0);	
	  }

	  htmp->mptmp[fr][l]->next = new MPED_FQ();
	  htmp->mptmp[fr][l] = htmp->mptmp[fr][l]->next;
	  mptmp = htmp->mptmp[fr][l];
	  mptmp->next = NULL;
	  mptmp->fq = stmp;
	  mptmp->paired = false;
	  mptmp->opaired[0] = false;
	  mptmp->opaired[1] = false;
	  mptmp->pair = NULL;

	  if(XN > 0){
	    if(NS && NE){
	      for(i=0;i+1<MD.size();i+=2){
		if(MD[i] == '0' && MD[i+1] == 'N')
		  mptmp->NS++;
		else
		  break;
	      }
	      for(i=MD.size()-1;i-1>=0;i-=2){
		if(MD[i-1] == 'N' && MD[i] == '0')
		  mptmp->NE++;
		else
		  break;
	      }
	      mptmp->s = htmp->es[l];
	      mptmp->e = htmp->ee[l];
	    }
	    else if(NS){
	      mptmp->s = htmp->es[l];
	      mptmp->e = start+ts-2;
	      mptmp->NS = XN;
	    }
	    else if(NE){
	      mptmp->s = start-1;
	      mptmp->e = htmp->ee[l];
	      mptmp->NE = XN;
	    }
	  }
	  else{
	    mptmp->s = start-1;
	    mptmp->e = start+ts-2;	
	  }
	}

      }                  
      ins.close();
    }
  }

  
  int exidl,exidr;
  FASTQ_CHAIN *fqctop,*fqctmp,*fqcback;
  bool hit2,hit3;

  for(l=0;l<lcnt-1;l++){
    exidl = lset->exidl[l];
    exidr = lset->exidr[l];
    inf.open(lset->faifile[l].c_str(),ios::in);
    cout << lset->faifile[l] << endl;
    if(inf){
      bool sread = true;
      do{
	if(sread){
	  if(!getline(inf,sline)){
	    break;
	  }	    
	}
	else{
	  sread = true;
	}
	if(sline[0] == '>'){
	  n = 1;
	  
	  word = GetWordsp(n,sline);
	  if(convert_read){
	    n2 = 0;
	    word2 = GetWordcol(n2,word);
	    if(word2 != gname){
	      cout << "Wrong sam file 1." << endl;
	      exit(0);
	    }
	    word3 = GetWordUbar(n2,word);
	    word3 = "intron" + word3.substr(6,word3.size()-6);
	    if(word3 != lset->inname[l]){
	      cout << "Wrong sam file 2." << endl;
	      exit(0);
	    }
	    word3 = GetWordtab(n2,word);
	    hid = 0;
	    j = word3.size();
	    k = 0;
	    i = 0;
	    while(1){
	      if(j-i-1 < 0)
		break;
	      sw = word3[j-i-1];
	      if ( sw < '0' || sw > '9' ) {
		i++;
		continue;
	      } else {	
		s = (int)(sw - '0');
	      }
	      hid += ls[k]*s;
	      k++;
	      if(k == hsize)
		break;
	      i++;
	    }	  
	    chtmp = chi[l][hid];
	    while(chtmp->next){
	      chtmp = chtmp->next;
	      if(chtmp->id == word3)
		break;
	    }
	    
	    tmphid = new int[chtmp->cnt];
	    tmpname = new string[chtmp->cnt];
	    tmpal = new string[chtmp->cnt];
	    for(j=0;j<chtmp->cnt;j++){
	      tmphid[j] = chtmp->hid[j];
	      tmpname[j] = chtmp->name[j];
	      tmpal[j] = chtmp->al[j];	      
	    }
	    tmphcnt = chtmp->cnt;
	  }
	  else{
	    word2 = GetWordsp(n,sline);
	    hid = 0;
	    j = word.size();
	    k = 0;
	    i = 0;
	    while(1){
	      if(j-i-1 < 0)
		break;
	      sw = word[j-i-1];
	      if ( sw < '0' || sw > '9' ) {
		i++;
		continue;
	      } else {	
		s = (int)(sw - '0');
	      }
	      hid += ls[k]*s;
	      k++;
	      if(k == hsize)
		break;
	      i++;
	    }
	    tmphid = new int[1];
	    tmpname = new string[1];
	    tmpal = new string[1];
	    tmphid[0] = hid;
	    tmpname[0] = word;
	    tmpal[0] = word2;
	    tmphcnt = 1;
	  }

	  word = GetWordsp(n,sline);
	  tmplen = atoi(word.substr(0,word.size()-3).c_str());
	  word2 = GetWordsp(n,sline);
	  tmpnlen = 0;
	  if(word2.substr(0,5)=="Nlen="){	      
	    tmpnlen = atoi(word2.substr(5,word2.size()-8).c_str());
	  } 
	  tmpcode = new char[tmplen+2*tmpnlen];

	  k = 0;
	  while(getline(inf,sline)){
	    if(sline[0] == '>'){
	      sread = false;
	      break;
	    }
	    for(i=0;i<sline.size();i++){
	      tmpcode[k] = UpperC(sline[i]);
	      k++;
	    }
	  }

	  for(s=0;s<tmphcnt;s++){ 
	    htmp = hh[tmphid[s]];
	    while(htmp->next){
	      htmp = htmp->next;
	      if(htmp->id == tmpname[s])
		break;
	    }

	    htmp->ihit[l] = true;
	    htmp->ilen[l] = tmplen;
	    htmp->inlen[l] = tmpnlen;
	    for(i=0;i<2;i++){
	      htmp->imp[i][l] = new MPED_FQ();
	      htmp->imp[i][l]->next = NULL;
	      htmp->imptmp[i][l] = htmp->imp[i][l];	   
	      htmp->imp[i][l]->fq = NULL;
	    }   
	    htmp->is[l] = htmp->inlen[l];
	    htmp->ie[l] = htmp->ilen[l]+htmp->inlen[l]-1;	  
	    htmp->ilen[l] += 2*htmp->inlen[l];
	    htmp->ihitcnt[l] = new bool[htmp->ilen[l]];
	    htmp->ihitcntp[l] = new bool[htmp->ilen[l]];
	    for(i=0;i<htmp->ilen[l];i++){
	      htmp->ihitcnt[l][i] = false;
	      htmp->ihitcntp[l][i] = false;
	    }	   
	    htmp->icode[l] = new char[htmp->ilen[l]];
	    for(i=0;i<htmp->ilen[l];i++){
	      htmp->icode[l][i] = tmpcode[i];
	    }

	    if(s == 0){
	      htmp->samei[l] = NULL;
	      htmp2 = htmp;
	    }
	    else
	      htmp->samei[l] = htmp2;
	  }

	  delete[] tmphid;
	  delete[] tmpname;
	  delete[] tmpal;
	  delete[] tmpcode;
	}
      }while(1);
      inf.close();
    }
    else{
      cout << "Fasta file couldn't be opened." << endl;
      exit(0);
    }

    for(ii=0;ii<2;ii++){
      ins.open(lset->infile[ii][l].c_str(),ios::in);
      if(!ins){
	cout << lset->infile[ii][l] << endl;
	cout << "Intron file couldn't be opend." << endl;
	exit(0);
      }

      fqctop = new FASTQ_CHAIN();
      fqctop->next = NULL;
      fqctmp = fqctop;
      
      while(getline(ins,sline)){
	n = 0;

	idword = GetWordtab(n,sline);
	if(idword[0] == '@')
	  continue;
	j = idword.size();  
	hid = 0;
	for(i=0;i<hsize2;i++){
	  if(hpos[i]>=j){
	    continue;
	  }
	  sw = idword[j-hpos[i]-1];
	  if ( sw < '0' || sw > '9' ) {
	    s = 0;
	  } else {	
	    s = (int)(sw - '0');
	  }
	  hid += ls2[i]*s;
	}
	
	mid = atoi(GetWordtab(n,sline).c_str());
	  
	fr = ii;	  
	stmp = sh[hid];
	  
	word2 = GetWordtab(n,sline);   

	start = atoi(GetWordtab(n,sline).c_str());
	GetWordtab(n,sline);
	word3 = GetWordtab(n,sline);


	if(word3 == "*")
	  continue;	  

	for(i=0;i<3;i++)
	  GetWordtab(n,sline);
	reads = GetWordtab(n,sline);


	hit = false;
	NS = false;
	NE = false;
	XN=0;
	do{
	  word = GetWordtab(n,sline);
	  if(word.substr(0,2) == "XN"){
	    n2 = 0;
	    for(i=0;i<2;i++)
	      GetWordcol(n2,word);
	    XN = atoi(GetWordtab(n2,word).c_str());
	  }	  
	  else if(word.substr(0,2) == "NM"){
	    n2 = 0;
	    for(i=0;i<2;i++)
	      GetWordcol(n2,word);
	    hitcnt = atoi(GetWordtab(n2,word).c_str());
	  }
	  else if(word.substr(0,2) == "MD"){
	    if(word[5] == '0' && word[6] == 'N')
	      NS = true;
	    if(word[word.size()-1] == '0' && word[word.size()-2] == 'N')
	      NE = true;
	    MD = word.substr(5,word.size()-5);	  
	  }
	}while(n < sline.size());
	if(hitcnt <= INMTH || XN + INMTH >= hitcnt)
	  hit = true;


	if(!hit)
	  continue;
	
	reads = UpperS(reads);
	ts = reads.size();
		
	if(ts < MINLENGTH)
	  continue;

	if(ts-hitcnt < MINMATCH)
	  continue;

	hit = false;
	while(stmp->next){
	  stmp = stmp->next;
	  if(stmp->id == idword){
	    hit = true;
	    break;
	  }
	}

	if(!hit){
	  stmp->next = new FASTQ_HASH();
	  stmp = stmp->next;
	  stmp->next = NULL;
	  stmp->id = idword;
	  for(i=0;i<2;i++){
	    stmp->w[i] = 1.0;
	    stmp->edge[i] = false;
	  }
	  stmp->paired = false;	
	  stmp->ipaired = false;
	  for(i=0;i<2;i++){
	    stmp->freq[i] = NULL;
	    stmp->minhit[i] = EMTH+INMTH+1;
	    stmp->gl[i] = new GENE_LIST();
	    stmp->gl[i]->next = NULL;
	    stmp->ehit[i] = false;
	  }
	  stmp->inhit = false;
	}else{
	  if(stmp->ehit[ii])
	    continue;
	}

	/*
	if(stmp->code[ii].size() > 0){
	  if(stmp->code[ii] != reads){
	    cout << stmp->id << " " << stmp->code[ii] << endl;
	    cout << stmp->id << " " << reads << endl;
	  }
	}
	*/
	
	stmp->code[ii] = reads;	
	stmp->ts[fr] = ts;
	if(hitcnt-XN < stmp->minhit[fr])
	  stmp->minhit[fr] = hitcnt-XN;

	if(!stmp->inhit){
	  for(i=0;i<2;i++){
	    stmp->inid[i] = new bool[lcnt-1];
	    for(j=0;j<lcnt-1;j++)
	      stmp->inid[i][j] = false;
	  }
	  stmp->inhit = true;
	}
	stmp->inid[fr][l] = true;
	
	//##################################################################
	//if(!intron_estimate && XN == 0)
	//  continue;
	//##################################################################

	fqctmp = fqctop;
	while(fqctmp->next){
	  fqctmp = fqctmp->next;
	  if(fqctmp->stmp == stmp)
	    break;
	}

	if(fqctmp->stmp != stmp){
	  fqctmp->next = new FASTQ_CHAIN();
	  fqctmp = fqctmp->next;
	  fqctmp->next = NULL;
	  fqctmp->stmp = stmp;
	  fqctmp->hit = false;
	  fqctmp->ilpos = new int[ts];
	  fqctmp->irpos = new int[ts];
	  for(j=0;j<ts;j++){
	    fqctmp->ilpos[j] = 0;
	    fqctmp->irpos[j] = 0;
	  }
	}

	if(mid == 0 || mid == 256)
	  dir = true;
	else if(mid == 16 || mid == 272){
	  dir = false;	  
	}
	else{
	  cout << "Wrong sam ID:" << mid << endl;
	  exit(0);
	}

	if(convert_read){
	  n2 = 0;
	  word4 = word2;
	  word2 = GetWordcol(n2,word2);
	  if(word2 != gname){
	    cout << "Wrong sam file 1." << endl;
	    exit(0);
	  }
	  word2 = word4;
	  word4 = GetWordUbar(n2,word4);
	  word4 = "intron" + word4.substr(6,word4.size()-6);

	  if(word4 != lset->inname[l]){
	    cout << "Wrong sam file 2." << endl;
	    exit(0);
	  }
	  word4 = GetWordtab(n2,word2);

	  hid = 0;
	  j = word4.size();
	  k = 0;
	  i = 0;
	  while(1){
	    if(j-i-1 < 0)
	      break;
	    sw = word4[j-i-1];
	    if ( sw < '0' || sw > '9' ) {
	      i++;
	      continue;
	    } else {	
	      s = (int)(sw - '0');
	    }
	    hid += ls[k]*s;
	    k++;
	    if(k == hsize)
	      break;
	    i++;
	  }	  
	  chtmp = chi[l][hid];
	  
	  while(chtmp->next){
	    chtmp = chtmp->next;
	    if(chtmp->id == word4)
	      break;
	  }
	  
	  D = 0;
	  I = 0;
	  n = 0;
	  do{
	    word2 = GetWordMID(n,word3);
	    if(word3[n-1]=='D'){
	      D += atoi(word2.c_str());
	    }
	    else if(word3[n-1]=='I'){
	      I += atoi(word2.c_str());
	    }
	  }while(n < word3.size());

	  for(j=0;j<chtmp->cnt;j++){
	    htmp = hh[chtmp->hid[j]];
	    while(htmp->next){
	      htmp = htmp->next;
	      if(htmp->id == chtmp->name[j]){
		break;
	      }
	    }
	    if(htmp->id != chtmp->name[j]){
	      cout << "Fasta file is not matched to sam data." << endl;
	      exit(0);	
	    }

	    htmp->imiss[l] += hitcnt-XN;
	    htmp->imptmp[fr][l]->next = new MPED_FQ();
	    htmp->imptmp[fr][l] = htmp->imptmp[fr][l]->next;	
	    mptmp = htmp->imptmp[fr][l];
	    mptmp->next = NULL;
	    mptmp->fq = stmp;
	    mptmp->paired = false;
	    mptmp->opaired[0] = false;
	    mptmp->opaired[1] = false;
	    mptmp->pair = NULL;
	    mptmp->NS = 0;
	    mptmp->NE = 0;
	
	    if(XN > 0){
	      if(NS && NE){
		mptmp->s = htmp->is[l];
		mptmp->e = htmp->ie[l];
		for(i=0;i+1<MD.size();i+=2){
		  if(MD[i] == '0' && MD[i+1] == 'N')
		    mptmp->NS++;
		  else
		    break;
		}
		for(i=MD.size()-1;i-1>=0;i-=2){
		  if(MD[i-1] == 'N' && MD[i] == '0')
		    mptmp->NE++;
		  else
		    break;
		}
	      }
	      else if(NS){
		mptmp->s = htmp->is[l];
		mptmp->e = start+ts+D-I-2;
		mptmp->NS = XN;
	      }
	      else if(NE){
		mptmp->s = start-1;
		mptmp->e = htmp->ie[l];
		mptmp->NE = XN;	  
	      }
	    }
	    else{
	      mptmp->s = start-1;
	      mptmp->e = start+ts+D-I-2;	
	    }

	    if(NE && NS){
	      if(start+ts+D-I-mptmp->NE-1 == htmp->ilen[l]-lset->rpos[l]){
		if(reads[ts-mptmp->NE-2] == 'A' && reads[ts-mptmp->NE-1] == 'G'){
		  fqctmp->irpos[ts-mptmp->NE]++;
		}
	      }
	      if(start+mptmp->NS-1==lset->lpos[l]){
		if(reads[mptmp->NS] == 'G' && reads[mptmp->NS+1] == 'T'){	    
		  fqctmp->ilpos[mptmp->NS-1]++;
		}
	      }
	    }
	    else if(NE){
	      if(start+ts+D-I-XN-1 == htmp->ilen[l]-lset->rpos[l]){
		if(reads[ts-XN-2] == 'A' && reads[ts-XN-1] == 'G'){
		  fqctmp->irpos[ts-XN]++;
		}
	      }
	    }      
	    else if(NS){
	      if(start+XN-1==lset->lpos[l]){
		if(reads[XN] == 'G' && reads[XN+1] == 'T'){	    
		  fqctmp->ilpos[XN-1]++;
		}
	      }
	    }
	  }
	}
	else{
	  hid = 0;
	  j = word2.size();
	  k = 0;
	  i = 0;
	  while(1){
	    if(j-i-1 < 0)
	      break;
	    sw = word2[j-i-1];
	    if ( sw < '0' || sw > '9' ) {
	      i++;
	      continue;
	    } else {	
	      s = (int)(sw - '0');
	    }
	    hid += ls[k]*s;
	    k++;
	    if(k == hsize)
	      break;
	    i++;
	  }
		
	  htmp = hh[hid];
	  while(htmp->next){
	    htmp = htmp->next;
	    if(htmp->id == word2){
	      break;
	    }
	  }
	  if(htmp->id != word2){
	    cout << "Fasta file is not matched to sam data." << endl;
	    exit(0);	
	  }

	  htmp->imiss[l] += hitcnt-XN;
	  htmp->imptmp[fr][l]->next = new MPED_FQ();
	  htmp->imptmp[fr][l] = htmp->imptmp[fr][l]->next;	
	  mptmp = htmp->imptmp[fr][l];
	  mptmp->next = NULL;
	  mptmp->fq = stmp;
	  mptmp->paired = false;
	  mptmp->opaired[0] = false;
	  mptmp->opaired[1] = false;
	  mptmp->pair = NULL;
	  mptmp->NS = 0;
	  mptmp->NE = 0;

	  D = 0;
	  I = 0;
	  n = 0;
	  do{
	    word2 = GetWordMID(n,word3);
	    if(word3[n-1]=='D'){
	      D += atoi(word2.c_str());
	    }
	    else if(word3[n-1]=='I'){
	      I += atoi(word2.c_str());
	    }
	  }while(n < word3.size());
	
	  if(XN > 0){
	    if(NS && NE){
	      mptmp->s = htmp->is[l];
	      mptmp->e = htmp->ie[l];
	      for(i=0;i+1<MD.size();i+=2){
		if(MD[i] == '0' && MD[i+1] == 'N')
		  mptmp->NS++;
		else
		  break;
	      }
	      for(i=MD.size()-1;i-1>=0;i-=2){
		if(MD[i-1] == 'N' && MD[i] == '0')
		  mptmp->NE++;
		else
		  break;
	      }
	    }
	    else if(NS){
	      mptmp->s = htmp->is[l];
	      mptmp->e = start+ts+D-I-2;
	      mptmp->NS = XN;
	    }
	    else if(NE){
	      mptmp->s = start-1;
	      mptmp->e = htmp->ie[l];
	      mptmp->NE = XN;	  
	    }
	  }
	  else{
	    mptmp->s = start-1;
	    mptmp->e = start+ts+D-I-2;	
	  }

	  if(NE && NS){
	    if(start+ts+D-I-mptmp->NE-1 == htmp->ilen[l]-lset->rpos[l]){
	      if(reads[ts-mptmp->NE-2] == 'A' && reads[ts-mptmp->NE-1] == 'G'){
		fqctmp->irpos[ts-mptmp->NE]++;
	      }
	    }
	    if(start+mptmp->NS-1==lset->lpos[l]){
	      if(reads[mptmp->NS] == 'G' && reads[mptmp->NS+1] == 'T'){	    
		fqctmp->ilpos[mptmp->NS-1]++;
	      }
	    }
	  }
	  else if(NE){
	    if(start+ts+D-I-XN-1 == htmp->ilen[l]-lset->rpos[l]){
	      if(reads[ts-XN-2] == 'A' && reads[ts-XN-1] == 'G'){
		fqctmp->irpos[ts-XN]++;
	      }
	    }
	  }      
	  else if(NS){
	    if(start+XN-1==lset->lpos[l]){
	      if(reads[XN] == 'G' && reads[XN+1] == 'T'){	    
		fqctmp->ilpos[XN-1]++;
	      }
	    }
	  }
	}
      }
          
      int maxpcnt,maxid;
      fqctmp = fqctop;
      while(fqctmp->next){
	fqctmp = fqctmp->next;
	stmp = fqctmp->stmp;
	//2-Jan-2015
	gltmp = stmp->gl[ii];
	while(gltmp->next){
	  gltmp = gltmp->next;
	}

	ts = fqctmp->stmp->code[ii].size();
	maxpcnt = 0;
	for(s=0;s<ts;s++){
	  if(fqctmp->irpos[s] > maxpcnt){
	    maxpcnt = fqctmp->irpos[s];
	    maxid = s;
	  }
	}

	hit3 = true;
	
	if(maxpcnt > 0){
	  hit3 = false;
	  
	  //2-Jan-2015
	  gltmp->next = new GENE_LIST();
	  gltmp = gltmp->next;
	  gltmp->next = NULL;
	  gltmp->name = gname;
	  gltmp->einame = lset->exname[exidr];
	  if(gltmp->einame.substr(0,4) == "exon"){
	    gltmp->einame[0] = 'E';
	  }
	  gltmp->freq = 0;

	  if(convert_read){
	    for(i=0;i<hcnt;i++){
	      chtmp = che[exidr][i];
	      while(chtmp->next){
		chtmp = chtmp->next;
		htmp = hh[chtmp->hid[0]];
		while(htmp->next){
		  htmp = htmp->next;
		  if(htmp->id == chtmp->name[0])
		    break;
		}
		
		hit = true;
		k = htmp->es[exidr];
		for(j=maxid;j<ts;j++){
		  if(k > htmp->ee[exidr])
		    break;
		  if(stmp->code[ii][j] != htmp->code[exidr][k]){
		    hit = false;
		    break;
		  }
		  k++;
		}
		
		
		if(hit){
		  hit3 = true;
		 
		  for(j=0;j<chtmp->cnt;j++){
		    htmp = hh[chtmp->hid[j]];
		    while(htmp->next){
		      htmp = htmp->next;
		      if(htmp->id == chtmp->name[j]){
			break;
		      }
		    }

		    /*
		    hit = false;		      
		    mptmp = htmp->mp[ii][exidr];
		    while(mptmp->next){
		      mptmp = mptmp->next;
		      if(mptmp->fq == stmp){
			hit = true;
			break;
		      }
		    }		      	      
		    if(hit){
		      continue;
		    }
		    */
		    
		    htmp->mptmp[ii][exidr]->next = new MPED_FQ();
		    htmp->mptmp[ii][exidr] = htmp->mptmp[ii][exidr]->next;
		    mptmp = htmp->mptmp[ii][exidr];		
		    mptmp->next = NULL;
		    mptmp->fq = stmp;
		    mptmp->paired = false;
		    mptmp->opaired[0] = false;
		    mptmp->opaired[1] = false;
		    mptmp->pair = NULL;
		    mptmp->s = htmp->es[exidr];
		    mptmp->e = k-1;
		    mptmp->NS=maxid;
		    mptmp->NE=0;
		  }
		  gltmp->freq += chtmp->cnt;
				  
		}
		
	      }
	    }
	  }
	  else{	    
	    for(i=0;i<hcnt;i++){
	      htmp = hh[i];
	      while(htmp->next){
		htmp = htmp->next;
		if(htmp->hit[exidr]){
		  hit = true;
		  k = htmp->es[exidr];
		  for(j=maxid;j<ts;j++){
		    if(k > htmp->ee[exidr])
		      break;
		    if(stmp->code[ii][j] != htmp->code[exidr][k]){
		      hit = false;
		      break;
		    }
		    k++;
		  }
	       
		  if(hit){
		    hit3 = true;
		    htmp->mptmp[ii][exidr]->next = new MPED_FQ();
		    htmp->mptmp[ii][exidr] = htmp->mptmp[ii][exidr]->next;
		    mptmp = htmp->mptmp[ii][exidr];		    
		    mptmp->next = NULL;
		    mptmp->fq = stmp;
		    mptmp->paired = false;
		    mptmp->opaired[0] = false;
		    mptmp->opaired[1] = false;		    
		    mptmp->pair = NULL;
		    mptmp->s = htmp->es[exidr];
		    mptmp->e = k-1;
		    mptmp->NS=maxid;
		    mptmp->NE=0;
		  
		    gltmp->freq++;
		  }
		}
	      }
	    }	   
	  }
	}
	
	maxpcnt = 0;
	for(s=ts-1;s>=0;s--){
	  if(fqctmp->ilpos[s] > maxpcnt){
	    maxpcnt = fqctmp->ilpos[s];
	    maxid = s;
	  }
	}
	
	if(maxpcnt > 0){
	  hit3 = false;
	  
	  gltmp->next = new GENE_LIST();
	  gltmp = gltmp->next;
	  gltmp->next = NULL;
	  gltmp->name = gname;
	  gltmp->einame = lset->exname[exidl];
	  if(gltmp->einame.substr(0,4) == "exon"){
	    gltmp->einame[0] = 'E';
	  }
	  gltmp->freq = 0;

	  if(convert_read){
	    for(i=0;i<hcnt;i++){
	      chtmp = che[exidl][i];
	      while(chtmp->next){
		chtmp = chtmp->next;
		htmp = hh[chtmp->hid[0]];
		while(htmp->next){
		  htmp = htmp->next;
		  if(htmp->id == chtmp->name[0])
		    break;
		}
		
		hit = true;
		k = htmp->ee[exidl];
		for(j=maxid;j>=0;j--){
		  if(k < htmp->es[exidl])
		    break;
		  if(stmp->code[ii][j] != htmp->code[exidl][k]){
		    hit = false;
		    break;
		  }
		  k--;
		}
		
		if(hit){
		  hit3 = true;

		  for(j=0;j<chtmp->cnt;j++){
		    htmp = hh[chtmp->hid[j]];
		    while(htmp->next){
		      htmp = htmp->next;
		      if(htmp->id == chtmp->name[j]){
			break;
		      }
		    }
		    	    
		    hit = false;
		    /*
		    mptmp = htmp->mp[ii][exidl];
		    while(mptmp->next){
		      mptmp = mptmp->next;
		      if(mptmp->fq == stmp){
			hit = true;
			break;
		      }
		    }		      	      
		    if(hit){
		      continue;
		    }
		    */
		    
		    htmp->mptmp[ii][exidl]->next = new MPED_FQ();
		    htmp->mptmp[ii][exidl] =  htmp->mptmp[ii][exidl]->next;
		    mptmp = htmp->mptmp[ii][exidl];
		    mptmp->next = NULL;
		    mptmp->fq = stmp;
		    mptmp->paired = false;
		    mptmp->opaired[0] = false;
		    mptmp->opaired[1] = false;
		    mptmp->pair = NULL;
		    mptmp->s = k+1;
		    mptmp->e = htmp->ee[exidl];
		    mptmp->NS=0;
		    mptmp->NE=ts-maxid-1;
		  }
		  gltmp->freq += chtmp->cnt;		    
		  		  
		}
		
	      }
	    }
	  }
	  else{	  
	    for(i=0;i<hcnt;i++){	    
	      htmp = hh[i];
	      while(htmp->next){
		htmp = htmp->next;
		if(htmp->hit[exidl]){
		  hit = true;
		  k = htmp->ee[exidl];
		  for(j=maxid;j>=0;j--){
		    if(k < htmp->es[exidl])
		      break;
		    if(stmp->code[ii][j] != htmp->code[exidl][k]){
		      hit = false;
		      break;
		    }
		    k--;
		  }
	
		  
		  if(hit){
		    hit3 = true;
		    htmp->mptmp[ii][exidl]->next = new MPED_FQ();
		    htmp->mptmp[ii][exidl] = htmp->mptmp[ii][exidl]->next;
		    mptmp = htmp->mptmp[ii][exidl];
		    mptmp->next = NULL;
		    mptmp->fq = stmp;
		    mptmp->paired = false;
		    mptmp->opaired[0] = false;
		    mptmp->opaired[1] = false;
		    mptmp->pair = NULL;
		    mptmp->s = k+1;
		    mptmp->e = htmp->ee[exidl];
		    mptmp->NS=0;
		    mptmp->NE=ts-maxid-1;
		    gltmp->freq++;
		  }
		  
		}
	      }
	    }	
	  }  	 
	}

	if(!hit3){
	  for(i=0;i<2;i++)
	    stmp->w[i] = -1.0;
	}else{
	  fqctmp->hit = true;
	}
		
      }

      fqctmp = fqctop;
      while(fqctmp->next){
	fqcback = fqctmp;
	fqctmp = fqctmp->next;
	if(fqctmp->hit){
	  fqctmp->stmp->ehit[ii] = true;
	}
	delete fqcback;
      }
      delete fqctmp;
      fqctop = NULL;
           
      ins.close();	
    }
  }
  
  return hlacnt;
}


