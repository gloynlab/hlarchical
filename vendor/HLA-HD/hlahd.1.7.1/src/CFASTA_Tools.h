int N_to_I(char a){
  if(a == 'A' || a == 'a'){
    return 0;
  }
  else if(a == 'T' || a == 't'){
    return 1;
  }
  else if(a == 'G' || a == 'g'){
    return 2;
  }
  else if(a == 'C' || a == 'c'){
    return 3;
  }
  else{
   return -1;
  }
}

string UpperS(string code){
  int i;
  string rcode = code;
  for(i=0;i<code.size();i++){
    if('a' <= code[i] && code[i] <= 'z'){
      rcode[i] = code[i] - 32;
    }
    else
      rcode[i] = code[i];
  }
  return rcode;
}

string Trans_U_T(string code){
  int i;
  string rcode = code;
  for(i=0;i<code.size();i++){
    if(code[i] == 'U')
      rcode[i] = 'T';
    else if(code[i] == 'u')
      rcode[i] = 't';
  }
  return rcode;
}

char UpperC(char c){
  int i;
  
  if('a' <= c && c <= 'z'){
    return c - 32;
  }
  else
    return c;
}


string deci_to_st(long int a){
    string ch[10];
    ch[0] = "0"; ch[1] = "1"; ch[2] = "2"; ch[3] = "3"; ch[4] = "4"; ch[5] = "5";
    ch[6] = "6"; ch[7] = "7"; ch[8] = "8"; ch[9] = "9";
    long int deci;
    string st;
    int k,l;

    if(a == 0){
      st = "0";
    }
    else{
      deci = 1;
      k = 0;
      while(a/deci >= 1){
        deci = deci*10;
        k++;
      }
      deci = deci/10;
      st = "";
      
      for(l=0;l<k;l++){
        st += ch[a/deci];
        a = a % deci;
        deci /= 10;
      }
    }
    return st;
  }


string GetWordtab(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != '\t'){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  return p;
}

string GetWordsp(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != ' '){    
    p += ssline[n];
    i++;
    n++;
  } 
  n++;
  while(n < ssline.size() &&
	ssline[n] == ' '){      
    n++;
  }    
  return p;
}

string GetWordcol(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != ':'){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  return p;
}

string GetWorddq(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != '"'){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  return p;
}

string GetWordscol(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != ';'){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  while(n < ssline.size() &&
	ssline[n] == ' '){      
    n++;
  }    
  return p;
}

string GetWordast(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != '*'){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  while(n < ssline.size() &&
	ssline[n] == ' '){      
    n++;
  }    
  return p;
}

string GetWordbar(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != '-'){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  while(n < ssline.size() &&
	ssline[n] == ' '){      
    n++;
  }    
  return p;
}


string GetWordcam(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != ','){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  while(n < ssline.size() &&
	ssline[n] == ' '){      
    n++;
  }   
  return p;
}

string GetWorddot(int &n,string ssline)
{
  int i; string p;
  i = 0;
  while(n < ssline.size() &&
	ssline[n] != '.'){    
    p += ssline[n];
    i++;
    n++;
  }    
  n++;
  while(n < ssline.size() &&
	ssline[n] == ' '){      
    n++;
  }   
  return p;
}
