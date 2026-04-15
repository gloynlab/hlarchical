  string GetWordtab(int &n,string ssline)
  {
    string p;
    int n2 = n;
    int s = ssline.size();
    while(n < s &&
          ssline[n] != '\t'){    
      n++;
    }    
    if(n-n2 > 0)
      p = ssline.substr(n2,n-n2);
    n++;
    return p;
  }

string GetWordsp(int &n,string ssline)
  {
    string p;
    int n2 = n;
    int s = ssline.size();
    while(n < s &&
          ssline[n] != ' '){    
      n++;
    } 
    if(n-n2 > 0)  
      p = ssline.substr(n2,n-n2); 
    n++;
    return p;
  }

  string GetWordcol(int &n,string ssline)
  {
    string p;
    int n2 = n;
    int s = ssline.size();
    while(n < s &&
          ssline[n] != ':'){    
      n++;
    } 
    if(n-n2 > 0)   
      p = ssline.substr(n2,n-n2); 
    n++;
    return p;
  }

  string GetWordcam(int &n,string ssline)
  {
      string p;
    int n2 = n;
    int s = ssline.size();
    while(n < s &&
          ssline[n] != ','){    
      n++;
    } 
    if(n-n2 >0)   
      p = ssline.substr(n2,n-n2); 
    n++;
    return p;
  }

 string GetWordast(int &n,string ssline)
  {
    string p;
    int n2 = n;
    int s = ssline.size();
    while(n < s &&
          ssline[n] != '*'){    
      n++;
    }    
    if(n-n2 > 0)
      p = ssline.substr(n2,n-n2); 
    n++;
    return p;
  }
