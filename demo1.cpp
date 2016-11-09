#include "lstm_all.h"

int main(){
  vector<double> c;
  vector<double> t; t.push_back(1);t.push_back(2);
  c = t;
  t.push_back(4);
  vector<double> c1 = sub_vector(c,0,2);
  for(size_t i=0; i<c1.size();i++){
    cout<<c1[i]<<endl;
  }
}
