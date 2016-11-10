#include "lstm_all.h"

double sigmoid(double num){
  return 1.0/(1+exp(num*(-1.0)));
}

vector<double> vec_nonlinear(vector<double> v,string fun="sigmoid"){
  if(v.size()==0){
    cout<<"vector is none, cannot do tanh"<<endl;
    exit(-1);
  }
  vector<double> ret;
  double num=0;
  for(size_t i=0;i<v.size();i++){
    if(fun.compare("tanh")==0){
      num = tanh(v[i]);
    }
    if(fun.compare("sigmoid")==0){
      num = sigmoid(v[i]);
    }
    ret.push_back(num);
  }
  return ret;
}


