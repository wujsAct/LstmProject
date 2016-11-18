#include "lstm_all.h"

LossLayer::LossLayer(){}

//flag = 0 表示为正例
double LossLayer::loss(vector<double> pred, vector<double> label, string type,int flag){
  
  if(pred.size() != label.size()){
    cout<<"pred and label size is mismatch, can not compute loss"<<endl;
    exit(-1);
  }
  double loss =0;
//  for(size_t i=0; i<pred.size();i++){
//    square_loss += pow(pred[i]-label[i],2);
//  }
  //
  //minimize negative log likelihodd
  if(type.compare("NLL")==0){
    for(size_t i = 0;i< label.size();i++)
      loss += -( label[i]*log(pred[i]) + (1-label[i])*log(1-pred[i]) );
    loss /= label.size();
  }
  else if(type.compare("dot")==0){
    //label[0]代表是正例还是反例！
    if(flag==0){
      loss +=(-sigmoid(vec_dot_vec(pred,label)));
    }
    else{
      loss +=(sigmoid(vec_dot_vec(pred,label))-1);
    }
  }
  else{
    loss = pow(pred[0]-label[0],2);
  }
  return loss;
}

vector<double> LossLayer::bottom_diff(vector<double> pred, vector<double> label,string type,string flag="dot"){
  if(pred.size() != label.size()){
    cout<<"pred and label size is mismatch, can not compute loss"<<endl;
    exit(-1);
  }
  vector<double> diff= zeros_init_like(pred);
  
  //cout<<"diff size:"<<diff.size()<<endl;
  //cout<<"pred size:"<<label.size()<<endl;
  //diff = vecA_minus_vecB(pred,label);
  if(type.compare("NLL")==0){
    for(size_t i = 0;i< label.size();i++)
      diff[i] = -1.0/label.size() *( label[i]/pred[i] - (1-label[i])/(1-pred[i]) );
  }
  else if(type.compare("dot")==0){
    if(flag==0)
    {
      for(size_t i = 0;i< label.size();i++)
          diff[i] = -label[i];
    }
    else{
      for(size_t i = 0;i< label.size();i++)
          diff[i] = label[i];
    }
  }else{
    diff[0] = 2*(pred[0]-label[0]);
  }
  return diff;
}