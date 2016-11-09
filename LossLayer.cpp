#include "lstm_all.h"

LossLayer::LossLayer(){}

double LossLayer::loss(vector<double> pred, vector<double> label){
  double square_loss =0;
//  for(size_t i=0; i<pred.size();i++){
//    square_loss += pow(pred[i]-label[i],2);
//  }
  square_loss += pow(pred[0]-label[0],2);
  //cout<<"loss right"<<endl;
  return square_loss;
}

vector<double> LossLayer::bottom_diff(vector<double> pred, vector<double> label){
  vector<double> diff= zeros_init_like(pred);
  //cout<<"diff size:"<<diff.size()<<endl;
  //cout<<"pred size:"<<label.size()<<endl;
//  diff = vecA_minus_vecB(pred,label);
  diff[0] = 2*(pred[0]-label[0]);
  return diff;
}