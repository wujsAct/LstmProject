#include "lstm_all.h"

LSTMNetwork::LSTMNetwork(LSTMParam* param){
  this->param = param;
}

double LSTMNetwork::y_list_is(vector< vector<double> > y_list,LossLayer lossLayer){
  if(y_list.size()!=this->x_list.size()){
    //cout<<"y_list_is: x and y dimension is mismatched!"<<endl;
    exit(-1);
  }
  
  int idx = this->x_list.size()-1;
  //first node only gets diffs from label.
  double loss = lossLayer.loss(this->lstm_node_list[idx].get_state()->get_h(),y_list[idx]);
  //cout<<"loss is right"<<endl;
  
  vector<double> diff_h = lossLayer.bottom_diff(this->lstm_node_list[idx].get_state()->get_h(),y_list[idx]);
  //cout<<"diff_h is right"<<endl;
  
  //here s has no effects on loss due to h(t+1), hence we set it equal to zero
  vector<double> diff_s = zeros_init(this->param->get_mem_cell_dim());
  //cout<<"diff_s is right"<<endl;
  
  this->lstm_node_list[idx].top_diff_is(diff_h,diff_s);
  
  //cout<<"lstm_node_list is right"<<endl;
  
  idx -=1;
  
  while(idx>=0){
    //cout<<"idx:"<<idx<<endl;
    loss += lossLayer.loss(this->lstm_node_list[idx].get_state()->get_h(),y_list[idx]);
    diff_h = lossLayer.bottom_diff(this->lstm_node_list[idx].get_state()->get_h(),y_list[idx]);
    //cout<<"idx diff_h size"<<diff_h.size()<<endl;
    //cout<<"this->lstm_node_list[idx+1].get_state()->get_bottom_diff_h() size:"<<this->lstm_node_list[idx+1].get_state()->get_bottom_diff_h().size()<<endl;
    diff_h = vecA_add_vecB(diff_h,this->lstm_node_list[idx+1].get_state()->get_bottom_diff_h());
    //cout<<"idx diff_h right"<<endl;
    diff_s = vecA_add_vecB(diff_s,this->lstm_node_list[idx+1].get_state()->get_bottom_diff_s());
    //cout<<"idx diff_s right"<<endl;
    this->lstm_node_list[idx].top_diff_is(diff_h,diff_s);
    idx -=1;
  }
  
  return loss;
}

void LSTMNetwork::x_list_clear(){
  this->x_list.clear();
  //cout << "x_list capacity" << this->x_list.capacity() << endl;
  vector< vector<double> >(this->x_list).swap(this->x_list);
  //cout << "x_list capacity" << this->x_list.capacity() << endl;
}

void LSTMNetwork::x_list_add(vector<double> x){
  //cout<<"x_list_add:"<<endl;
  for(size_t i=0;i<x.size();i++){
    //cout<<x[i]<<"\t";
  }
  //cout<<endl;
  
  this->x_list.push_back(x);
  
  if(this->x_list.size() >= this->lstm_node_list.size()){  //we need to add new lstm node, create new state men
    LSTMState* lstm_state = new LSTMState(this->param->get_mem_cell_dim(),this->param->get_x_dim());
    this->lstm_node_list.push_back(LSTMNode(this->param,lstm_state));
  }
  //cout<<"x_list push right...."<<endl;
  
  //get index of most recent x input
  int idx = this->x_list.size() - 1;
  if(idx ==0 ){
    //no recurrent inputs yet
    //cout<<"idx...."<<idx<<endl;
    vector<double> temps,temph;
    this->lstm_node_list[idx].bottom_data_is(x,temps,temph);
    //cout<<"idx out"<<idx<<endl;
  }
  else{
    vector<double> s_prev = this->lstm_node_list[idx-1].get_state()->get_s();
    vector<double> h_prev = this->lstm_node_list[idx-1].get_state()->get_h();
    this->lstm_node_list[idx].bottom_data_is(x,s_prev,h_prev); 
  }
}

vector<LSTMNode> LSTMNetwork::get_lstm_node_list(){
  return this->lstm_node_list;
}

vector< vector<double> > LSTMNetwork::get_x_list(){
  return this->x_list;
}
