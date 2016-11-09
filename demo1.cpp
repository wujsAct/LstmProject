#include "lstm_all.h"

void example(){
  int mem_cell_dim = 100; int x_dim = 50; int concat_len = mem_cell_dim+x_dim;
  
  LSTMParam param = LSTMParam(mem_cell_dim,x_dim);
  LSTMNetwork lstm_net = LSTMNetwork(&param);
  cout<<"right lstm_net"<<endl;
  vector< vector<double> > y_list;
  vector<double> y;
  y.push_back(-0.5);y.push_back(0.2);y.push_back(0.1);y.push_back(-0.5);
  for(size_t i=0;i<4;i++){
    vector<double> temp;
    temp.push_back(y[i]);
    y_list.push_back(temp);
  }
  cout<<"right y_list"<<endl;
  
  LossLayer lossLayer = LossLayer();
  cout<<"right lossLayer"<<endl;
  
  vector< vector<double> > x_list = uniform_ditr_init(4,x_dim);
  double lr=0.1;
  cout<<"right x_list"<<endl;
  
  for(int cur_iter=0;cur_iter<100;cur_iter++){
    //cout<<"cur_iter: "<<cur_iter<<endl;
    int i=0;
    for(auto x : x_list){
      lstm_net.x_list_add(x);
      //cout<<"y_pred["<<i<<"]:"<<(lstm_net.get_lstm_node_list()[i].get_state()->get_h())[0]<<endl;;
      i++;
    }
    
    double loss = lstm_net.y_list_is(y_list,lossLayer);
    if(cur_iter%10==0){ 
      cout<<"loss:"<<loss<<endl;
    }
    param.apply_diff(lr);
    lstm_net.x_list_clear();
  }
}
int main(){
  example();
}

