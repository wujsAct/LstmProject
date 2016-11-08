#include "lstm_all.h"

LSTMNode::LSTMNode(int mem_cell_dim,int x_dim){
  this->param = new LSTMParam(mem_cell_dim, x_dim);
  this->state = new LSTMState(mem_cell_dim, x_dim);
}

void LSTMNode::bottom_data_is(vector<double> x, vector<double> s_prev, vector<double> h_prev){
  if(s_prev.size()==0) {s_prev=zeros_init_like(this->state->get_s());}
  if(h_prev.size()==0) {h_prev=zeros_init_like(this->state->get_h());}
  
  this->s_prev = s_prev;
  this->h_prev = h_prev;
  
  //concatenate x(t) and h(t-1)
  xc = concatenate(x,h_prev);
  this->state->set_g(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wg(),xc),this->param->get_bg()),"tanh"));
  this->state->set_i(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wi(),xc),this->param->get_bi()),"sigmoid"));
  this->state->set_f(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wf(),xc),this->param->get_bg()),"sigmoid"));
  this->state->set_o(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wo(),xc),this->param->get_bg()),"sigmoid"));
  
  this->state->set_s(vecA_add_vecB(vecA_mul_vecB(this->state->get_g(),this->state->get_i()),vecA_mul_vecB(s_prev,this->state->get_f())));
  this->state->set_h(vecA_mul_vecB(this->state->get_s(),this->state->get_o()));
  
  this->x = x;
  this->xc = xc;
}

