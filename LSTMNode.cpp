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

void LSTMNode::top_diff_is(vector<double> top_diff_h, vector<double> top_diff_s){
  vector<double> d_s = vecA_add_vecB( vecA_mul_vecB(this->state->get_o(),top_diff_h), top_diff_s);
  vector<double> d_o = vecA_mul_vecB(this->state->get_s(),top_diff_h);
  vector<double> d_i = vecA_mul_vecB(this->state->get_g(),d_s);
  vector<double> d_g = vecA_mul_vecB(this->state->get_i(),d_s);
  vector<double> d_f = vecA_mul_vecB(this->s_prev,d_s);
  
  //diffs w.r.t vector inside sigma/tanh function
  vector<double> di_input = vecA_mul_vecB(vecA_mul_vecB(num_minus_vec(1,this->state->get_i()), this->state->get_i()),d_i);
  vector<double> df_input = vecA_mul_vecB(vecA_mul_vecB(num_minus_vec(1,this->state->get_f()), this->state->get_f()),d_f);
  vector<double> do_input = vecA_mul_vecB(vecA_mul_vecB(num_minus_vec(1,this->state->get_o()), this->state->get_o()),d_o);
  vector<double> dg_input = vecA_mul_vecB(num_minus_vec(1,vecA_mul_vecB(this->state->get_o(),this->state->get_o())),d_g);
  
  //diffs w.r.t inputs
  this->param->set_wi_diff
}

