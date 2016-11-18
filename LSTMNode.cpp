#include "lstm_all.h"

LSTMNode::LSTMNode(LSTMParam* param,LSTMState* state){
  this->param = param;
  this->state = state;
}

void LSTMNode::bottom_data_is(vector<double> x, vector<double> s_prev, vector<double> h_prev){
  //cout<<"s_prev.size():"<<s_prev.size()<<endl;
  //cout<<"h_prev.size():"<<h_prev.size()<<endl;
  if(s_prev.size()==0) {s_prev=zeros_init_like(this->state->get_s());}
  if(h_prev.size()==0) {h_prev=zeros_init_like(this->state->get_h());}
  
  this->s_prev = s_prev;
  this->h_prev = h_prev;
  
  //cout<<"s_prev.size():"<<s_prev.size()<<endl;
  //cout<<"h_prev.size():"<<h_prev.size()<<endl;
  
  //concatenate x(t) and h(t-1)
  xc = concatenate(x,h_prev);
  //cout<<"x.size():"<<x.size()<<endl;
  //cout<<"h_prev.size():"<<h_prev.size()<<endl;
  //cout<<"xc_prev.size():"<<this->xc.size()<<endl;
  
  this->state->set_g(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wg(),xc),this->param->get_bg()),"tanh"));
  this->state->set_i(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wi(),xc),this->param->get_bi()),"sigmoid"));
  this->state->set_f(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wf(),xc),this->param->get_bf()),"sigmoid"));
  this->state->set_o(vec_nonlinear(vecA_add_vecB(mat_dot_vec(this->param->get_wo(),xc),this->param->get_bo()),"sigmoid"));
  
  this->state->set_s(vecA_add_vecB(vecA_mul_vecB(this->state->get_g(),this->state->get_i()),vecA_mul_vecB(s_prev,this->state->get_f())));
  this->state->set_h(vecA_mul_vecB(this->state->get_s(),this->state->get_o()));
  
  this->x = x;
  this->xc = xc;
}

void LSTMNode::top_diff_is(vector<double> top_diff_h, vector<double> top_diff_s,LSTMNode prev_node, string type){
  vector<double> d_s = vecA_add_vecB( vecA_mul_vecB(this->state->get_o(),top_diff_h), top_diff_s);
  vector<double> d_o = vecA_mul_vecB(this->state->get_s(),top_diff_h);
  vector<double> d_i = vecA_mul_vecB(this->state->get_g(),d_s);
  vector<double> d_g = vecA_mul_vecB(this->state->get_i(),d_s);
  vector<double> d_f = vecA_mul_vecB(this->s_prev,d_s);
  //cout<<"top_diff_is first part is right!"<<endl;
  vector<double> di_input = vecA_mul_vecB(vecA_mul_vecB(num_minus_vec(1,this->state->get_i()), this->state->get_i()),d_i);
  vector<double> df_input = vecA_mul_vecB(vecA_mul_vecB(num_minus_vec(1,this->state->get_f()), this->state->get_f()),d_f);
  vector<double> do_input = vecA_mul_vecB(vecA_mul_vecB(num_minus_vec(1,this->state->get_o()), this->state->get_o()),d_o);
  vector<double> dg_input = vecA_mul_vecB(num_minus_vec(1,vecA_mul_vecB(this->state->get_g(),this->state->get_g())),d_g);
  //cout<<"top_diff_is second part is right!"<<endl;
    
  //diffs w.r.t inputs
  //diffs w.r.t vector inside sigma/tanh function
  this->param->set_wi_diff(matA_add_matB(this->param->get_wi_diff(), vecA_outer_vecB(di_input, this->xc)));
  this->param->set_wf_diff(matA_add_matB(this->param->get_wf_diff(), vecA_outer_vecB(df_input, this->xc)));
  this->param->set_wo_diff(matA_add_matB(this->param->get_wo_diff(), vecA_outer_vecB(do_input, this->xc)));
  this->param->set_wg_diff(matA_add_matB(this->param->get_wg_diff(), vecA_outer_vecB(dg_input, this->xc)));
    
  this->param->set_bi_diff(vecA_add_vecB(this->param->get_bi_diff(),di_input));
  this->param->set_bf_diff(vecA_add_vecB(this->param->get_bf_diff(),df_input));
  this->param->set_bo_diff(vecA_add_vecB(this->param->get_bo_diff(),do_input));
  this->param->set_bg_diff(vecA_add_vecB(this->param->get_bg_diff(),dg_input));
    
  if(type.compare("dot")==0){
    vector<double> di_input_h = vec_multiply_mat(vecA_mul_vecB(num_minus_vec(1,this->state->get_i()), this->state->get_i()),this->param->get_wi());
    vector<double> df_input_h = vec_multiply_mat(vecA_mul_vecB(num_minus_vec(1,this->state->get_f()), this->state->get_f()),this->param->get_wf());
    vector<double> do_input_h = vec_multiply_mat(vecA_mul_vecB(num_minus_vec(1,this->state->get_o()), this->state->get_o()),this->param->get_wo());
    vector<double> dg_input_h = vec_multiply_mat(num_minus_vec(1,vecA_mul_vecB(this->state->get_g(),this->state->get_g())),this->param->get_wg());
    vector<double> dxc = zeros_init_like(this->xc);
    
    di_input_h = vecA_mul_vecB(this->param->get_g(),di_input_h);
    dg_input_h = vecA_mul_vecB(this->param->get_i(),dg_input_h);
    df_input_h = vecA_mul_vecB(prev_node->param->get_s(),df_input_h);
    
    do_input_h = vecA_mul_vecB(this->param->get_s(),do_input_h);
    
    //compute bottom diff
    vector<double> dxc = zeros_init_like(this->xc);
    dxc = vecA_mul_vecB(this->param->get_o(),vecA_add_vecB(di_input_h,vecA_add_vecB(dg_input_h,df_input_h)));
    dxc = vecA_add_vecB(dxc,do_input_h);
    
    dxc = vecA_mul_vecB(top_diff_h,dxc);
    //save bottom diffs
    this->state->set_bottom_diff_x(sub_vector(dxc,0,this->param->get_x_dim()));
    this->state->set_bottom_diff_h(sub_vector(dxc,this->param->get_x_dim(),dxc.size()));
  }
  else{
    //compute bottom diff
    vector<double> dxc = zeros_init_like(this->xc);
    dxc = vecA_add_vecB(dxc,mat_dot_vec(mat_transpose(this->param->get_wi()),di_input));
    dxc = vecA_add_vecB(dxc,mat_dot_vec(mat_transpose(this->param->get_wf()),df_input));
    dxc = vecA_add_vecB(dxc,mat_dot_vec(mat_transpose(this->param->get_wo()),do_input));
    dxc = vecA_add_vecB(dxc,mat_dot_vec(mat_transpose(this->param->get_wg()),dg_input));
    
    //save bottom diffs
    this->state->set_bottom_diff_x(sub_vector(dxc,0,this->param->get_x_dim()));
    this->state->set_bottom_diff_h(sub_vector(dxc,this->param->get_x_dim(),dxc.size()));
  }
  this->state->set_bottom_diff_s(vecA_mul_vecB(d_s,this->state->get_f()));
  //cout<<"top_diff_is final part is right!"<<endl;
}

LSTMParam* LSTMNode::get_param(){
  return this->param;
}

LSTMState* LSTMNode::get_state(){
  return this->state;
}
