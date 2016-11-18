#ifndef LSTM_H
#define LSTM_H
#include <iostream>
#include <string>
#include <random>
#include <ctime>
#include <cmath>
using namespace std;

//initialize_matrix.cpp
vector< vector<double> > uniform_ditr_init(int row,int column);
vector<double> uniform_ditr_init(int row);
vector< vector<double> > zeros_init(int row,int column);
vector<double> zeros_init(int row);
vector< vector<double> >  zeros_init_like(vector< vector<double> > m);
vector<double>  zeros_init_like(vector<double>  m);
vector<double> vecA_minus_vecB(vector<double> vecA, vector<double> vecB);
vector<double> vecA_add_vecB(vector<double> vecA, vector<double> vecB);
vector< vector<double> > matA_minus_matB(vector< vector<double> > matA, vector< vector<double> > matB);
vector< vector<double> > matA_add_matB(vector< vector<double> > matA, vector< vector<double> > matB);
vector<double> vec_multiply_num(vector<double> vecA, double num);
vector< vector<double> > mat_multiply_num(vector< vector<double> > matA, double num);
vector<double> concatenate(vector<double> m1, vector<double> m2);
vector<double> mat_dot_vec(vector< vector<double> > m, vector<double> v);
double vec_dot_vec(vector<double> v1, vector<double> v2);
vector<double> vecA_mul_vecB(vector<double> v1, vector<double> v2);
vector<double> num_minus_vec(double num, vector<double> v);
vector< vector<double> > vecA_outer_vecB(vector<double> v1, vector<double> v2);
vector <vector<double>> mat_transpose(vector< vector<double> > m);
vector<double> sub_vector(vector<double> v, int start, int end);

//vector nonlinear transformation
double sigmoid(double num);
vector<double> vec_nonlinear(vector<double> v,string fun);


class LSTMParam{
  private:
    int mem_cell_dim; //hidden dimension
    int x_dim; //input data dimension
    int concat_dim;
    //weight matrix
    vector< vector<double> > wg,wi,wf,wo;
    //bias terms
    vector<double> bg,bi,bf,bo;
    //diffs(derivative loss of function w.r.t all parameters)
    vector< vector<double> > wg_diff,wi_diff,wf_diff,wo_diff;
    vector<double> bg_diff,bi_diff,bf_diff,bo_diff;
    
  public:
    LSTMParam(int mem_cell_dim, int x_dim); //constructor
    void apply_diff(double lr);
    double get_x_dim(); double get_mem_cell_dim();
    vector< vector<double> > get_wg(); vector< vector<double> > get_wg_diff(); vector<double> get_bg(); vector<double> get_bg_diff();
    vector< vector<double> > get_wi(); vector< vector<double> > get_wi_diff(); vector<double> get_bi(); vector<double> get_bi_diff();
    vector< vector<double> > get_wf(); vector< vector<double> > get_wf_diff(); vector<double> get_bf(); vector<double> get_bf_diff();
    vector< vector<double> > get_wo(); vector< vector<double> > get_wo_diff(); vector<double> get_bo(); vector<double> get_bo_diff();
    
    void set_wg(vector< vector<double> >); void set_wg_diff(vector< vector<double> > ); void set_bg(vector<double>); void set_bg_diff(vector<double>);
    void set_wi(vector< vector<double> >); void set_wi_diff(vector< vector<double> > ); void set_bi(vector<double>); void set_bi_diff(vector<double>);
    void set_wf(vector< vector<double> >); void set_wf_diff(vector< vector<double> > ); void set_bf(vector<double>); void set_bf_diff(vector<double>);
    void set_wo(vector< vector<double> >); void set_wo_diff(vector< vector<double> > ); void set_bo(vector<double>); void set_bo_diff(vector<double>);
};

class LSTMState{
  private:
    int mem_cell_dim; //hidden dimension
    int x_dim; //input data dimension
    vector<double> g,i,f,o,s,h;
    vector<double> bottom_diff_h,bottom_diff_s,bottom_diff_x;
  public:
    LSTMState(int mem_cell_dim, int x_dim); //constructor
    vector<double> get_g(); void set_g(vector<double>);
    vector<double> get_i(); void set_i(vector<double>);
    vector<double> get_f(); void set_f(vector<double>);
    vector<double> get_o(); void set_o(vector<double>);
    vector<double> get_s(); void set_s(vector<double>);
    vector<double> get_h(); void set_h(vector<double>);
    vector<double> get_bottom_diff_h(); void set_bottom_diff_h(vector<double>);
    vector<double> get_bottom_diff_s(); void set_bottom_diff_s(vector<double>);
    vector<double> get_bottom_diff_x(); void set_bottom_diff_x(vector<double>);
};

class LSTMNode{
  private:
    LSTMParam *param;
    LSTMState *state;

    vector<double> x;  //non-recurrent input to node;
    vector<double> xc; //non-recurrent input concatenated with recurrent input
    
    vector<double> s_prev,h_prev; //save data to use in backprop
    
  public:
    LSTMNode(LSTMParam *,LSTMState *);
    void bottom_data_is(vector<double> x, vector<double> s_prev, vector<double> h_prev);
    void top_diff_is(vector<double> top_diff_h, vector<double> top_diff_s);
    LSTMParam* get_param(); LSTMState* get_state();
};

class LossLayer{
  public:
    //simple loss
    LossLayer();
    double loss(vector<double> pred, vector<double> label,string str,int flag=0);
    vector<double> bottom_diff(vector<double> pred, vector<double> label,string str,int flag=0);
};

class LSTMNetwork{
  private:
    LSTMParam *param;
    vector<LSTMNode> lstm_node_list;
    vector< vector<double> > x_list;
  public:
    LSTMNetwork(LSTMParam*);
    /***
    *update diffs by setting target sequence with corrensponding loss layer
    */
    double y_list_is(vector< vector<double> > y_list,LossLayer lossLayer);
    void x_list_clear();
    void x_list_add(vector<double> x);
    
    vector<LSTMNode> get_lstm_node_list();
    vector< vector<double> > get_x_list();
};
#endif
