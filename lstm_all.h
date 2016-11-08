#ifndef LSTM_H
#define LSTM_H
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>
#include <ctime>
using namespace boost::numeric::ublas;
using namespace std;

class LSTMParam{
  private:
    int mem_cell_dim; //hidden dimension
    int x_dim; //input data dimension
    int concat_dim;
    //weight matrix
    matrix<double> wg,wi,wf,wo;
    //bias terms
    matrix<double> bg,bi,bf,bo;
    //diffs(derivative loss of function w.r.t all parameters)
    matrix<double> wg_diff,wi_diff,wf_diff,wo_diff,bg_diff,bi_diff,bf_diff,bo_diff;
    
  public:
    LSTMParam(int mem_cell_dim, int x_dim); //constructor
    void apply_diff(double lr);
    
};

class LSTMState{
  private:
    int mem_cell_dim; //hidden dimension
    int x_dim; //input data dimension
    matrix<double> g,i,f,o,s,h;
    matrix<double> bottom_diff_h,bottom_diff_s,bottom_diff_x;
  public:
    LSTMState(int mem_cell_dim, int x_dim); //constructor
};

class LSTMNode{
  private:
    LSTMParam lstmParam;
    LSTMState lstmState;
    //non-recurrent input to node;
    
    //non-recurrent input concatenated with recurrent input
  public:
    LSTMNode(LSTMParam lstmParam,LSTMState lstmState);
    
}
//initialize_matrix.cpp
matrix<double> uniform_ditr_init(int row,int column);
matrix<double> zeros_init(int row,int column);
matrix<double> zeros_init_like(matrix<double> m);
#endif
