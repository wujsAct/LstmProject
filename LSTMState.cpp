#include "lstm_all.h"

LSTMState::LSTMState(int mem_cell_dim, int x_dim){
  this->mem_cell_dim = mem_cell_dim;
  this->x_dim = x_dim;
  
  this->g = uniform_ditr_init(1,mem_cell_dim);
  this->i = uniform_ditr_init(1,mem_cell_dim);
  this->f = uniform_ditr_init(1,mem_cell_dim);
  this->o = uniform_ditr_init(1,mem_cell_dim);
  this->s = uniform_ditr_init(1,mem_cell_dim);
  this->h = uniform_ditr_init(1,mem_cell_dim); 
  
  
  this->bottom_diff_h = zeros_init_like(this->h);
  this->bottom_diff_s = zeros_init_like(this->s);
  this->bottom_diff_x = zeros_init(1,x_dim); 
}
