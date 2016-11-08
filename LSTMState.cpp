#include "lstm_all.h"

LSTMState::LSTMState(int mem_cell_dim, int x_dim){
  this->mem_cell_dim = mem_cell_dim;
  this->x_dim = x_dim;
  
  this->g = uniform_ditr_init(mem_cell_dim);
  this->i = uniform_ditr_init(mem_cell_dim);
  this->f = uniform_ditr_init(mem_cell_dim);
  this->o = uniform_ditr_init(mem_cell_dim);
  this->s = uniform_ditr_init(mem_cell_dim);
  this->h = uniform_ditr_init(mem_cell_dim); 
  
  
  this->bottom_diff_h = zeros_init_like(this->h);
  this->bottom_diff_s = zeros_init_like(this->s);
  this->bottom_diff_x = zeros_init(x_dim); 
}

vector<double> LSTMState::get_g(){
  return this->g;
}

vector<double> LSTMState::get_i(){
  return this->i;
}

vector<double> LSTMState::get_f(){
  return this->f;
}

vector<double> LSTMState::get_o(){
  return this->o;
}

vector<double> LSTMState::get_s(){
  return this->s;
}

vector<double> LSTMState::get_h(){
  return this->h;
}

vector<double> LSTMState::get_bottom_diff_h(){
  return this->bottom_diff_h;
}

vector<double> LSTMState::get_bottom_diff_s(){
  return this->bottom_diff_s;
}

vector<double> LSTMState::get_bottom_diff_x(){
  return this->bottom_diff_x;
}

//set values
void LSTMState::set_g(vector<double> g){
  this->g = g;
}

void LSTMState::set_i(vector<double> i){
  this->i=i;
}

void LSTMState::set_f(vector<double> f){
  this->f=f;
}

void LSTMState::set_o(vector<double> o){
  this->o=o;
}

void LSTMState::set_s(vector<double> s){
  this->s=s;
}

void LSTMState::set_h(vector<double> h){
  this->h=h;
}

void LSTMState::set_bottom_diff_h(vector<double> bottom_diff_h){
  this->bottom_diff_h=bottom_diff_h;
}

void LSTMState::set_bottom_diff_s(vector<double> bottom_diff_s){
  this->bottom_diff_s=bottom_diff_s;
}

void LSTMState::set_bottom_diff_x(vector<double> bottom_diff_x){
  this->bottom_diff_x=bottom_diff_x;
}