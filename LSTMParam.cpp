#include "lstm_all.h"

LSTMParam::LSTMParam(int mem_cell_dim, int x_dim)
{
  this->mem_cell_dim = mem_cell_dim;
  this->x_dim = x_dim;
  this->concat_dim = mem_cell_dim + x_dim;
  
  //weight initialize
  this->wg = uniform_ditr_init(mem_cell_dim,concat_dim);
  this->wi = uniform_ditr_init(mem_cell_dim,concat_dim);
  this->wf = uniform_ditr_init(mem_cell_dim,concat_dim);
  this->wo =  uniform_ditr_init(mem_cell_dim,concat_dim);
  
  //bias terms initialize
  this->bg = uniform_ditr_init(mem_cell_dim);
  this->bi = uniform_ditr_init(mem_cell_dim);
  this->bf = uniform_ditr_init(mem_cell_dim);
  this->bo = uniform_ditr_init(mem_cell_dim);
  
  //diffs
  this->wg_diff = zeros_init(mem_cell_dim,concat_dim);
  this->wi_diff = zeros_init(mem_cell_dim,concat_dim);
  this->wf_diff = zeros_init(mem_cell_dim,concat_dim);
  this->wo_diff = zeros_init(mem_cell_dim,concat_dim);
  
  this->bg_diff = zeros_init(mem_cell_dim);
  this->bi_diff = zeros_init(mem_cell_dim);
  this->bf_diff = zeros_init(mem_cell_dim);
  this->bo_diff = zeros_init(mem_cell_dim);
}

void LSTMParam::apply_diff(double lr=1.0)
{
  this->wg = matA_minus_matB(this->wg, mat_multiply_num(this->wg_diff,lr));
  this->wi = matA_minus_matB(this->wi, mat_multiply_num(this->wi_diff,lr));
  this->wf = matA_minus_matB(this->wf, mat_multiply_num(this->wf_diff,lr));
  this->wo = matA_minus_matB(this->wo, mat_multiply_num(this->wo_diff,lr));
  this->bg = vecA_minus_vecB(this->bg, vec_multiply_num(this->bg_diff,lr));
  this->bi = vecA_minus_vecB(this->bi, vec_multiply_num(this->bi_diff,lr));
  this->bf = vecA_minus_vecB(this->bf, vec_multiply_num(this->bf_diff,lr));
  this->bo = vecA_minus_vecB(this->bo, vec_multiply_num(this->bo_diff,lr));
  
  //reset diffs to zeros
  this->wg_diff = zeros_init_like(this->wg);
  this->wi_diff = zeros_init_like(this->wi);
  this->wf_diff = zeros_init_like(this->wf);
  this->wo_diff = zeros_init_like(this->wo);
  
  this->bg_diff = zeros_init_like(this->bg);
  this->bi_diff = zeros_init_like(this->bi);
  this->bf_diff = zeros_init_like(this->bf);
  this->bo_diff = zeros_init_like(this->bo);
}

vector< vector<double> > LSTMParam::get_wg(){
  return this->wg;
}
vector< vector<double> > LSTMParam::get_wi(){
  return this->wi;
}
vector< vector<double> > LSTMParam::get_wf(){
  return this->wf;
}
vector< vector<double> > LSTMParam::get_wo(){
  return this->wo;
}

vector< vector<double> > LSTMParam::get_wg_diff(){
  return this->wg_diff;
}
vector< vector<double> > LSTMParam::get_wi_diff(){
  return this->wi_diff;
}
vector< vector<double> > LSTMParam::get_wf_diff(){
  return this->wf_diff;
}
vector< vector<double> > LSTMParam::get_wo_diff(){
  return this->wo_diff;
}

void LSTMParam::set_wg(vector< vector<double> > wg){
  this->wg = wg;
}
void LSTMParam::set_wi(vector< vector<double> > wi){
  this->wi = wi;
}
void LSTMParam::set_wf(vector< vector<double> > wf){
  this->wf = wf;
}
void LSTMParam::set_wo(vector< vector<double> > wo){
  this->wo = wo;
}

void LSTMParam::set_wg_diff(vector< vector<double> > wg_diff){
  this->wg_diff = wg_diff;
}
void LSTMParam::set_wi_diff(vector< vector<double> > wi_diff){
  this->wi_diff = wi_diff;
}
void LSTMParam::set_wf_diff(vector< vector<double> > wf_diff){
  this->wf_diff = wf_diff;
}
void LSTMParam::set_wo_diff(vector< vector<double> > wo_diff){
  this->wo_diff = wo_diff;
}




vector<double> LSTMParam::get_bg(){
  return this->bg;
}
vector<double> LSTMParam::get_bi(){
  return this->bi;
}
vector<double> LSTMParam::get_bf(){
  return this->bf;
}
vector<double> LSTMParam::get_bo(){
  return this->bo;
}

vector<double> LSTMParam::get_bg_diff(){
  return this->bg;
}
vector<double> LSTMParam::get_bi_diff(){
  return this->bi;
}
vector<double> LSTMParam::get_bf_diff(){
  return this->bf;
}
vector<double> LSTMParam::get_bo_diff(){
  return this->bo;
}

void LSTMParam::set_bg(vector<double> bg){
  this->bg = bg;
}
void LSTMParam::set_bi(vector<double> bi){
  this->bi = bi;
}
void LSTMParam::set_bf(vector<double> bf){
  this->bf = bf;
}
void LSTMParam::set_bo(vector<double> bo){
  this->bo = bo;
}

void LSTMParam::set_bg_diff(vector<double> bg_diff){
  this->bg_diff = bg_diff;
}
void LSTMParam::set_bi_diff(vector<double> bi_diff){
  this->bi_diff = bi_diff;
}
void LSTMParam::set_bf_diff(vector<double> bf_diff){
  this->bf_diff = bf_diff;
}
void LSTMParam::set_bo_diff(vector<double> bo_diff){
  this->bo_diff = bo_diff;
}