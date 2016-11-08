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
  this->bg = uniform_ditr_init(1,mem_cell_dim);
  this->bi = uniform_ditr_init(1,mem_cell_dim);
  this->bf = uniform_ditr_init(1,mem_cell_dim);
  this->bo = uniform_ditr_init(1,mem_cell_dim);
  
  //diffs
  this->wg_diff = zeros_init(mem_cell_dim,concat_dim);
  this->wi_diff = zeros_init(mem_cell_dim,concat_dim);
  this->wf_diff = zeros_init(mem_cell_dim,concat_dim);
  this->wo_diff = zeros_init(mem_cell_dim,concat_dim);
  
  this->bg_diff = zeros_init(1,mem_cell_dim);
  this->bi_diff = zeros_init(1,mem_cell_dim);
  this->bf_diff = zeros_init(1,mem_cell_dim);
  this->bo_diff = zeros_init(1,mem_cell_dim);
}

void LSTMParam::apply_diff(double lr=1.0)
{
  this->wg -= lr*this->wg_diff;
  this->wi -= lr*this->wi_diff;
  this->wf -= lr*this->wf_diff;
  this->wo -= lr*this->wo_diff;
  this->bg -= lr*this->bg_diff;
  this->bi -= lr*this->bi_diff;
  this->bf -= lr*this->bf_diff;
  this->bo -= lr*this->bo_diff;
  
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

