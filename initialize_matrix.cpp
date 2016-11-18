#include "lstm_all.h"

vector< vector<double> > uniform_ditr_init(int row,int column){
  static uniform_real_distribution<double> distributions(-0.1,0.1);
  static default_random_engine generator(time(0)); //在调试的过程，随机数生成器可以产生相同的随机序列！但是一旦调试完毕，通过一个seed又可以产生不同的随机数。
  double number=0.0;
  vector< vector<double> > m;
  
  for(int i =0;i<row;i++){
    vector<double> temp;
    for(int j=0;j<column;j++){
      number = distributions(generator);
      temp.push_back(number);
    }
    m.push_back(temp);
  }
  return m;
}

vector<double> uniform_ditr_init(int row){
  static uniform_real_distribution<double> distributions(-0.1,0.1);
  static default_random_engine generator(time(0)); //在调试的过程，随机数生成器可以产生相同的随机序列！但是一旦调试完毕，通过一个seed又可以产生不同的随机数。
  double number=0.0;
  vector<double> m;
  
  for(size_t i =0;i<row;i++){
    number = distributions(generator);
    m.push_back(number);
  }
  return m;
}


vector< vector<double> > zeros_init(int row,int column){
  vector< vector<double> > m;
  
  for(int i =0;i<row;i++){
    vector<double> temp;
    for(int j=0;j<column;j++){
      temp.push_back(0.0);
    }
    m.push_back(temp);
  }
  return m;
}

vector<double> zeros_init(int row){
  vector<double> m;
  
  for(int i =0;i<row;i++){
    m.push_back(0.0);
  }
  return m;
}


vector< vector<double> >  zeros_init_like(vector< vector<double> > m){
  int row = m.size();
  int column = m[0].size();
  return zeros_init(row,column);
}

vector<double>  zeros_init_like(vector<double>  m){
  int row = m.size();
  return zeros_init(row);
}

vector<double> vecA_minus_vecB(vector<double> vecA, vector<double> vecB){
  vector<double> ret;
  //cout<<"vecA.size():"<<vecA.size()<<endl;
  //cout<<"vecB.size():"<<vecB.size()<<endl;
  if(vecA.size()!=vecB.size()) {
    cout<<"vecA_minus_vecB wrong: dimension mismatch!"<<endl;
    exit(-1);
  }
  for(size_t i=0; i< vecA.size(); i++){
    ret.push_back(vecA[i]-vecB[i]); 
  }
  return ret;
}

vector<double> vecA_add_vecB(vector<double> vecA, vector<double> vecB){
  vector<double> ret;
  if(vecA.size()!=vecB.size()) {
    cout<<"vecA_add_vecB wrong: dimension mismatch!"<<endl;
    exit(-1);
  }
  for(size_t i=0; i< vecA.size(); i++){
    ret.push_back(vecA[i]+vecB[i]); 
  }
  return ret;
}

vector< vector<double> > matA_minus_matB(vector< vector<double> > matA, vector< vector<double> > matB){
  vector< vector<double> > ret;
  if(matA.size()!=matB.size() || matA[0].size() != matB[0].size()) {
    cout<<"matA_minus_matB wrong: dimension mismatch!"<<endl;
    exit(-1);
  }
  for(size_t i=0; i< matA.size(); i++){
    vector<double> temp;
    for(size_t j=0; j< matA[0].size();j++){
      temp.push_back(matA[i][j]-matB[i][j]);
    }
    ret.push_back(temp);
  }
  return ret;
}

vector< vector<double> > matA_add_matB(vector< vector<double> > matA, vector< vector<double> > matB){
  vector< vector<double> > ret;
  if(matA.size()!=matB.size() || matA[0].size() != matB[0].size()) {
    cout<<"matA_add_matB wrong: dimension mismatch!"<<endl;
    exit(-1);
  }
  for(size_t i=0; i< matA.size(); i++){
    vector<double> temp;
    for(size_t j=0; j< matA[0].size();j++){
      temp.push_back(matA[i][j]+matB[i][j]);
    }
    ret.push_back(temp);
  }
  return ret;
}

vector<double> vec_multiply_num(vector<double> vecA, double num){
  vector<double> ret;
  for(size_t i=0; i< vecA.size(); i++){
    ret.push_back(vecA[i]*num); 
  }
  return ret;
}

vector< vector<double> > mat_multiply_num(vector< vector<double> > matA, double num){
  vector< vector<double> > ret;

  for(size_t i=0; i< matA.size(); i++){
    ret.push_back(vec_multiply_num(matA[i],num));
  }
  return ret;
}

//two vecotrs concatenate
vector<double> concatenate(vector<double> m1, vector<double> m2){
  vector<double> ret;
  for(size_t i=0;i<m1.size();i++)
    ret.push_back(m1[i]);
  for(size_t i=0;i<m2.size();i++)
    ret.push_back(m2[i]);
  return ret;
}

//dot(matrix m, vector xc)
vector<double> mat_dot_vec(vector< vector<double> > m, vector<double> v){
  if(m[0].size() != v.size()) {
    cout<<"mat_dot_vec wrong: dimension mismatch!"<<endl;
    exit(-1);
  }
  int row = m.size();
  double nums=0;
  vector<double> ret;
  for(size_t i =0;i<m.size();i++){
    nums =0;
    for(size_t j =0;j<v.size();j++){
      nums += m[i][j] * v[j];
    }
    ret.push_back(nums);
  }  
  return ret;
}

double vec_dot_vec( vector<double>  v1, vector<double> v2){
  if(v1.size() != v2.size()) {
    cout<<"vec_dot_vec wrong: dimension mismatch!"<<endl;
    exit(-1);
  }
  double ret;
  for(size_t i =0;i<v1.size();i++){
    ret += v1[i]*v2[i];
  }  
  return ret;
}

//vector * vector
vector<double> vecA_mul_vecB(vector<double> v1, vector<double> v2){
  if(v1.size()!=v2.size()) {
    cout<<"vecA_mul_vecB wrong: dimension mismatch!"<<endl;
    exit(-1);
  }
  vector<double> ret;
  double num = 0;
  for(size_t i=0;i<v1.size();i++){
    num = v1[i] * v2[i];
    ret.push_back(num);
  }
  return ret;
}

vector< vector<double> > vecA_outer_vecB(vector<double> v1, vector<double> v2){
  vector< vector<double> > ret;
  double num = 0;
  for(size_t i=0;i<v1.size();i++){
    vector<double> temp;
    for(size_t j=0;j<v2.size();j++){
      num = v1[i] * v2[j];
      temp.push_back(num);
    }
    ret.push_back(temp);
  }
  return ret;
}

//nums-vector
vector<double> num_minus_vec(double num, vector<double> v){
  if(v.size()==0){
    cout<<"num_minus_vec wrong: vector is null"<<endl; 
    exit(-1);
  }
  vector<double> ret;
  for(size_t i=0;i<v.size();i++){
    ret.push_back(num-v[i]);
  }
  return ret;
}

//transpose matrix
vector <vector<double>> mat_transpose(vector< vector<double> > m){
  
  if(m.size()==0 || m[0].size()==0){
    cout<<"mat_transpose is wrong, can not be transposed"<<endl;
    exit(-1);
  }
  vector <vector<double>> ret;
  for(size_t j=0;j<m[0].size();j++){
    vector<double> temp;
    for(size_t i=0;i<m.size();i++){
      temp.push_back(m[i][j]);
    }
    ret.push_back(temp);
  }
  return ret;
}

//sub vector
vector<double> sub_vector(vector<double> v, int start, int end){
  int len = end -start;
  vector<double>::const_iterator first = v.begin() + start;
  vector<double>::const_iterator last = first + len;
  vector<double> ret(first, last);

  return ret;
}