#include "lstm_all.h"

matrix<double> uniform_ditr_init(int row,int column){
  static uniform_real_distribution<double> distributions(-0.1,0.1);
  static default_random_engine generator(time(0)); //�ڵ��ԵĹ��̣���������������Բ�����ͬ��������У�����һ��������ϣ�ͨ��һ��seed�ֿ��Բ�����ͬ���������
  double number=0.0;
  matrix<double> m= matrix<double>(row,column);
  
  for(size_t i =0;i<m.size1();i++){
    for(size_t j=0;j<m.size2();j++){
      number = distributions(generator);
      m(i,j) = number;
    }
  }
  return m;
}

matrix<double> zeros_init(int row,int column){
  zero_matrix<double> mat (row,column);
  return mat;
}

matrix<double> zeros_init_like(matrix<double> m){
  int row = m.size1();
  int column = m.size2();
  zero_matrix<double> mat (row,column);
  return mat;
}

