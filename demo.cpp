#include <iostream>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <random>
#include <ctime>
using namespace boost::numeric::ublas;
using namespace std;

void uniform_ditr_init(matrix<double> &m){
  static uniform_real_distribution<double> distributions(-0.1,0.1);
  static default_random_engine generator(time(0)); //�ڵ��ԵĹ��̣���������������Բ�����ͬ��������У�����һ��������ϣ�ͨ��һ��seed�ֿ��Բ�����ͬ���������
  double number=0.0;
  for(size_t i =0;i<m.size1();i++){
    for(size_t j=0;j<m.size2();j++){
      number = distributions(generator);
      m(i,j) = number;
      cout<<number<<"\t";
    }
    cout<<endl;
  }
}

int main () {
    matrix<double> m = matrix<double> (1,4);
    uniform_ditr_init(m);
    cout << m << endl;
    matrix<double> m1 = matrix<double> (1,4);
    uniform_ditr_init(m1);
    cout<<m+m1<<endl;
}
