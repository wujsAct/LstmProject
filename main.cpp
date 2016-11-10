#include <iostream>
#include <vector>
#include <typeinfo>
using namespace std;

int main(){
  vector< vector <double> > tt;
  vector<double> t; t.push_back(1);
  tt.push_back(t);
  int t1 = 0;
  cout<<typeid(t1).name()<<endl;
  return 1;
}
