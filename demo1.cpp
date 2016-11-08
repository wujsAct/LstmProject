#include <iostream>
#include <vector>

using namespace std;


int main(){
  vector<double> c;
  vector<double> t; t.push_back(1);t.push_back(2);
  c = t;
  t.push_back(4);
  cout<<c.size()<<endl;
}
