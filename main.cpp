#include <iostream>
#include "lstm_all.h"

using namespace std;

int main(){
  LSTMParam lstmParam(2,3);
  cout<<lstmParam.printM()<<endl;
  return 1;
}
