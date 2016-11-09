g++ -c initialize_matrix.cpp -std=c++11;
g++ -c nonlinear_fun.cpp -std=c++11;
g++ -c LSTMParam.cpp -std=c++11;
g++ -c LSTMState.cpp -std=c++11;
g++ -c LSTMNode.cpp -std=c++11;
g++ -c LossLayer.cpp -std=c++11;
g++ -c LSTMNetwork.cpp -std=c++11;
g++ -c demo1.cpp -std=c++11;
g++ -o out initialize_matrix.o nonlinear_fun.o LSTMParam.o LSTMState.o LSTMNode.o LossLayer.o LSTMNetwork.o demo1.o
