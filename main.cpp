#include <iostream>
#include "matrix.hpp"
#include "vanilla_nn.hpp"

using namespace std;

// for ease of use and macros
typedef long double T;

int main(){

    Matrix<T>& a = *(new Matrix<T>(4, 1, NORMAL, 0, 1));
    Matrix<T>& b = *(new Matrix<T>(3, 1));

    b(0, 0) = 1.0;

    cout << "a = \n";
    a.print();
    cout << "\n";

    Vanilla_nn<T> *net = new Vanilla_nn<T>(SIGMOID, D_SIGMOID, SOFTMAX, D_SOFTMAX, CCE, D_CCE, 3, 4, 6, 3);

    net->print_summary();
    cout << "\n";

    cout << "forward pass of a = \n";
    net->predict(a).print();
    cout << "\n";

    cout << "truth = \n";
    b.print();
    cout << "\n";

    net->fit(a, b, 0.001);

    Vanilla_nn<float>::displayProgress(0.7, 4);

    delete &a;
    delete &b;
    delete net;

    // cout<<"Copied "<<Matrix<T>::copied<<" elements in total\n";
    return 0;
}