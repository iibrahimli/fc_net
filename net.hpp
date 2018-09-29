//
// Created by Imran on 05-Sep-18.
//

#ifndef VANILLA_NN_NET_HPP
#define VANILLA_NN_NET_HPP

/*
 *  The simplest neural network, using vanilla SGD
*/

#include <iostream>
#include <cstdarg>
#include <cmath>
#include "matrix.hpp"
#include "func.hpp"

template <typename E>
class Net{
private:

    int nlayers;
    int *nperlayer;         // size: nlayers
    Matrix<E> **weights;    // size: nlayers-1
    Matrix<E> **biases;     // size: nlayers-1
    Matrix<E> **interm;     // size: nlayers  interm[i] = activation of ith layer, starting from i=0 = input

    Matrix<E> (*activation)(Matrix<E> x);
    Matrix<E> (*d_activation)(Matrix<E> x);

    Matrix<E> (*final_activation)(Matrix<E>& in);
    Matrix<E> (*d_final_activation)(Matrix<E>& in);

    E (*cost)(Matrix<E>& output, Matrix<E>& truth);
    Matrix<E> (*d_cost)(Matrix<E>& output, Matrix<E>& truth);

public:

    Net(Activation<E> activation, Final_Activation<E> final_activation, Cost<E> cost, int nlayers, ...);
    ~Net();

    // ----------------------------- decl ----------------------------------

    Matrix<E>** init_weights(int nlayers, int *nperlayer);
    Matrix<E>** init_biases(int nlayers, int *nperlayer);
    Matrix<E> predict(const Matrix<E>& input);
    E fit(Matrix<E>&input, Matrix<E>& truth, E learning_rate, bool verbose=false);
    void save_weights(std::string filename);
    void load_weights(std::string filename);
    Net<E> snapshot();
    void print_summary(bool verbose = false);

    static void displayProgress(float ratio, int size){
        std::cout<<"[";
        for(int i=0; i<int(ratio*size); i++)
            std::cout<<"=";
        if((int)ratio*size % 10 != 0){
            for(int i=0; i<size-int(ratio*size)-1; i++)
                std::cout<<" ";
        }
        else {
            for (int i = 0; i < size - int(ratio * size); i++)
                std::cout << " ";
        }
        std::cout<<"]   " << ratio*100.0 << " %" << std::endl;
    }

};

#include "net.cpp"

#endif //VANILLA_NN_NET_HPP
