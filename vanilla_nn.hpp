#ifndef VANILLA_NN_HPP
#define VANILLA_NN_HPP

/*
 *  The simplest neural network, using vanilla SGD
*/

#include <iostream>
#include <cstdarg>
#include <cmath>
#include "matrix.hpp"

// ------------- REWRITE AS FUNCTION CLASS MAYBE, NEW NAMESPACE -----------
#define SIGMOID       &Vanilla_nn<T>::sigmoid
#define D_SIGMOID     &Vanilla_nn<T>::d_sigmoid
#define TANH          &Vanilla_nn<T>::tanh
#define D_TANH        &Vanilla_nn<T>::d_tanh
#define RELU          &Vanilla_nn<T>::relu
#define D_RELU        &Vanilla_nn<T>::d_relu
#define LEAKY_RELU    &Vanilla_nn<T>::leaky_relu
#define D_LEAKY_RELU  &Vanilla_nn<T>::d_leaky_relu

#define SOFTMAX       &Vanilla_nn<T>::softmax
#define D_SOFTMAX     &Vanilla_nn<T>::d_softmax

#define MSE           &Vanilla_nn<T>::mse
#define D_MSE         &Vanilla_nn<T>::d_mse
#define CCE           &Vanilla_nn<T>::cce
#define D_CCE         &Vanilla_nn<T>::d_cce

template <typename E>

class Vanilla_nn{
private:

    int nlayers;
    int *nperlayer;         // size: nlayers
    Matrix<E> **weights;    // size: nlayers-1
    Matrix<E> **biases;     // size: nlayers-1

    Matrix<E> (*activation)(Matrix<E> x);
    Matrix<E> (*d_activation)(Matrix<E> x);

    Matrix<E> (*final_activation)(Matrix<E>& in);
    Matrix<E> (*d_final_activation)(Matrix<E>& in);

    E (*cost)(Matrix<E>& output, Matrix<E>& truth);
    Matrix<E> (*d_cost)(Matrix<E>& output, Matrix<E>& truth);

public:

    //  arguments:
    //      pointer to activation()
    //      pointer to d_activation()
    //      pointer to final_activation()
    //      pointer to d_final_activation()
    //      pointer to cost()
    //      pointer to d_cost()
    //      nlayers
    //      nperlayer[0]
    //      ...
    //      nperlayer[nlayers-1]

    Vanilla_nn(Matrix<E> (*activation)(Matrix<E> x), Matrix<E> (*d_activation)(Matrix<E> x),
               Matrix<E> (*final_activation)(Matrix<E>& in), Matrix<E> (*d_final_activation)(Matrix<E>& in),
               E (*cost)(Matrix<E>& output, Matrix<E>& truth), 
               Matrix<E> (*d_cost)(Matrix<E>& output, Matrix<E>& truth),
               const int nlayers, ...){
        
        this->activation = activation;
        this->d_activation = d_activation;
        
        this->final_activation = final_activation;
        this->d_final_activation = d_final_activation;

        this->cost = cost;
        this->d_cost = d_cost;

        va_list args;
        va_start(args, nlayers);
        this->nlayers = nlayers;
        this->nperlayer = new int[this->nlayers];
        for(int i=0; i<nlayers; i++){
            this->nperlayer[i] = va_arg(args, int);
        }
        va_end(args);

        weights = init_weights(nlayers, nperlayer);
        biases = init_biases(nlayers, nperlayer);
    }


    ~Vanilla_nn(){
        delete[] nperlayer;
        for(int i=0; i<nlayers-1; i++){
            delete weights[i];
            delete biases[i];
        }
        delete[] weights;
        delete[] biases;
    }

    // ----------------------------- decl ----------------------------------

    Matrix<E>** init_weights(int nlayers, int *nperlayer);
    Matrix<E>** init_biases(int nlayers, int *nperlayer);
    Matrix<E> predict(const Matrix<E>& input);
    void fit( Matrix<E>&input,  Matrix<E>& truth, E learning_rate);
    void save_weights(char *filename);
    void load_weights(char *filename);
    Vanilla_nn<E> snapshot();
    void print_summary();

    static void displayProgress(float ratio, int size){
        std::cout<<"[";
        for(int i=0; i<ratio*size; i++)
            std::cout<<"█";
        for(int i=0; i< size-1-ratio*size; i++)
            std::cout<<" ";
        std::cout<<"]   " << ratio*100 << "%" << std::endl;
    }

    // ----------------------------- act ----------------------------------

    static Matrix<E> sigmoid(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = 1/(1+expl((-1)*x(i, 0)));
        return out;
    }

    static Matrix<E> d_sigmoid(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = expl((-1)*x(i, 0))/((1+expl((-1)*x(i, 0)))*(1+expl((-1)*x(i, 0))));
        return out;
    }

    static Matrix<E> tanh(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = 2/(1+expl((-2)*x(i, 0))) - 1;
        return out;
    }

    static Matrix<E> d_tanh(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = (4*expl((-2)*x(i, 0)))/((1+expl((-2)*x(i, 0)))*(1+expl((-2)*x(i, 0))));
        return out;
    }

    static Matrix<E> relu(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = (x(i, 0) > 0) ? x(i, 0) : 0;
        return out;
    }

    static Matrix<E> d_relu(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = (x(i, 0) > 0) ? 1 : 0;
        return out;
    }

    static Matrix<E> leaky_relu(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = (x(i, 0) > 0) ? x(i, 0) : 0.001*x(i, 0);
        return out;
    }

    static Matrix<E> d_leaky_relu(Matrix<E> x){
        Matrix<E> out(x.getRows(), 1);
        for(int i=0; i<x.getRows(); i++)
            out(i, 0) = (x(i, 0) > 0) ? 1 : 0.001;
        return out;
    }

    // -------------------------- final act ---------------------------------------

    static Matrix<E> softmax(Matrix<E>& x){
        E sum = 0;
        Matrix<E> out(x);
        for(int i=0; i<x.getRows(); i++) sum += expl(x(i, 0));
        for(int i=0; i<x.getRows(); i++) out(i, 0) = expl(x(i, 0))/sum;
        return out;
    }

    // returns transpose of Jacobian
    static Matrix<E> d_softmax(Matrix<E>& x){
        Matrix<E> d_s(x.getRows(), x.getRows());
        Matrix<E> s(x);
        s = softmax(x);
        for(int i=0; i<x.getRows(); i++){
            for(int j=0; j<x.getRows(); j++)
                d_s(i, j) = (i == j) ? (s(j, 0)*(1 - s(i, 0))) : (-s(j, 0)*s(i, 0));
        }
        return d_s;
    }


    // -------------------------- cost ------------------------------------

    static E mse( Matrix<E>& output,  Matrix<E>& truth){
        E error = 0;
        for(int i=0; i<output.getRows(); i++){
            error +=  0.5 * (output(i, 0) - truth(i, 0)) * (output(i, 0) - truth(i, 0));
        }
        return error;
    }

    static Matrix<E> d_mse( Matrix<E>& output,  Matrix<E>& truth){
        Matrix<E> d(output);
        for(int i=0; i<d.getRows(); i++){
            d(i, 0) = output(i, 0) - truth(i, 0);
        }
    }

    static E cce( Matrix<E>& output,  Matrix<E>& truth){
        E error = 0;
        for(int i=0; i<output.getRows(); i++){
            error -= truth(i, 0)*logl(output(i, 0));
        }
        return error;
    }

    static Matrix<E> d_cce( Matrix<E>& output,  Matrix<E>& truth){
        Matrix<E> d(output);
        for(int i=0; i<output.getRows(); i++){
            d(i, 0) = output(i, 0) - truth(i, 0);
        }
        return d;
    }
};

#include "vanilla_nn.cpp"

#endif