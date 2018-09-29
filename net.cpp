//
// Created by Imran on 05-Sep-18.
//

#ifndef VANILLA_NN_NET_CPP
#define VANILLA_NN_NET_CPP

#include "net.hpp"

//  arguments:
//      an activation function            (of type F::Activation)
//      a final layer activation function (of type F::Final_Activation)
//      a cost function                   (of type F::Cost)
//      nlayers
//      nperlayer[0]
//      ...
//      nperlayer[nlayers-1]

template <typename E>
Net<E>::Net(Activation<E> activation, Final_Activation<E> final_activation, Cost<E> cost, const int nlayers, ...){

    this->activation = activation.f;
    this->d_activation = activation.d;

    this->final_activation = final_activation.f;
    this->d_final_activation = final_activation.d;

    this->cost = cost.f;
    this->d_cost = cost.d;

    va_list args;
    va_start(args, nlayers);
    this->nlayers = nlayers;
    this->nperlayer = new int[this->nlayers];
    for(int i=0; i<nlayers; i++){
        this->nperlayer[i] = va_arg(args, int);
    }
    va_end(args);

    weights = init_weights(nlayers, nperlayer);
    biases  = init_biases(nlayers, nperlayer);
    interm  = new Matrix<E>*[nlayers];
    for(int i=0; i<nlayers; i++){
        interm[i] = new Matrix<E>(nperlayer[i], 1);
    }
}

template <typename E>
Net<E>::~Net(){
    delete[] nperlayer;
    for(int i=0; i<nlayers-1; i++){
        delete weights[i];
        delete biases[i];
        delete interm[i];
    }
    delete interm[nlayers-1];

    delete[] weights;
    delete[] biases;
    delete[] interm;
}


template<typename E>
Matrix<E>** Net<E>::init_weights(int nlayers, int *nperlayer){
    Matrix<E> **weights = new Matrix<E>*[nlayers-1];
    for(int i=0; i<nlayers-1; i++){
        weights[i] = new Matrix<E>(nperlayer[i+1], nperlayer[i], NORMAL, 0, 1);
    }
    return weights;
}

template<typename E>
Matrix<E>** Net<E>::init_biases(int nlayers, int *nperlayer){
    Matrix<E> **biases = new Matrix<E>*[nlayers-1];
    for(int i=0; i<nlayers-1; i++){
        biases[i] = new Matrix<E>(nperlayer[i+1], 1, NORMAL, 0, 1);
    }
    return biases;
}

template<typename E>
Matrix<E> Net<E>::predict(const Matrix<E>& input){
    Matrix<E> out(input);
    *interm[0] = input;
    for(int i=0; i<nlayers-2; i++){
        out = *(weights[i]) * out;
        out += *(biases[i]);
        out = activation(out);
        *interm[i+1] = out;
    }
    out = *(weights[nlayers-2]) * out;
    out += *(biases[nlayers-2]);
    *interm[nlayers-1] = out;
    out = final_activation(out);
    return out;
}

template<typename E>
E Net<E>::fit(Matrix<E>&input, Matrix<E>& truth, E learning_rate, bool verbose){
    Matrix<E> output = predict(input);
    E final_error = cost(output, truth);
    if(verbose) std::cout << "error = " << final_error << "\n";

    Matrix<E> d_curbias;
    Matrix<E> d_curweight;
    Matrix<E> temp;
    Matrix<E> temp2;                                         // gets passed between layers  ( f l o w s )

    temp  = d_cost(output, truth);                           // wrt to final_act(final_layer)
    temp2 = d_final_activation(*interm[nlayers-1]) * temp;   // wrt to final_layer

    for(int i=nlayers-2; i>=0; i--){                         // layer index = i
        d_curbias = temp2;
        d_curweight = d_curbias * ((*interm[i]).T());

        // do smth with d_curbias and d_curweight
        *weights[i] -= d_curweight * learning_rate;
        *biases[i]  -= d_curbias * learning_rate;

        temp = d_curbias.T() * (*weights[i]);
        temp = temp.T();                                     // wrt to (input to current layer)
        temp2 = d_activation(*interm[i]).hadamard(temp);
    }

    return final_error;
}

template<typename E>
void Net<E>::save_weights(std::string filename){

}

template<typename E>
void Net<E>::load_weights(std::string filename){

}

template<typename E>
Net<E> Net<E>::snapshot(){

}

template<typename E>
void Net<E>::print_summary(bool verbose){

    std::cout<<"# of layers:           "<<nlayers<<"\n";

    std::cout<<"neurons in each layer: "<<"[";
    for(int i=0; i<nlayers-1; i++) std::cout<<nperlayer[i]<<", ";
    std::cout<<nperlayer[nlayers-1]<<"]\n";

    std::cout<<"weight matrices:       "<<"[";
    for(int i=0; i<nlayers-2; i++) std::cout<<"["<<weights[i]->getRows()<<", "<<weights[i]->getCols()<<"], ";
    std::cout<<"["<<weights[nlayers-2]->getRows()<<", "<<weights[nlayers-2]->getCols()<<"]]\n";

    if(verbose) {
        // printing weight matrices
        for (int i = 0; i < nlayers - 1; i++) {
            weights[i]->print();
            std::cout << "\n";
        }
    }

    std::cout<<"biases:                "<<"[";
    for(int i=0; i<nlayers-2; i++) std::cout<<"["<<biases[i]->getRows()<<", "<<biases[i]->getCols()<<"], ";
    std::cout<<"["<<biases[nlayers-2]->getRows()<<", "<<biases[nlayers-2]->getCols()<<"]]\n";

    if(verbose) {
        // printing bias matrices
        for (int i = 0; i < nlayers - 1; i++) {
            biases[i]->print();
            std::cout << "\n";
        }
    }

    std::cout<<"interm:                "<<"[";
    for(int i=0; i<nlayers-1; i++) std::cout<<"["<<interm[i]->getRows()<<", "<<interm[i]->getCols()<<"], ";
    std::cout<<"["<<interm[nlayers-1]->getRows()<<", "<<interm[nlayers-1]->getCols()<<"]]\n";

    unsigned long params = 0;
    for(int i=0; i<nlayers-1; i++) {
        params += weights[i]->getRows()*weights[i]->getCols();
        params += biases[i]->getRows()*biases[i]->getCols();
    }

    std::cout<<"total # of parameters: "<<params<<"\n";

}

#endif