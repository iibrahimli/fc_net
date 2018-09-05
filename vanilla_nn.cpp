#ifndef VANILLA_NN_CPP
#define VANILLA_NN_CPP

#include "vanilla_nn.hpp"


template<typename E>
Matrix<E>** Vanilla_nn<E>::init_weights(int nlayers, int *nperlayer){
    Matrix<E> **weights = new Matrix<E>*[nlayers-1];
    for(int i=0; i<nlayers-1; i++){
        weights[i] = new Matrix<E>(nperlayer[i+1], nperlayer[i], NORMAL, 10, 1);
    }

    return weights;
}

template<typename E>
Matrix<E>** Vanilla_nn<E>::init_biases(int nlayers, int *nperlayer){
    Matrix<E> **biases = new Matrix<E>*[nlayers-1];
    for(int i=0; i<nlayers-1; i++){
        biases[i] = new Matrix<E>(nperlayer[i+1], 1, NORMAL, 0, 1);
    }
    return biases;
}

template<typename E>
Matrix<E> Vanilla_nn<E>::predict(const Matrix<E>& input){
    Matrix<E> out(input);
    for(int i=0; i<nlayers-2; i++){
        out = *(weights[i]) * out;
        out = out + *(biases[i]);
        out = activation(out);
    }
    out = *(weights[nlayers-2]) * out;
    out = out + *(biases[nlayers-2]);
    out = final_activation(out);
    return out;
}

template<typename E>
void Vanilla_nn<E>::fit( Matrix<E>&input,  Matrix<E>& truth, E learning_rate){
    Matrix<E> output = predict(input);
    E final_error = cost(output, truth);
    std::cout<<"error = "<<final_error<<"\n";

    Matrix<E> update(1, 1);
    Matrix<E> d_next(1, 1);

    update = d_cost(output, truth);
    d_next = d_final_activation(update);
    update = d_next * update;
    *biases[nlayers-2] = *biases[nlayers-2] - learning_rate*update;

}

template<typename E>
void Vanilla_nn<E>::save_weights(char *filename){

}

template<typename E>
void Vanilla_nn<E>::load_weights(char *filename){

}

template<typename E>
Vanilla_nn<E> Vanilla_nn<E>::snapshot(){

}

template<typename E>
void Vanilla_nn<E>::print_summary(){

    std::cout<<"# of layers:           "<<nlayers<<"\n";

    std::cout<<"neurons in each layer: "<<"[";
    for(int i=0; i<nlayers-1; i++) std::cout<<nperlayer[i]<<", ";
    std::cout<<nperlayer[nlayers-1]<<"]\n";
    
    std::cout<<"weight matrices:       "<<"[";
    for(int i=0; i<nlayers-2; i++) std::cout<<"["<<weights[i]->getRows()<<", "<<weights[i]->getCols()<<"], ";
    std::cout<<"["<<weights[nlayers-2]->getRows()<<", "<<weights[nlayers-2]->getCols()<<"]]\n";

    // // printing weight matrices
    // for(int i=0; i<nlayers-1; i++) {
    //     weights[i]->print();
    //     std::cout << "\n";
    // }
    
    std::cout<<"biases:                "<<"[";
    for(int i=0; i<nlayers-2; i++) std::cout<<"["<<biases[i]->getRows()<<", "<<biases[i]->getCols()<<"], ";
    std::cout<<"["<<biases[nlayers-2]->getRows()<<", "<<biases[nlayers-2]->getCols()<<"]]\n";

    // // printing bias matrices
    // for(int i=0; i<nlayers-1; i++) {
    //     biases[i]->print();
    //     std::cout << "\n";
    // }

    unsigned long params = 0;
    for(int i=0; i<nlayers-1; i++) {
        params += weights[i]->getRows()*weights[i]->getCols();
        params += biases[i]->getRows()*biases[i]->getCols();
    }

    std::cout<<"total # of parameters: "<<params<<"\n";

}

#endif