# goofiNN-around

A basic C++ deep learning library that I wrote as a side project, and to make sure I understand how simple NNs work. Supports only fully-connected feedforward neural networks (`Net` class) right now.

## **Supports**:
  
  * **Flexible and easy interface**. To create a network, you specify: functions to be used (see second point), number of layers, and number of neurons in each layer. Constructor supports variable number of arguments
    
  * **Custom functions**: 
    * _layer activation_ - used in hidden layers, ex: tanh, ReLU
    * _final activation_ - used in final layer, ex: softmax, linear
    * _cost_ - used to calculate error, ex: mean squared error, categorical cross entropy

      The library includes some of the most popular functions. You can easily use your own functions by providing a function and its derivative

  * **Different data types**. Since everything is templated, you can use `Matrix<T>` and `Net<T>` classes with any built-in numeric type you desire. Useful in controlling the precision and memory consumption

  * **Random weight initialization**. You can use normal distibution with given mean and standard deviation, or uniform distribution with given min and max values to initialize weights and biases. By default, `Matrix<T>` class uses normal distribution with mean of 0 and standard deviation of 1. You can change this in `matrix.cpp`


### **Example**:

```C++
#include <iostream>
#include "net.hpp"

// convenience
typedef double T;

int main(){

    const T   LEARNING_RATE = 0.01;
    const int INPUT_SIZE    = 4;
    const int OUTPUT_SIZE   = 3;
    
    auto x = Matrix<T>(INPUT_SIZE, 1, 0);     // creating a new 3x1 matrix (column vector) filled with zeros
    x(1, 0) = 3;                              // accessing matrix elements
    x(0, 0) = 0.5;
    
    auto truth = Matrix<T>(OUTPUT_SIZE, 1, 0);
    truth(2, 0) = 1;

    // creating a 3-layer (1 hidden) network with ReLU activation, using a softmax layer and cross-entropy loss
    // 4 input neurons, 5 hidden neurons, and 3 output neurons
    auto net = Net<T>(F::relu<T>, F::softmax<T>, F::cce<T>, 3, INPUT_SIZE, 5, OUTPUT_SIZE);
    
    // feeding x to untrained net
    auto pred = net.predict(x);
    
    // running a single weight update using one sample, and displaying loss 
    T err = net.fit(x, truth, LEARNING_RATE);
    
    std::cout << "Loss is: " << err << std::endl;

    return 0;
}

```
