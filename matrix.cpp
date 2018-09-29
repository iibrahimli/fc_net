//
// Created by Imran on 05-Sep-18.
//

#ifndef MATRIX_CPP
#define MATRIX_CPP

#include "matrix.hpp"
#include <iostream>
#include <iomanip>

template<typename E>
void Matrix<E>::operator += (const Matrix<E>& m){
    if(rows != m.rows || cols != m.cols){
        std::cerr<<"addition: bad dimensions ["<<this->getRows()<<", "<<this->getCols()<<"] and ["<<m.getRows()<<", "<<m.getCols()<<"]\n";
    }
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            mat[i*cols+j] += m.mat[i*cols+j];
    }
}

template<typename E>
void Matrix<E>::operator -= (const Matrix<E>& m){
    if(rows != m.rows || cols != m.cols){
        std::cerr<<"subtraction: bad dimensions ["<<this->getRows()<<", "<<this->getCols()<<"] and ["<<m.getRows()<<", "<<m.getCols()<<"]\n";
    }
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            mat[i*cols+j] -= m.mat[i*cols+j];
    }
}

template<typename E>
void Matrix<E>::operator *= (E n){
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            mat[i*cols+j] *= n;
    }
}


template<typename E>
Matrix<E>& Matrix<E>::hadamard(const Matrix<E>& m){
    if(rows != m.rows || cols != m.cols){
        std::cerr<<"hadamard: bad dimensions ["<<this->getRows()<<", "<<this->getCols()<<"] and ["<<m.getRows()<<", "<<m.getCols()<<"]\n";
    }
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            mat[i*cols+j] *= m.mat[i*cols+j];
    }
    return *this;
}

template<typename E>
Matrix<E> Matrix<E>::T() const{
    Matrix t(cols, rows);
    t.cols = rows;
    t.rows = cols;
    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
            t.mat[i*cols+j] = mat[j*rows+i];
    }
    return t;
}

template<typename E>
E& Matrix<E>::operator ()(int i, int j){
    return mat[i*cols+j];
}

template<typename E>
void Matrix<E>::print() const {
    for(int i=0; i<rows; i++){
        std::cout << "|";
        for(int j=0; j<cols; j++)
            std::cout << std::right << std::setw(10) << mat[i*cols+j] << " ";
        std::cout << "|" << std::endl;
    }
}

template<typename E>
std::string Matrix<E>::shape() const {
    std::string shape = "(" + std::to_string(rows) + ", " + std::to_string(cols) + ")";
    return shape;
}

template<typename E>
Matrix<E> operator + (const Matrix<E>& a, const Matrix<E>& b){
    Matrix<E> res(a);
    res += b;
    return res;
}

template<typename E>
Matrix<E> operator - (const Matrix<E>& a, const Matrix<E>& b){
    Matrix<E> res(a);
    res -= b;
    return res;
}

#endif
