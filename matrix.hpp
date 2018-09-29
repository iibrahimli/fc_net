//
// Created by Imran on 05-Sep-18.
//

#ifndef VANILLA_NN_MATRIX_HPP
#define VANILLA_NN_MATRIX_HPP

#include <iostream>
#include <random>

static const int MAT_SEED = 0;

enum Distribution { UNIFORM, NORMAL };  // for Net

template <typename E>
class Matrix{
private:
    int rows{}, cols{};
    E *mat{};

public:
    static size_t copied;

    Matrix(){
        this->rows = 0;
        this->cols = 0;
        this->mat = nullptr;
    }

    Matrix(int rows, int cols){
        this->rows = rows;
        this->cols = cols;
        this->mat = new E [rows*cols];
        for(auto i=0; i<rows*cols; i++)
            mat[i] = 0;
    }

    Matrix(int rows, int cols, E val){
        this->rows = rows;
        this->cols = cols;
        this->mat = new E [rows*cols];
        for(auto i=0; i<rows*cols; i++)
            mat[i] = val;
    }

    // to initialize using a distribution
    Matrix(int rows, int cols, Distribution d, E mean, E std_dev){
        this->rows = rows;
        this->cols = cols;
        this->mat = new E [rows*cols];

        switch(d){
            case UNIFORM:
            {
                std::random_device rand_dev;
                std::mt19937 rng(rand_dev());

                std::uniform_real_distribution<E> uni_dist(mean, std_dev);
                for(int i=0; i<rows*cols; i++)
                    mat[i] = uni_dist(rng);
                break;
            }
            case NORMAL:
            {
                std::random_device rand_dev;
                std::mt19937 rng(rand_dev());

                std::normal_distribution<E> normal_dist(mean, std_dev);
                for(int i=0; i<rows*cols; i++)
                    mat[i] = normal_dist(rng);
                break;
            }
        }

    }

    //copy constructor
    Matrix(const Matrix<E>& m){
//      std::cout << "copy constructor called\n";
        rows = m.rows;
        cols = m.cols;
        mat = new E [rows*cols];
        for(int i=0; i<rows; i++){
            for(int j=0; j<cols; j++)
                mat[i*cols+j] = m.mat[i*cols+j];
        }
        copied += m.cols*m.rows;
        // std::cout << "constructor copied "<< m.rows * m.cols<<"elements\n";
    }

    //move constructor
    Matrix(Matrix<E>&& m) noexcept {
//      std::cout << "move constructor called\n";
        rows = m.rows;
        cols = m.cols;
        mat  = m.mat;

        m.rows = 0;
        m.cols = 0;
        m.mat  = nullptr;
    }

    //copy assignment operator
    Matrix<E>& operator = (const Matrix<E>& m){
//      std::cout << "copy assignment called\n";
        if(this != &m){
            delete [] mat;

            rows = m.rows;
            cols = m.cols;

            mat = new E[rows*cols];

            for(int i=0; i<rows*cols; i++)
                mat[i] = m.mat[i];

            copied += m.cols*m.rows;
            // std::cout << "copy assignment copied "<< m.rows * m.cols<<"elements\n";
            return *this;
        }
        else{
            std::cerr << "trying to copy the same matrix\n";
            return *this;
        }
    }

    //move assignment operator
    Matrix<E>& operator = (Matrix<E>&& m) noexcept {
//      std::cout << "move assignment called\n";
        if(this != &m){
            rows = m.rows;
            cols = m.cols;
            mat  = m.mat;

            m.rows = 0;
            m.cols = 0;
            m.mat  = nullptr;
        }
        return *this;
    }

    ~Matrix(){
        delete [] this->mat;
        this->mat = nullptr;
    }

    int getRows() const { return this->rows; }
    int getCols() const { return this->cols; }
    std::string shape() const;
    void operator += (const Matrix&  m);
    void operator -= (const Matrix&  m);
    void operator *= (E n);
    Matrix<E>& hadamard(const Matrix<E>& m);
    Matrix<E> T() const;
    E& operator ()(int i, int j);   // access to matrix elements
    void print() const;

    friend Matrix<E> operator * (const Matrix<E>& a, const Matrix<E>& b){
        if(a.getCols() != b.getRows()){
            std::cerr<<"multiplication: bad dimensions ["<<a.getRows()<<", "<<a.getCols()<<"] and ["<<b.getRows()<<", "<<b.getCols()<<"]\n";
        }
//        if(a.getRows() == b.getRows() && a.getCols() == 1 && b.getCols() == 1) b = b.T();

        Matrix<E> res = Matrix<E>(a.getRows(), b.getCols());

        for(int j=0; j<b.getCols(); j++){
            for(int i=0; i<a.getRows(); i++){
                for(int k=0; k<b.getRows(); k++)
                    res.mat[i*res.cols+j] += a.mat[i*a.cols+k] * b.mat[k*b.cols+j];
            }
        }
        return res;
    }

    friend Matrix<E> operator * (Matrix<E>& a, E scalar){

        Matrix<E> res = Matrix<E>(a);

        for(int i=0; i<a.rows; i++){
            for(int j=0; j<a.cols; j++){
                res.mat[i*a.cols+j] *= scalar;
            }
        }

        return res;
    }

    friend Matrix<E> operator * (E scalar, Matrix<E>& a){

        Matrix<E> res = Matrix<E>(a);

        for(int i=0; i<a.rows; i++){
            for(int j=0; j<a.cols; j++){
                res.mat[i*a.cols+j] *= scalar;
            }
        }

        return res;
    }

    friend bool operator == (Matrix<E>& a, Matrix<E>& b){
        if(a.rows != b.rows || a.cols != b.cols)
            return false;
        for(int i=0; i<a.rows; i++){
            for(int j=0; j<a.cols; j++){
                if(a(i, j) != b(i, j))
                    return false;
            }
        }
        return true;
    }

};

template <typename E>
size_t Matrix<E>::copied = 0;

#include "matrix.cpp"

#endif //VANILLA_NN_MATRIX_HPP
