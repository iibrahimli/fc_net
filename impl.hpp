//
// Created by Imran on 05-Sep-18.
//

#ifndef VANILLA_NN_IMPL_HPP
#define VANILLA_NN_IMPL_HPP

#include "matrix.hpp"

// implementations of all the functions, dont look

namespace F{
    namespace impl{

        template <typename E>
        Matrix<E> sigmoid(Matrix<E> x){
            Matrix<E> out(x.getRows(), 1);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = 1/(1+expl((-1)*x(i, 0)));
            return out;
        }

        template <typename E>
        Matrix<E> d_sigmoid(Matrix<E> x){
            Matrix<E> out(x.getRows(), 1);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = expl((-1)*x(i, 0))/((1+expl((-1)*x(i, 0)))*(1+expl((-1)*x(i, 0))));
            return out;
        }

        template <typename E>
        Matrix<E> tanh(Matrix<E> x){
            Matrix<E> out(x.getRows(), 1);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = 2/(1+expl((-2)*x(i, 0))) - 1;
            return out;
        }

        template <typename E>
        Matrix<E> d_tanh(Matrix<E> x){
            Matrix<E> out(x.getRows(), 1);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = (4*expl((-2)*x(i, 0)))/((1+expl((-2)*x(i, 0)))*(1+expl((-2)*x(i, 0))));
            return out;
        }

        template <typename E>
        Matrix<E> relu(Matrix<E> x){
            Matrix<E> out(x);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = (x(i, 0) > 0) ? x(i, 0) : 0;
            return out;
        }

        template <typename E>
        Matrix<E> d_relu(Matrix<E> x){
            Matrix<E> out(x.getRows(), 1);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = (x(i, 0) > 0) ? 1 : 0;
            return out;
        }

        template <typename E>
        Matrix<E> leaky_relu(Matrix<E> x){
            Matrix<E> out(x.getRows(), 1);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = (x(i, 0) > 0) ? x(i, 0) : 0.001*x(i, 0);
            return out;
        }

        template <typename E>
        Matrix<E> d_leaky_relu(Matrix<E> x){
            Matrix<E> out(x.getRows(), 1);
            for(int i=0; i<x.getRows(); i++)
                out(i, 0) = (x(i, 0) > 0) ? 1 : 0.001;
            return out;
        }

        // -------------------------- final act ---------------------------------------

        template <typename E>
        Matrix<E> softmax(Matrix<E>& x){
            E sum = 0;
            E max = x(0, 0);
            for(auto i = 1; i<x.getRows(); i++) {
                if(x(i, 0) > max) max = x(i, 0);
            }
            Matrix<E> out(x);
            for(int i=0; i<x.getRows(); i++) sum += expl(x(i, 0) - max);
            for(int i=0; i<x.getRows(); i++) out(i, 0) = expl(x(i, 0) - max)/sum;
            return out;
        }

        // returns T of Jacobian
        template <typename E>
        Matrix<E> d_softmax(Matrix<E>& x){
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

        template <typename E>
        E mse( Matrix<E>& output,  Matrix<E>& truth){
            E error = 0;
            for(int i=0; i<output.getRows(); i++){
                error +=  0.5 * (output(i, 0) - truth(i, 0)) * (output(i, 0) - truth(i, 0));
            }
            return error;
        }

        template <typename E>
        Matrix<E> d_mse( Matrix<E>& output,  Matrix<E>& truth){
            Matrix<E> d(output);

            if(output.getCols() != truth.getCols()){
                std::cerr << "Wrong dimensions in d_mse\n";
                return Matrix<E>(1, 1);
            }

            for(int i=0; i<d.getRows(); i++){
                d(i, 0) = output(i, 0) - truth(i, 0);
            }
            return d;
        }

        template <typename E>
        E cce( Matrix<E>& output,  Matrix<E>& truth){
            E error = 0;
            for(int i=0; i<output.getRows(); i++){
                error -= truth(i, 0)*logl(output(i, 0));
            }
            return error;
        }

        template <typename E>
        Matrix<E> d_cce( Matrix<E>& output,  Matrix<E>& truth){
            Matrix<E> d(output);
            for(int i=0; i<output.getRows(); i++){
                d(i, 0) = output(i, 0) - truth(i, 0);
            }
            return d;
        }
    }
}

#endif //VANILLA_NN_IMPL_HPP
