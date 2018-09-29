//
// Created by Imran on 05-Sep-18.
//

#ifndef VANILLA_NN_FUNC_HPP
#define VANILLA_NN_FUNC_HPP

#include "matrix.hpp"
#include "impl.hpp"


template <typename E>
struct Activation{

    Matrix<E> (*f)(Matrix<E>);
    Matrix<E> (*d)(Matrix<E>);

    Activation(Matrix<E> (*f_)(Matrix<E>),
               Matrix<E> (*d_)(Matrix<E>))
               : f(f_), d(d_)
    {
    }
};

template <typename E>
struct Final_Activation{

    Matrix<E> (*f)(Matrix<E>&);
    Matrix<E> (*d)(Matrix<E>&);

    Final_Activation(Matrix<E> (*f_)(Matrix<E>&),
                     Matrix<E> (*d_)(Matrix<E>&))
                     : f(f_), d(d_)
    {
    }
};

template <typename E>
struct Cost{

    E         (*f)(Matrix<E>&, Matrix<E>&);
    Matrix<E> (*d)(Matrix<E>&, Matrix<E>&);

    Cost(        E (*f_)(Matrix<E>&, Matrix<E>&),
         Matrix<E> (*d_)(Matrix<E>&, Matrix<E>&))
         : f(f_), d(d_)
    {
    }
};


namespace F{

    // ----------    Activation    ----------

    template <typename E>
    Activation<E> sigmoid(impl::sigmoid, impl::d_sigmoid);

    template <typename E>
    Activation<E> relu(impl::relu, impl::d_relu);

    template <typename E>
    Activation<E> leaky_relu(impl::leaky_relu, impl::d_leaky_relu);

    template <typename E>
    Activation<E> tanh(impl::tanh, impl::d_tanh);


    // ---------- Final Activation ----------

    template <typename E>
    Final_Activation<E> softmax(impl::softmax, impl::d_softmax);


    // ------------     Cost     ------------

    template <typename E>
    Cost<E> mse(impl::mse, impl::d_mse);

    template <typename E>
    Cost<E> cce(impl::cce, impl::d_cce);

}

#endif //VANILLA_NN_FUNC_HPP
