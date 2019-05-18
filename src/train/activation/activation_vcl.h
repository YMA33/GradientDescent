#ifndef TRAIN_ACTIVATION_VCL_H_
#define TRAIN_ACTIVATION_VCL_H_

#include "viennacl/matrix.hpp"
////////////////////////////////////////
viennacl::matrix<double> sigmoid(
    viennacl::matrix<double> _x, 
    int row, int col){
    // 1/(1+exp(-x))
    viennacl::matrix<double> one = viennacl::scalar_matrix<double>(row, col, 1.);
    return viennacl::linalg::element_div(one, one + viennacl::linalg::element_exp(-_x));
}
////////////////////////////////////////
viennacl::matrix<double> tanh(
    viennacl::matrix<double> _x, 
    int row, int col){
    // (1-exp(-2x))/(1+exp(-2x))
	viennacl::matrix<double> one = viennacl::scalar_matrix<double>(row, col, 1.);
    return viennacl::linalg::element_div(one - viennacl::linalg::element_exp(-2*_x), one + viennacl::linalg::element_exp(-2*_x));
}
////////////////////////////////////////
viennacl::matrix<double> relu(
    viennacl::matrix<double> _x, 
    int row, int col){
    // max(0,x)
	viennacl::matrix<double> two = viennacl::scalar_matrix<double>(row, col, 2.);
	return viennacl::linalg::element_div(_x + viennacl::linalg::element_fabs(_x), two);
}
////////////////////////////////////////
viennacl::matrix<double> der_sigmoid(
    viennacl::matrix<double>& _x,
    int row, int col){
    // (1-x)*x
    viennacl::matrix<double> one = viennacl::scalar_matrix<double>(row, col, 1.);
    return viennacl::linalg::element_prod((one - _x), _x);
}
////////////////////////////////////////
viennacl::matrix<double> der_tanh(
    viennacl::matrix<double>& _x,
    int row, int col){
    // 1-x^2
    viennacl::matrix<double> one = viennacl::scalar_matrix<double>(row, col, 1.);
    return one - viennacl::linalg::element_prod(_x, _x);
}
////////////////////////////////////////
#endif
