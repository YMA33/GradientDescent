#ifndef TRAIN_LOSS_VCL_H_
#define TRAIN_LOSS_VCL_H_

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/sum.hpp"
////////////////////////////////////////
double cross_entropy(
    viennacl::matrix<double>& _x, // output
    viennacl::matrix<double>& _y, // label
    int row){
    // -mean.sum(label*log(output) by row)
    viennacl::matrix<double> t_eprod = viennacl::linalg::element_prod(_y, viennacl::linalg::element_log(_x));
    return -viennacl::linalg::sum(viennacl::linalg::row_sum(t_eprod)) / row;
}
////////////////////////////////////////
viennacl::matrix<double> der_sce(
    viennacl::matrix<double>& vc_label,
    viennacl::matrix<double>& vc_yout,
    int row, int col){ 
    // row=num_tuples, col=num_classes
    // (yout-ylabel)/m
    viennacl::matrix<double> mat_tuples = viennacl::scalar_matrix<double>(row, col, row);
    return viennacl::linalg::element_div(vc_yout - vc_label, mat_tuples);
}
////////////////////////////////////////
#endif
