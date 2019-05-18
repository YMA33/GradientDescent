#ifndef TRAIN_LOSS_VCL_CPU_H_
#define TRAIN_LOSS_VCL_CPU_H_

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/sum.hpp"
#include "viennacl/linalg/host_based/common.hpp"  // CPU
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
////////////////////////////////////////
void mat_vec_div(
    double* _mat,
    unsigned int start1, unsigned int start2,
    unsigned int inc1, unsigned int inc2,
    unsigned int size1, unsigned int size2,
    unsigned int internal_size1, unsigned int internal_size2,
    double* _vec){
    #pragma omp parallel for
    for(unsigned int row = 0; row < size1; row++){
        for(unsigned int col = 0; col < size2; col++){
            _mat[viennacl::row_major::mem_index(row*inc1+start1,col*inc2+start2,internal_size1,internal_size2)] /= _vec[row];
        }
    }
}
////////////////////////////////////////
viennacl::matrix<double> softmax(
    viennacl::matrix<double>& _x,
    int row, int col){
    // exp(x_i)/row_sum(exp(x_i))
    viennacl::matrix<double> exp_x = viennacl::linalg::element_exp(_x);
    viennacl::vector<double> rsum_exp_x = viennacl::linalg::row_sum(exp_x);
    mat_vec_div(viennacl::linalg::host_based::detail::extract_raw_pointer<double>(exp_x), // CPU
        static_cast<unsigned int>(viennacl::traits::start1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::start2(exp_x)),
        static_cast<unsigned int>(viennacl::traits::stride1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::stride2(exp_x)),
        static_cast<unsigned int>(viennacl::traits::size1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::size2(exp_x)),
        static_cast<unsigned int>(viennacl::traits::internal_size1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::internal_size2(exp_x)),
        viennacl::linalg::host_based::detail::extract_raw_pointer<double>(rsum_exp_x));  // CPU
    return exp_x;
}
////////////////////////////////////////
#endif
