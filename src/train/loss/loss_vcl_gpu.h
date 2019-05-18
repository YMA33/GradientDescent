#ifndef TRAIN_LOSS_VCL_GPU_H_
#define TRAIN_LOSS_VCL_GPU_H_

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/sum.hpp"
#include "viennacl/linalg/cuda/common.hpp"    // GPU
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
////////////////////////////////////////
__global__ void mat_vec_div_kernel(
    double* _mat,
    unsigned int start1, unsigned int start2,
    unsigned int inc1, unsigned int inc2,
    unsigned int size1, unsigned int size2,
    unsigned int internal_size1, unsigned int internal_size2,
    double* _vec){
    unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
    unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
    for (unsigned int row = row_gid; row < size1; row += gridDim.x){
        for (unsigned int col = col_gid; col < size2; col += blockDim.x){
            _mat[(row*inc1+start1)*internal_size2+col*inc2+start2] /= _vec[row];
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
    mat_vec_div_kernel<<<128,128>>>(viennacl::cuda_arg(exp_x),    // GPU
        static_cast<unsigned int>(viennacl::traits::start1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::start2(exp_x)),
        static_cast<unsigned int>(viennacl::traits::stride1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::stride2(exp_x)),
        static_cast<unsigned int>(viennacl::traits::size1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::size2(exp_x)),
        static_cast<unsigned int>(viennacl::traits::internal_size1(exp_x)), 
        static_cast<unsigned int>(viennacl::traits::internal_size2(exp_x)),
        viennacl::cuda_arg(rsum_exp_x)); // GPU
    return exp_x;
}
////////////////////////////////////////
#endif
