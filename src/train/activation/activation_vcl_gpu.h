#ifndef TRAIN_ACTIVATION_VCL_GPU_H_
#define TRAIN_ACTIVATION_VCL_GPU_H_

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/cuda/common.hpp"    // GPU
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
////////////////////////////////////////
__global__ void der_relu_kernel(
    double* _mat,
    unsigned int start1, unsigned int start2,
    unsigned int inc1, unsigned int inc2,
    unsigned int size1, unsigned int size2,
    unsigned int internal_size1, unsigned int internal_size2,
    double* tmp){
    unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
    unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
    for (unsigned int row = row_gid; row < size1; row += gridDim.x){
        for (unsigned int col = col_gid; col < size2; col += blockDim.x){
            int idx = (row*inc1+start1)*internal_size2+col*inc2+start2;
            if(_mat[idx] > 0.)   tmp[idx] = 1.;
            else    tmp[idx] = 0.;
        }
    }
}
////////////////////////////////////////
viennacl::matrix<double> der_relu(
    viennacl::matrix<double>& _x,
    int row, int col){
    // _x>0: 1, else 0
    viennacl::matrix<double> tmp = _x;
    der_relu_kernel<<<128,128>>>(viennacl::cuda_arg(_x),    // GPU
        static_cast<unsigned int>(viennacl::traits::start1(_x)), 
        static_cast<unsigned int>(viennacl::traits::start2(_x)),
        static_cast<unsigned int>(viennacl::traits::stride1(_x)), 
        static_cast<unsigned int>(viennacl::traits::stride2(_x)),
        static_cast<unsigned int>(viennacl::traits::size1(_x)), 
        static_cast<unsigned int>(viennacl::traits::size2(_x)),
        static_cast<unsigned int>(viennacl::traits::internal_size1(_x)), 
        static_cast<unsigned int>(viennacl::traits::internal_size2(_x)),
        viennacl::cuda_arg(tmp));    // GPU
    return tmp;
}
////////////////////////////////////////
#endif
