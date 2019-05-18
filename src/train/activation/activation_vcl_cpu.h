#ifndef TRAIN_ACTIVATION_VCL_CPU_H_
#define TRAIN_ACTIVATION_VCL_CPU_H_

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/host_based/common.hpp"  // CPU
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
////////////////////////////////////////
void der_relu_cpu(
    double* _mat,
    unsigned int start1, unsigned int start2,
    unsigned int inc1, unsigned int inc2,
    unsigned int size1, unsigned int size2,
    unsigned int internal_size1, unsigned int internal_size2,
    double* tmp){
    #pragma omp parallel for
    for(unsigned int row = 0; row < size1; row++){
        for(unsigned int col = 0; col < size2; col++){
            int idx = viennacl::row_major::mem_index(row*inc1+start1,col*inc2+start2,internal_size1,internal_size2);
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
    der_relu_cpu(viennacl::linalg::host_based::detail::extract_raw_pointer<double>(_x), // CPU
        static_cast<unsigned int>(viennacl::traits::start1(_x)), 
        static_cast<unsigned int>(viennacl::traits::start2(_x)),
        static_cast<unsigned int>(viennacl::traits::stride1(_x)), 
        static_cast<unsigned int>(viennacl::traits::stride2(_x)),
        static_cast<unsigned int>(viennacl::traits::size1(_x)), 
        static_cast<unsigned int>(viennacl::traits::size2(_x)),
        static_cast<unsigned int>(viennacl::traits::internal_size1(_x)), 
        static_cast<unsigned int>(viennacl::traits::internal_size2(_x)),
        viennacl::linalg::host_based::detail::extract_raw_pointer<double>(tmp)); // CPU
    return tmp;
}
////////////////////////////////////////
#endif
