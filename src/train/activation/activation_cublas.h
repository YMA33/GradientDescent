#ifndef TRAIN_ACTIVATION_CUBLAS_H_
#define TRAIN_ACTIVATION_CUBLAS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
////////////////////////////////////////
__global__ void bgd_element_sigmoid(double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        double t = exp(-d_y[i]);
        d_y[i] = 1/(1+t); 
	}	
}
////////////////////////////////////////
__global__ void bgd_element_tanh(double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        double t = exp(-2*d_y[i]);
        d_y[i] = (1-t)/(1+t); 
	}	
}
////////////////////////////////////////
__global__ void bgd_element_relu(double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        if(d_y[i] < 0.)  d_y[i] = 0.;  
	}	
}
////////////////////////////////////////
__global__ void bgd_element_der_sigmoid_prod(double* dlossx, double* dlossy, double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        dlossx[i] = dlossy[i] * (1 - d_y[i]) * d_y[i];
    }
}
////////////////////////////////////////
__global__ void bgd_element_der_tanh_prod(double* dlossx, double* dlossy, double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        dlossx[i] = dlossy[i] * (1 - d_y[i]*d_y[i]);
    }
}
////////////////////////////////////////
__global__ void bgd_element_der_relu_prod(double* dlossx, double* dlossy, double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        if(d_y[i] > 0.)    dlossx[i] = dlossy[i];
        else    dlossx[i] = 0.;
    }
}
////////////////////////////////////////
#endif
