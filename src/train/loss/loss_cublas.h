#ifndef TRAIN_LOSS_CUBLAS_H_
#define TRAIN_LOSS_CUBLAS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
////////////////////////////////////////
__global__ void bgd_element_prodlog(double* d_prodlog, double* d_label, double* d_y, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        d_prodlog[i] = d_label[i] * log(d_y[i]);
    }
}
////////////////////////////////////////
__global__ void softmax_bgd(double* d_rsum_y, double* d_y, int row, int col, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < row; i+= n_threads){
        d_rsum_y[i] = 0.;
        for(int j = 0; j < col; j++){
            d_y[i*col+j] = exp(d_y[i*col+j]);
            d_rsum_y[i] += d_y[i*col+j];
        }
        for(int j = 0; j < col; j++){
	        d_y[i*col+j] = d_y[i*col+j] / d_rsum_y[i];
        }
    }	
}
////////////////////////////////////////
double cross_entropy_bgd(cublasHandle_t handle, double* d_prodlog, double* d_label, double* d_y, int row, int col){
    bgd_element_prodlog<<<52,1024>>>(d_prodlog, d_label, d_y, row*col, 52*1024);
	cudaDeviceSynchronize();
    double loss = 0.; 
    cublasDasum(handle, row*col, d_prodlog, 1, &loss);
	cudaDeviceSynchronize();
    return loss/row;
}
////////////////////////////////////////
__global__ void bgd_element_der_sce(double* dloss, double* d_label, double* d_y, int num_elements, int num_tuples, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        dloss[i] = (d_y[i] - d_label[i]) / num_tuples; 
    }
}
////////////////////////////////////////
#endif
