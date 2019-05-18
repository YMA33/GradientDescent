#ifndef TRAIN_CUBLAS_H_
#define TRAIN_CUBLAS_H_

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "activation/activation_cublas.h"
#include "loss/loss_cublas.h"
////////////////////////////////////////
__global__ void bgd_element_update(double* d_weight, double* d_gradient, double learning_rate, int num_elements, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < num_elements; i+= n_threads){
        d_weight[i] -= learning_rate * d_gradient[i];
    }
}
////////////////////////////////////////
void update_model_bgd_cublas(vector<double*>& d_bgd_weight, vector<double*>& d_bgd_gradient, NeuralNets* nn, double learning_rate){
	for(int i = 0; i < nn->num_grad; i++){
		bgd_element_update<<<52,1024>>>(d_bgd_weight[i], d_bgd_gradient[i], learning_rate, nn->num_units[i+1]*nn->num_units[i], 52*1024);
		cudaDeviceSynchronize();
	}
}
////////////////////////////////////////
void compute_gradient_bgd_cublas(
	cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	NeuralNets* nn, int num_tuples, double learning_rate, double alpha, double beta,
	vector<double*>& d_bgd_gradient, vector<double*>& d_bgd_dlossy, vector<double*>& d_bgd_dlossx){
    
    int i = nn->num_grad;
    bgd_element_der_sce<<<52,1024>>>(d_bgd_dlossy[i-1], d_label, d_y[i], num_tuples*nn->num_units[i], num_tuples, 52*1024);
    cudaDeviceSynchronize();

    d_bgd_dlossx[i-1] = d_bgd_dlossy[i-1]; 

    int m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, d_bgd_dlossx[i-1], n, d_y[i-1], m, &beta, d_bgd_gradient[i-1], n);
    cudaDeviceSynchronize();

    for(i = nn->num_grad-1; i > 0; i--){
        m = num_tuples, n = nn->num_units[i], k = nn->num_units[i+1];
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i], k, d_bgd_dlossx[i], k, &beta, d_bgd_dlossy[i-1], n);
        cudaDeviceSynchronize();

		#ifdef SIGMOID
        bgd_element_der_sigmoid_prod<<<52,1024>>>(d_bgd_dlossx[i-1], d_bgd_dlossy[i-1], d_y[i], num_tuples*nn->num_units[i], 52*1024);
		#endif
		#ifdef TANH
        bgd_element_der_tanh_prod<<<52,1024>>>(d_bgd_dlossx[i-1], d_bgd_dlossy[i-1], d_y[i], num_tuples*nn->num_units[i], 52*1024);
		#endif
		#ifdef RELU
        bgd_element_der_relu_prod<<<52,1024>>>(d_bgd_dlossx[i-1], d_bgd_dlossy[i-1], d_y[i], num_tuples*nn->num_units[i], 52*1024);
		#endif
        cudaDeviceSynchronize();

        m = nn->num_units[i-1], n = nn->num_units[i], k = num_tuples;
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, k, &alpha, d_bgd_dlossx[i-1], n, d_y[i-1], m, &beta, d_bgd_gradient[i-1], n);
        cudaDeviceSynchronize();
    }
}
////////////////////////////////////////
void forward_mgd_cublas(cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double alpha, double beta){

	int i;
	for(i = 1; i < num_units.size()-1; i++){
		int m = num_tuples, n = num_units[i], k = num_units[i-1];
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i-1], n, d_y[i-1], k, &beta, d_y[i], n);
		cudaDeviceSynchronize();

		#ifdef SIGMOID
		bgd_element_sigmoid<<<52,1024>>>(d_y[i], m*n, 52*1024);
		#endif
		#ifdef TANH
		bgd_element_tanh<<<52,1024>>>(d_y[i], m*n, 52*1024);
		#endif
		#ifdef RELU
		bgd_element_relu<<<52,1024>>>(d_y[i], m*n, 52*1024);
		#endif
		cudaDeviceSynchronize();
	}

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[i], num_tuples, num_units[i-1],
		&alpha, d_bgd_weight[i-1], num_units[i], d_y[i-1], num_units[i-1], &beta, d_y[i], num_units[i]);
	cudaDeviceSynchronize();

    softmax_bgd<<<52,1024>>>(d_rsum_y, d_y[i], num_tuples, num_units[i], 52*1024);
	cudaDeviceSynchronize();
}
////////////////////////////////////////
double get_loss_bgd_cublas(cublasHandle_t handle, double* d_label, vector<double*>& d_y, vector<double*>& d_bgd_weight,
	vector<int>& num_units, int num_tuples, double* d_rsum_y, double* d_prodlog, double alpha, double beta){

	int i;
	for(i = 1; i < num_units.size()-1; i++){

		int m = num_tuples, n = num_units[i], k = num_units[i-1];
		cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[i-1], n, d_y[i-1], k, &beta, d_y[i], n);
		cudaDeviceSynchronize();

		#ifdef SIGMOID
		bgd_element_sigmoid<<<52,1024>>>(d_y[i], m*n, 52*1024);
		#endif
		#ifdef TANH
		bgd_element_tanh<<<52,1024>>>(d_y[i], m*n, 52*1024);
		#endif
		#ifdef RELU
		bgd_element_relu<<<52,1024>>>(d_y[i], m*n, 52*1024);
		#endif
		cudaDeviceSynchronize();

	}

	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[i], num_tuples, num_units[i-1],
		&alpha, d_bgd_weight[i-1], num_units[i], d_y[i-1], num_units[i-1], &beta, d_y[i], num_units[i]);
	cudaDeviceSynchronize();

    softmax_bgd<<<52,1024>>>(d_rsum_y, d_y[i], num_tuples, num_units[i], 52*1024);	
	cudaDeviceSynchronize();

	return cross_entropy_bgd(handle, d_prodlog, d_label, d_y[i], num_tuples, num_units[i]);
}
////////////////////////////////////////
double get_loss_mgd_cublas(cublasHandle_t handle, vector<double*>& d_label, vector<vector<double*> >& d_y,
	vector<double*>& d_bgd_weight, vector<int>& num_units, HyperPara* hpara, vector<double*>& d_rsum_y,
	vector<double*>& d_prodlog, double alpha, double beta){

    double loss = 0.;
    int i, j, processed_tuples = hpara->batch_size;

    for(i = 0; i < hpara->num_batches; i++){
        if(i == hpara->num_batches-1 && hpara->last_batch_processed)  processed_tuples = hpara->tuples_last_batch;
        for(j = 1; j < num_units.size()-1; j++){
        	int m = processed_tuples, n = num_units[j], k = num_units[j-1];
        	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[j-1], n, d_y[i][j-1], k, &beta, d_y[i][j], n);
        	cudaDeviceSynchronize();

        	#ifdef SIGMOID
        	bgd_element_sigmoid<<<52,1024>>>(d_y[i][j], m*n, 52*1024);
        	#endif
        	#ifdef TANH
        	bgd_element_tanh<<<52,1024>>>(d_y[i][j], m*n, 52*1024);
        	#endif
        	#ifdef RELU
        	bgd_element_relu<<<52,1024>>>(d_y[i][j], m*n, 52*1024);
        	#endif
        	cudaDeviceSynchronize();
        }

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[j], processed_tuples, num_units[j-1],
        	&alpha, d_bgd_weight[j-1], num_units[j], d_y[i][j-1], num_units[j-1], &beta, d_y[i][j], num_units[j]);
        cudaDeviceSynchronize();

        softmax_bgd<<<52,1024>>>(d_rsum_y[i], d_y[i][j], processed_tuples, num_units[j], 52*1024);
        cudaDeviceSynchronize();

        loss += processed_tuples * cross_entropy_bgd(handle, d_prodlog[i], d_label[i], d_y[i][j], processed_tuples, num_units[j]);
        //printf("batch:%d, loss:%.5f\n", i, loss);
    }
    return loss;
}
////////////////////////////////////////
////////////////////////////////////////
double get_loss_mgd_cublas_para(cublasHandle_t handle, vector<double*>& d_label, vector<vector<double*> >& d_y,
	vector<double*>& d_bgd_weight, vector<int>& num_units, HyperPara* hpara, vector<double*>& d_rsum_y,
	vector<double*>& d_prodlog, double alpha, double beta){

    double loss = 0.;
    int i;
    vector<double> t_loss(hpara->num_batches);

    #pragma omp parallel for schedule(dynamic)
    for(i = 0; i < hpara->num_batches; i++){
    	int processed_tuples = hpara->batch_size;
    	if(i == hpara->num_batches-1 && hpara->last_batch_processed)  processed_tuples = hpara->tuples_last_batch;
    	int j;
    	for(j = 1; j < num_units.size()-1; j++){
    		int m = processed_tuples, n = num_units[j], k = num_units[j-1];
           	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_bgd_weight[j-1], n, d_y[i][j-1], k, &beta, d_y[i][j], n);
           	cudaDeviceSynchronize();

           	#ifdef SIGMOID
           	bgd_element_sigmoid<<<52,1024>>>(d_y[i][j], m*n, 52*1024);
           	#endif
           	#ifdef TANH
           	bgd_element_tanh<<<52,1024>>>(d_y[i][j], m*n, 52*1024);
           	#endif
           	#ifdef RELU
           	bgd_element_relu<<<52,1024>>>(d_y[i][j], m*n, 52*1024);
           	#endif
           	cudaDeviceSynchronize();
    	}

    	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, num_units[j], processed_tuples, num_units[j-1],
           	&alpha, d_bgd_weight[j-1], num_units[j], d_y[i][j-1], num_units[j-1], &beta, d_y[i][j], num_units[j]);
        cudaDeviceSynchronize();

        softmax_bgd<<<52,1024>>>(d_rsum_y[i], d_y[i][j], processed_tuples, num_units[j], 52*1024);
        cudaDeviceSynchronize();

        t_loss[i] = processed_tuples * cross_entropy_bgd(handle, d_prodlog[i], d_label[i], d_y[i][j], processed_tuples, num_units[j]);
    }

    for(i = 0; i < hpara->num_batches; i++)    loss += t_loss[i];
    return loss;
}
////////////////////////////////////////
#endif
