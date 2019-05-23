#ifndef TRAIN_GLM_H_
#define TRAIN_GLM_H_
#include <vector>

using std::cout;
using std::endl;
////////////////////////////////////////
__global__ void LogisticRegression_loss_row_rr(double* example, int* label, double* weight, double* loss, int n_tuples, int n_gradient, int n_threads){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < n_tuples; i += n_threads){	
		double ret = 0.0;
		for(int j = 0; j < n_gradient; j++) ret += weight[j] * example[i * n_gradient + j];
		loss[tid] += log(1 + exp(-label[i] * ret));
    }
}
////////////////////////////////////////
__global__ void LogisticRegression_row_rr(double* example, int* label, double* gradient, double* weight, int n_tuples, int n_gradient, double stepsize, int n_threads){	
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	for(int i = tid; i < n_tuples; i += n_threads){	
		double ret = 0.0, sig = 0.0;
		for(int j = 0; j < n_gradient; j++) ret += weight[j] * example[i * n_gradient + j];
        ret = - ret * label[i]; 
		if(ret > 30)    sig = 1.0 / (1.0 + exp(-ret)); 
		else    sig = exp(ret) / (1.0 + exp(ret));
		double c = -label[i] * sig;
		for(int j = 0; j < n_gradient; j++)    gradient[tid * n_gradient + j] = example[i * n_gradient + j] * c;
		for(int j = 0; j < n_gradient; j++)
            //weight[(tid + j) % n_gradient] = weight[(tid + j) % n_gradient] - stepsize * gradient[tid * n_gradient + (tid + j) % n_gradient];
            weight[j] = weight[j] - stepsize * gradient[tid * n_gradient + j];
	}// end of processing tuples
}
////////////////////////////////////////
#endif
