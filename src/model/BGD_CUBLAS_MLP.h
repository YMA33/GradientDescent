#ifndef MODEL_BGD_CUBLAS_MLP_H_
#define MODEL_BGD_CUBLAS_MLP_H_

#include <stdio.h>
#include <stdlib.h>
#include <random>

#include "../util/Timer.h"
#include "../util/InputData.h"
#include "../util/HyperPara.h"
#include "../util/NeuralNets.h"
#include "../train/train_cublas.h"

using std::default_random_engine;
using std::uniform_real_distribution;
using std::normal_distribution;

class BGD_CUBLAS_MLP{
public:
    cublasHandle_t handle;
    const double alpha = 1.;
    const double beta = 0.;

	InputData* input_data;
    HyperPara* hyper_para;
    NeuralNets* neural_nets;

	double* d_label;
	vector<double*> d_y;
	vector<double*> d_bgd_weight;
	vector<double*> d_bgd_gradient;
	
    double* d_rsum_y;
	double* d_prodlog;
	vector<double*> d_bgd_dlossy;
	vector<double*> d_bgd_dlossx;

    double stepsize;
    double loss;
    double forward_time;
    double backprop_time;

    Timer bgd_timer;

    BGD_CUBLAS_MLP(int num_tuples, int gradient_size, int num_classes, char* filename,
    	int num_threads, int batch_size, double decay, double N_0, int iterations,
		int seed, int num_layers, vector<int>& units);
    ~BGD_CUBLAS_MLP();
    void load_data();
    void init_model(int c_init);
    void train();
    void compute_gradient();
    void update_model();
    double get_loss();
};

BGD_CUBLAS_MLP::BGD_CUBLAS_MLP(int n_tuples, int grad_size, int n_classes, char* fname,
    int n_threads, int b_size, double d, double N, int iter, int s,
    int n_layers, vector<int>& units){
	cublasCreate(&handle);

	input_data = new InputData(n_tuples, grad_size, n_classes, fname);
    hyper_para = new HyperPara(n_threads, b_size, d, N, iter, s);
    neural_nets = new NeuralNets(n_layers, units);

	cudaMalloc(&d_label,sizeof(double)*input_data->num_tuples*input_data->num_classes);
	d_y.resize(neural_nets->num_layers);
	d_bgd_weight.resize(neural_nets->num_grad);
	d_bgd_gradient.resize(neural_nets->num_grad);
	cudaMalloc(&d_y[0],sizeof(double)*input_data->num_tuples*input_data->gradient_size);
	
    cudaMalloc(&d_rsum_y, sizeof(double)*input_data->num_tuples);
	cudaMalloc(&d_prodlog, sizeof(double)*input_data->num_tuples*input_data->num_classes);
	d_bgd_dlossy.resize(neural_nets->num_grad);
	d_bgd_dlossx.resize(neural_nets->num_grad);

	for(int i = 0; i < neural_nets->num_grad; i++){
		cudaMalloc(&d_bgd_weight[i], sizeof(double)*neural_nets->num_units[i]*neural_nets->num_units[i+1]);
		cudaMalloc(&d_bgd_gradient[i], sizeof(double)*neural_nets->num_units[i]*neural_nets->num_units[i+1]);
		cudaMalloc(&d_y[i+1],sizeof(double)*input_data->num_tuples*neural_nets->num_units[i+1]);
	    
        cudaMalloc(&d_bgd_dlossy[i],sizeof(double)*input_data->num_tuples*neural_nets->num_units[i+1]);
	    cudaMalloc(&d_bgd_dlossx[i],sizeof(double)*input_data->num_tuples*neural_nets->num_units[i+1]);
	}

    stepsize = N;
    loss = 0.;
    forward_time = 0.;
    backprop_time = 0.;
}

BGD_CUBLAS_MLP::~BGD_CUBLAS_MLP(){
    delete input_data;
    delete hyper_para;
    delete neural_nets;

    cudaFree(d_label);
   	cudaFree(d_y[0]);
    
   	cudaFree(d_rsum_y);
    cudaFree(d_prodlog);
   	
    for(int i = 0; i < neural_nets->num_grad; i++){
   		cudaFree(d_bgd_weight[i]);
   		cudaFree(d_bgd_gradient[i]);
   		cudaFree(d_y[i+1]);
        
        cudaFree(d_bgd_dlossy[i]);
        cudaFree(d_bgd_dlossx[i]);
   	}
}

void BGD_CUBLAS_MLP::load_data(){
	char str[1000];
	double val, y_val;
    double* h_label = (double*)malloc(sizeof(double)*input_data->num_tuples*input_data->num_classes);
	double* h_data = (double*)malloc(sizeof(double)*input_data->num_tuples*input_data->gradient_size);
	
    bgd_timer.Restart();
	FILE *file = fopen(input_data->filename, "r");
	for(int i = 0; i < input_data->num_tuples; i++){
		fscanf(file, "%lf", &y_val);
		if(y_val == -1.){ // [1,0]
			h_label[i*input_data->num_classes] = 1.;
			h_label[i*input_data->num_classes+1] = 0.;
		} else{ // [0,1]
			h_label[i*input_data->num_classes] = 0.;
			h_label[i*input_data->num_classes+1] = 1.;
		}
		for(int j = 0; j < input_data->gradient_size; j++){
			fscanf(file, ",%lf", &val);
			h_data[i*input_data->gradient_size+j] = val;
		}
		fgets(str, 1000, file);
	}
	fclose(file);
	printf("reading_time, %.10f\n", bgd_timer.GetTime());
	
    cudaMemcpy(d_y[0],h_data,sizeof(double)*input_data->num_tuples*input_data->gradient_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_label,h_label,sizeof(double)*input_data->num_tuples*input_data->num_classes, cudaMemcpyHostToDevice);
	free(h_label);
	free(h_data);
}

void BGD_CUBLAS_MLP::init_model(int c_init){
	vector<double*> h_bgd_weight(neural_nets->num_grad);
	for(int i = 0; i < neural_nets->num_grad; i++){
		h_bgd_weight[i] = (double*)malloc(sizeof(double)*neural_nets->num_units[i]*neural_nets->num_units[i+1]);
	}

    double t_dist = 0.;
	switch(c_init){
	case 0:	// normal_distribution
		t_dist = int(sqrt(neural_nets->num_units[0] + neural_nets->num_units[1]))+1;
		printf("t_dist: %.f\n", t_dist);
		for(int i = 0; i < neural_nets->num_grad; i++){
			default_random_engine generator(hyper_para->seed);
			normal_distribution<double> distributions(
				0, sqrt(2.*t_dist/(neural_nets->num_units[i]+neural_nets->num_units[i+1])));
			//printf("layer: %d, normal_dist: 0;%.5f\n", i,	sqrt(2.*t_dist/(neural_nets->num_units[i]+neural_nets->num_units[i+1])));
			for(int j = 0; j < neural_nets->num_units[i]*neural_nets->num_units[i+1]; j++){
				h_bgd_weight[i][j] = distributions(generator);
			}
		}
		break;
	case 1:	// uniform_distribution for sigmoid
		for(int i = 0; i < neural_nets->num_grad; i++){
			t_dist = sqrt(6./(neural_nets->num_units[i]+neural_nets->num_units[i+1]));
			default_random_engine generator(hyper_para->seed);
			uniform_real_distribution<double> distributions(-t_dist,t_dist);
			//printf("layer: %d, uniform_dist for sigmoid: %.5f;%.5f\n", i, -t_dist, t_dist);
			for(int j = 0; j < neural_nets->num_units[i]*neural_nets->num_units[i+1]; j++){
				h_bgd_weight[i][j] = distributions(generator);
			}
		}
		break;
	case 2:	// uniform_distribution for tanh
		for(int i = 0; i < neural_nets->num_grad; i++){
			t_dist = 4 * sqrt(6./(neural_nets->num_units[i]+neural_nets->num_units[i+1]));
			default_random_engine generator(hyper_para->seed);
			uniform_real_distribution<double> distributions(-t_dist,t_dist);
			//printf("layer: %d, uniform_dist for tanh: %.5f;%.5f\n", i, -t_dist, t_dist);
			for(int j = 0; j < neural_nets->num_units[i]*neural_nets->num_units[i+1]; j++){
				h_bgd_weight[i][j] = distributions(generator);
			}
		}
	}

	for(int i = 0; i < neural_nets->num_grad; i++){
		cudaMemcpy(d_bgd_weight[i], h_bgd_weight[i], sizeof(double)
			*neural_nets->num_units[i]*neural_nets->num_units[i+1], cudaMemcpyHostToDevice);
	}
	h_bgd_weight.clear();
}

void BGD_CUBLAS_MLP::compute_gradient(){
	compute_gradient_bgd_cublas(handle, d_label, d_y, d_bgd_weight, neural_nets, input_data->num_tuples,
		stepsize, alpha, beta, d_bgd_gradient, d_bgd_dlossy, d_bgd_dlossx);
}

double BGD_CUBLAS_MLP::get_loss(){
	return get_loss_bgd_cublas(handle, d_label, d_y, d_bgd_weight, neural_nets->num_units,
		input_data->num_tuples, d_rsum_y, d_prodlog, alpha, beta);
}

void BGD_CUBLAS_MLP::update_model(){
    update_model_bgd_cublas(d_bgd_weight, d_bgd_gradient, neural_nets, stepsize);
}

void BGD_CUBLAS_MLP::train(){
	loss = get_loss();
	printf("initial loss,%.10f\n",loss);

	for(int i = 0; i < hyper_para->iterations; i++){
		stepsize = hyper_para->N_0 * exp(-hyper_para->decay * i);

		bgd_timer.Restart();
		compute_gradient();
        update_model();
		backprop_time += bgd_timer.GetTime();

	    bgd_timer.Restart();
	    loss = 	get_loss();
	    forward_time += bgd_timer.GetTime();

	    printf("iter,%d,stepsize,%.10f,backprop_time,%.10f,forward_time,%.10f,loss,%.10f\n",
	    	i+1, stepsize, backprop_time, forward_time, loss);
	}
}

#endif /* MODEL_BGD_CUBLAS_MLP_H_ */
