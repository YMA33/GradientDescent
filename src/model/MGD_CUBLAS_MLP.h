#ifndef ALGO_MGD_CUBLAS_MLP_H_
#define ALGO_MGD_CUBLAS_MLP_H_

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

class MGD_CUBLAS_MLP{
public:
    cublasHandle_t handle;
    const double alpha = 1.;
    const double beta = 0.;

	InputData* input_data;
    HyperPara* hyper_para;
    NeuralNets* neural_nets;

	vector<double*> d_label;
	vector<vector<double*> > d_y;
	vector<double*> d_weight;
	vector<vector<double*> > d_gradient;
	
    vector<double*> d_rsum_y;
	vector<double*> d_prodlog;
	vector<vector<double*> > d_dlossy;
	vector<vector<double*> > d_dlossx;

    double stepsize;
    double loss;
    double forward_time;
    double backprop_time;

    Timer mgd_timer;

    MGD_CUBLAS_MLP(int num_tuples, int gradient_size, int num_classes, char* filename,
            int num_threads, int batch_size, double decay, double N_0, int iterations, int seed,
            int num_layers, vector<int>& units);
    ~MGD_CUBLAS_MLP();
    void load_data();
    void init_model(int c_init);
    double get_loss();
    double get_loss_para();
    void train();

    void compute_gradient(int batch_idx, int processed_tuples);
    void update_model();
    void forward(int batch_idx, int processed_tuples);

};

MGD_CUBLAS_MLP::MGD_CUBLAS_MLP(int n_tuples, int grad_size, int n_classes, char* fname,
    int n_threads, int b_size, double d, double N, int iter, int s,
    int n_layers, vector<int>& units){

	cublasCreate(&handle);

	input_data = new InputData(n_tuples, grad_size, n_classes, fname);
    hyper_para = new HyperPara(n_threads, b_size, d, N, iter, s);
    neural_nets = new NeuralNets(n_layers, units);

    hyper_para->num_batches = input_data->num_tuples/hyper_para->batch_size + 1;
    hyper_para->tuples_last_batch = input_data->num_tuples - (hyper_para->num_batches-1)*hyper_para->batch_size;
    if(hyper_para->tuples_last_batch==0){
    	hyper_para->num_batches--;
    	hyper_para->last_batch_processed = false;
    }
    printf("num_batches: %d, #tuples in last batch: %d\n", hyper_para->num_batches, hyper_para->tuples_last_batch);

    d_label.resize(hyper_para->num_batches);
    d_y.resize(hyper_para->num_batches);
    d_gradient.resize(hyper_para->num_batches);
    d_rsum_y.resize(hyper_para->num_batches);
    d_prodlog.resize(hyper_para->num_batches);
    d_dlossy.resize(hyper_para->num_batches);
    d_dlossx.resize(hyper_para->num_batches);

	#pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < hyper_para->num_batches; i++){
    	int processed_tuples = hyper_para->batch_size;
    	if(i == hyper_para->num_batches-1 && hyper_para->last_batch_processed)
    		processed_tuples = hyper_para->tuples_last_batch;

    	//cudaMalloc(&d_label[i],sizeof(double)*processed_tuples*input_data->num_classes);
    	d_y[i].resize(neural_nets->num_layers);
    	d_gradient[i].resize(neural_nets->num_grad);
    	//cudaMalloc(&d_y[i][0],sizeof(double)*processed_tuples*input_data->gradient_size);
    	cudaMalloc(&d_rsum_y[i], sizeof(double)*processed_tuples);
    	cudaMalloc(&d_prodlog[i], sizeof(double)*processed_tuples*input_data->num_classes);
    	d_dlossy[i].resize(neural_nets->num_grad);
    	d_dlossx[i].resize(neural_nets->num_grad);

    	for(int j = 0; j < neural_nets->num_grad; j++){
    		cudaMalloc(&d_gradient[i][j], sizeof(double)*neural_nets->num_units[j]*neural_nets->num_units[j+1]);
    		cudaMalloc(&d_y[i][j+1],sizeof(double)*processed_tuples*neural_nets->num_units[j+1]);
    	    cudaMalloc(&d_dlossy[i][j],sizeof(double)*processed_tuples*neural_nets->num_units[j+1]);
    	    cudaMalloc(&d_dlossx[i][j],sizeof(double)*processed_tuples*neural_nets->num_units[j+1]);
    	}
    }

    d_weight.resize(neural_nets->num_grad);
    for(int i = 0; i < neural_nets->num_grad; i++){
    	cudaMalloc(&d_weight[i], sizeof(double)*neural_nets->num_units[i]*neural_nets->num_units[i+1]);
    }

    stepsize = N;
    loss = 0.;
    forward_time = 0.;
    backprop_time = 0.;
    batchforward_time = 0.;
    batch_time = 0.;
}

MGD_CUBLAS_MLP::~MGD_CUBLAS_MLP(){
    delete input_data;
    delete hyper_para;
    delete neural_nets;

    for(int i = 0; i < hyper_para->num_batches; i++){
    	cudaFree(d_label[i]);
       	cudaFree(d_y[i][0]);
       	cudaFree(d_rsum_y[i]);
        cudaFree(d_prodlog[i]);
    	for(int j = 0; j < neural_nets->num_grad; j++){
       		cudaFree(d_gradient[i][j]);
       		cudaFree(d_y[i][j+1]);
            cudaFree(d_dlossy[i][j]);
            cudaFree(d_dlossx[i][j]);
    	}
    }
   	for(int i = 0; i < neural_nets->num_grad; i++){
   		cudaFree(d_weight[i]);
   	}
}

void MGD_CUBLAS_MLP::load_data(){
	double* h_data = (double*)malloc(sizeof(double)*hyper_para->batch_size*input_data->gradient_size);
	double* h_label = (double*)malloc(sizeof(double)*hyper_para->batch_size*input_data->num_classes);

	char str[1000];
	double val, y_val;
	mgd_timer.Restart();
	FILE *file = fopen(input_data->filename, "r");

	for(int i = 0; i < hyper_para->num_batches; i++){
        for(int j = 0; j < hyper_para->batch_size && i*hyper_para->batch_size+j < input_data->num_tuples; j++){
        	fscanf(file, "%lf", &y_val);
        	if(y_val == -1.){ // [1,0]
        		h_label[j*input_data->num_classes] = 1.;
        		h_label[j*input_data->num_classes+1] = 0.;
        	} else{ // [0,1]
        		h_label[j*input_data->num_classes] = 0.;
        		h_label[j*input_data->num_classes+1] = 1.;
        	}
        	for(int k = 0; k < input_data->gradient_size; k++){
        		fscanf(file, ",%lf", &val);
        		h_data[j*input_data->gradient_size+k] = val;
        	}
        }
		
        int processed_tuples = hyper_para->batch_size;
    	if(i == hyper_para->num_batches-1 && hyper_para->last_batch_processed)	processed_tuples = hyper_para->tuples_last_batch;
    	
        cudaMalloc(&d_label[i],sizeof(double)*processed_tuples*input_data->num_classes);
   		cudaMalloc(&d_y[i][0],sizeof(double)*processed_tuples*input_data->gradient_size);
   		cudaMemcpy(d_label[i],h_label,sizeof(double)*processed_tuples*input_data->num_classes, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_y[i][0],h_data,sizeof(double)*processed_tuples*input_data->gradient_size, cudaMemcpyHostToDevice);

        fgets(str, 1000, file);
	}
	fclose(file);
	printf("reading_time, %.10f\n", mgd_timer.GetTime());

	free(h_label);
	free(h_data);
}

void MGD_CUBLAS_MLP::init_model(int c_init){
	vector<double*> h_bgd_weight(neural_nets->num_grad);
	for(int i = 0; i < neural_nets->num_grad; i++){
		h_bgd_weight[i] = (double*)malloc(sizeof(double)*neural_nets->num_units[i]*neural_nets->num_units[i+1]);
	}

    double t_dist = 0.;
	switch(c_init){
	case 0:	// normal_distribution
		t_dist = int(sqrt(neural_nets->num_units[0] + neural_nets->num_units[1]))+1;
		//printf("t_dist: %.f\n", t_dist);
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
		cudaMemcpy(d_weight[i], h_bgd_weight[i], sizeof(double)*neural_nets->num_units[i]*neural_nets->num_units[i+1], cudaMemcpyHostToDevice);
	}
	h_bgd_weight.clear();
}

void MGD_CUBLAS_MLP::update_model(){
    update_model_bgd_cublas(d_weight, d_gradient, neural_nets, stepsize);
}

void MGD_CUBLAS_MLP::forward(int batch_idx, int processed_tuples){
	forward_mgd_cublas(handle, d_label[batch_idx], d_y[batch_idx], d_weight, neural_nets->num_units, processed_tuples,
		d_rsum_y[batch_idx], alpha, beta);
}

void MGD_CUBLAS_MLP::compute_gradient(int batch_idx, int processed_tuples){
	compute_gradient_bgd_cublas(handle, d_label[batch_idx], d_y[batch_idx], d_weight, neural_nets, processed_tuples,
		stepsize, alpha, beta, d_gradient[batch_idx], d_dlossy[batch_idx], d_dlossx[batch_idx]);
}

double MGD_CUBLAS_MLP::get_loss(){
	double loss = get_loss_mgd_cublas(handle, d_label, d_y, d_weight, neural_nets->num_units,
		hyper_para, d_rsum_y, d_prodlog, alpha, beta);
	return	loss / input_data->num_tuples;
}

double MGD_CUBLAS_MLP::get_loss_para(){
	double loss = get_loss_mgd_cublas_para(handle, d_label, d_y, d_weight, neural_nets->num_units,
		hyper_para,	d_rsum_y, d_prodlog, alpha, beta);
	return loss / input_data->num_tuples;
}

void MGD_CUBLAS_MLP::train(){
	mgd_timer.Restart();
	loss = 	get_loss();
	// loss = get_loss_para();
    printf("initial loss,%.10f, time, %.10f\n",loss, mgd_timer.GetTime());

	for(int i = 0; i < hyper_para->iterations; i++){
		stepsize = hyper_para->N_0 * exp(-hyper_para->decay * i);

		mgd_timer.Restart();
		//#pragma omp parallel for schedule(dynamic)
		for(int j = 0; j < hyper_para->num_batches; j++){
			int processed_tuples = hyper_para->batch_size;
			if(j == hyper_para->num_batches-1 && hyper_para->last_batch_processed)
				processed_tuples = hyper_para->tuples_last_batch;

			forward(j, processed_tuples);
			compute_gradient(j, processed_tuples);
            update_model();			
		}
	    backprop_time += mgd_timer.GetTime();

	    mgd_timer.Restart();
	    loss = 	get_loss();
        //loss = get_loss_para();
	    forward_time += mgd_timer.GetTime();

	    printf("iter,%d,stepsize,%.10f,backprop_time,%.10f,forward_time,%.10f,loss,%.10f\n",
	    	i+1, stepsize, backprop_time, forward_time, loss);
	}

}


#endif /* ALGO_MGD_CUBLAS_MLP_H_ */
