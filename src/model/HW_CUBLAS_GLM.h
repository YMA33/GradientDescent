#ifndef MODEL_HW_CUBLAS_GLM_H_
#define MODEL_HW_CUBLAS_GLM_H_

#include <stdio.h>
#include <stdlib.h>
#include <random>

#include "../util/Timer.h"
#include "../util/InputData.h"
#include "../util/HyperParaGLM.h"
#include "../util/NeuralNets.h"
#include "../train/train_cublas.h"

using std::default_random_engine;
using std::uniform_real_distribution;
using std::normal_distribution;

class HW_CUBLAS_GLM{
public:
    InputData* input_data;
    HyperPara* hyper_para;
    
    double* h_weight;
    double* h_pweight;
    double* h_loss;
    
    double* h_gradient;

    double* d_example;
    int* d_label;
    double* d_loss;
    double* d_weight;
    double* d_gradient;
       
    double stepsize;
    double loss;
    double last_loss;
    
    double loss_time;
    double train_time;

    Timer bgd_timer;

    HW_CUBLAS_GLM(int num_tuples, int gradient_size, int num_classes, char* filename,
    	int num_blocks, int num_threads, int batch_size, double decay, double N_0, int iterations);
    ~HW_CUBLAS_GLM();
    void load_data();
    void init_model();
    void train();
    void compute_gradient_and_update_model();
    double get_loss();
};

HW_CUBLAS_GLM::HW_CUBLAS_GLM(int n_tuples, int grad_size, int n_classes, char* fname,
    int n_blocks, int n_threads, int b_size, double d, double N, int iter){
	
	input_data = new InputData(n_tuples, grad_size, n_classes, fname);
    hyper_para = new HyperParaGLM(n_blocks, n_threads, b_size, d, N, iter);
    
    h_weight = (double*) malloc(sizeof(double) * input_data->gradient_size);
    h_pweight = (double*) malloc(sizeof(double) * input_data->gradient_size);
	h_loss = (double*) malloc(sizeof(double) * hyper_para->num_blocks * hyper_para->num_threads);
    h_gradient = (double*) malloc(sizeof(double) * hyper_para->num_blocks * hyper_para->num_threads * input_data->gradient_size);
    
    cudaMalloc(&d_example, sizeof(double) * input_data->num_tuples * input_data->gradient_size);
	cudaMalloc(&d_label, sizeof(int) * input_data->num_tuples);
	cudaMalloc(&d_loss, sizeof(double) * hyper_para->num_blocks * hyper_para->num_threads);
    cudaMalloc(&d_weight, sizeof(double) * input_data->gradient_size);
    cudaMalloc(&d_gradient, sizeof(double) * hyper_para->num_blocks * hyper_para->num_threads * input_data->gradient_size);

    for(int j = 0; j < hyper_para->num_blocks * hyper_para->num_threads; j++)  h_loss[j] = 0.0;
    cudaMemcpy(d_loss, h_loss, sizeof(double) * hyper_para->num_blocks * hyper_para->num_threads, cudaMemcpyHostToDevice);
	
    stepsize = N;
    loss = 0.;
    loss_time = 0.;
    train_time = 0.;
}

HW_CUBLAS_GLM::~HW_CUBLAS_GLM(){
    delete input_data;
    delete hyper_para;
}

void HW_CUBLAS_GLM::load_data(){
    char str[1000];
    char ch;
    
    double* h_example = (float*) malloc(sizeof(float) * input_data->num_tuples * input_data->gradient_size);
	int* h_label = (int*) malloc(sizeof(int) * input_data->num_tuples);
    
    bgd_timer.Restart();
    FILE *file = fopen(input_data->filename, "r");
	for(int i = 0; i < input_data->num_tuples; i++){
		fscanf(file, "%c", &ch);
		for(int j = 0; j < input_data->gradient_size - 1; j++){
			fscanf(file, "%lf,", &h_example[i * input_data->gradient_size + j]);
		}
		fscanf(file, "%f", &h_example[i * input_data->gradient_size + input_data->gradient_size - 1]);
		fscanf(file, "%s", str);
		fscanf(file, "%d", &h_label[i]);
		fgets(str, 1000, file);		
	}
	fclose(file);
    printf("reading_time, %.10f\n", bgd_timer.GetTime());
    
    cudaMemcpy(d_example, h_example, sizeof(double) * input_data->num_tuples * input_data->gradient_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_label, h_label, sizeof(int) * input_data->num_tuples, cudaMemcpyHostToDevice);
	
    free(h_example);
    free(h_label);
}

void HW_CUBLAS_GLM::init_model(){
	double* h_weight = (double*) malloc(sizeof(double) * gradient_size);
    for(int i = 0; i < gradient_size; i++) h_weight[i] = 0;
    cudaMemcpy(d_weight, h_weight, sizeof(double) * gradient_size, cudaMemcpyHostToDevice);
	free(h_weight);
    
    h_pweight = (double*) malloc(sizeof(double) * gradient_size);
}

void HW_CUBLAS_GLM::compute_gradient_and_update_model(){
    // row-major round-robin
	LogisticRegression_row_rr<<<hyper_para->num_blocks, hyper_para->num_threads>>>(d_example, d_label, d_gradient, d_weight, input_data->num_tuples, input_data->gradient_size, stepsize, hyper_para->num_blocks * hypar_para->num_threads);
    //
    cudaDeviceSynchronize();
}

double HW_CUBLAS_GLM::get_loss(){
	// row-major round-robin
    LogisticRegression_loss_row_rr<<<hypar_para->num_blocks, hypar_para->num_threads>>>(d_example, d_label, d_weight, d_loss, input_data->num_tuples, input_data->gradient_size, hypar_para->num_blocks * hypar_para->num_threads);	
    // 
    cudaDeviceSynchronize();
    cudaMemcpy(h_loss, d_loss, sizeof(double) * hypar_para->num_blocks * hypar_para->num_threads, cudaMemcpyDeviceToHost);
    for(int i = 1; i < hypar_para->num_blocks * hypar_para->num_threads; i++)   h_loss[0] += h_loss[i];
    double loss = h_loss[0];
    
    for(int i = 1; i < hypar_para->num_blocks * hypar_para->num_threads; i++)   h_loss[0] += h_loss[i];
    cudaMemcpy(d_loss, h_loss, sizeof(double) * hypar_para->num_blocks * hypar_para->num_threads, cudaMemcpyHostToDevice);
	
    return loss/input_data->num_tuples;    
}

void HW_CUBLAS_GLM::train(){
	loss = get_loss();
	last_loss = loss;
    printf("initial loss,%.10f\n", loss);
    
    for(int i = 0; i < hypar_para->iterations; i++){
		stepsize = hypar_para->N_0 * exp(-hypar_para->decay * i);
		
        if( i != 0){
			cudaMemcpy(h_weight, d_weight, sizeof(double) * input_data->gradient_size, cudaMemcpyDeviceToHost);
			for(int j = 0; j < gradient_size; j++) h_pweight[j] = h_weight[j];
		}
        
		bgd_timer.Restart();
		compute_gradient_and_update_model();
        train_time += bgd_timer.GetTime();
        
        bgd_timer.Restart();
        loss = get_loss();
        loss_time += bgd_timer.GetTime();
        if(loss >= last_loss){
			printf("current_loss = %.10f,", loss);
			cudaMemcpy(d_weight, h_pweight, sizeof(double) * input_data->gradient_size, cudaMemcpyHostToDevice);
        } else{
            last_loss = loss;
        }
	    printf("Iteration,%d,Stepsize,%.12f,%.10f,%.10f,%.12f\n", i+1, stepsize, train_time, loss_time, last_loss);
	}
}

#endif /* MODEL_HW_CUBLAS_GLM_H_ */
