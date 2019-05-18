#ifndef ALGO_MGD_VCL_MLP_H_
#define ALGO_MGD_VCL_MLP_H_

#include <stdio.h>
#include <math.h>
#include <random>

#include "../util/Timer.h"
#include "../util/InputData.h"
#include "../util/HyperPara.h"
#include "../util/NeuralNets.h"
#include "../train/train_vcl.h"

using std::default_random_engine;
using std::uniform_real_distribution;
using std::normal_distribution;
using std::cout;
using std::endl;

class MGD_VCL_MLP{
public:
	InputData* input_data;
    HyperPara* hyper_para;
    NeuralNets* neural_nets;

    vector<viennacl::matrix<double, viennacl::row_major> > vc_mgd_weight;
    vector<vector<viennacl::matrix<double, viennacl::row_major> > > vc_mgd_gradient;
    vector<vector<viennacl::matrix<double, viennacl::row_major> > > vc_mgd_y;
    vector<viennacl::matrix<double, viennacl::row_major> > vc_mgd_label;
    
    vector<vector<viennacl::matrix<double> > > dloss_dy;
    vector<vector<viennacl::matrix<double> > > dloss_dx;

    double stepsize;
    double loss;
    double forward_time;
    double backprop_time;

    Timer mgd_timer;

    MGD_VCL_MLP(int num_tuples, int gradient_size, int num_classes, char* filename,
            int num_threads, int batch_size, double decay, double N_0, int iterations, int seed,
            int num_layers, vector<int>& units);
    ~MGD_VCL_MLP();
    void load_data();
    void init_model(int c_init);
    double get_loss();
    double get_loss_para();
    void train();

    void compute_gradient(int batch_idx, int processed_tuples);
    void update_model();
    void forward(int batch_idx, int processed_tuples);
};

MGD_VCL_MLP::MGD_VCL_MLP(int n_tuples, int grad_size, int n_classes, char* fname,
    int n_threads, int b_size, double d, double N, int iter, int s,
    int n_layers, vector<int>& units){

    input_data = new InputData(n_tuples, grad_size, n_classes, fname);
    hyper_para = new HyperPara(n_threads, b_size, d, N, iter, s);
    neural_nets = new NeuralNets(n_layers, units);

    hyper_para->num_batches = input_data->num_tuples/hyper_para->batch_size + 1;
    hyper_para->tuples_last_batch = input_data->num_tuples - (hyper_para->num_batches-1)*hyper_para->batch_size;
    if(hyper_para->tuples_last_batch==0){
    	hyper_para->num_batches--;
    	hyper_para->last_batch_processed = false;
    }
    //cout<<"num_batches: " << hyper_para->num_batches
    //    	<<", #tuples in last batch: "<< hyper_para->tuples_last_batch<<endl;

    vc_mgd_weight.resize(neural_nets->num_grad);
    vc_mgd_gradient.resize(hyper_para->num_batches);
    vc_mgd_y.resize(hyper_para->num_batches);
    vc_mgd_label.resize(hyper_para->num_batches);

    dloss_dy.resize(hyper_para->num_batches);
    dloss_dx.resize(hyper_para->num_batches);

    #pragma omp parallel for schedule(dynamic)
    for(int i = 0; i < hyper_para->num_batches; i++){
    	int processed_tuples = hyper_para->batch_size;
    	if( i == hyper_para->num_batches-1 && hyper_para->last_batch_processed)
    		processed_tuples = hyper_para->tuples_last_batch;
    	vc_mgd_label[i] = viennacl::matrix<double,viennacl::row_major>(processed_tuples, input_data->num_classes);
    	vc_mgd_y[i].resize(neural_nets->num_layers);
    	vc_mgd_y[i][0] = viennacl::matrix<double, viennacl::row_major>(processed_tuples, input_data->gradient_size);
    	dloss_dy[i].resize(neural_nets->num_grad);
    	dloss_dx[i].resize(neural_nets->num_grad);
    }

    stepsize = N;
    loss = 0.;
    forward_time = 0.;
    backprop_time = 0.;
}

MGD_VCL_MLP::~MGD_VCL_MLP(){
    delete input_data;
    delete hyper_para;
    delete neural_nets;
}

void MGD_VCL_MLP::load_data(){
	char str[1000];
	double val, y_val;
	vector<double>* mgd_data = new vector<double>[hyper_para->num_batches];
	vector<double>* mgd_label = new vector<double>[hyper_para->num_batches];

	mgd_timer.Restart();
	FILE *file = fopen(input_data->filename, "r");
	for(int i = 0; i < hyper_para->num_batches; i++){
		mgd_data[i].resize(vc_mgd_y[i][0].internal_size(), 0.);
		mgd_label[i].resize(vc_mgd_label[i].internal_size(), 0.);
		//for(int j = 0; j < vc_mgd_y[i][0].internal_size(); j++)	mgd_data[i].push_back(0.);
		//for(int j = 0; j < vc_mgd_label[i].internal_size(); j++)	mgd_label[i].push_back(0.);
        for(int j = 0; j< hyper_para->batch_size
	        && i*hyper_para->batch_size+j< input_data->num_tuples; j++){
            fscanf(file, "%lf", &y_val);
            if(y_val == -1.){ // [1,0]
                mgd_label[i][viennacl::row_major::mem_index(
                	j,0,vc_mgd_label[i].internal_size1(),vc_mgd_label[i].internal_size2())] = 1.;
                mgd_label[i][viennacl::row_major::mem_index(
                	j,1,vc_mgd_label[i].internal_size1(), vc_mgd_label[i].internal_size2())] = 0.;
            } else{ // [0,1]
                mgd_label[i][viennacl::row_major::mem_index(
                	j,0,vc_mgd_label[i].internal_size1(), vc_mgd_label[i].internal_size2())] = 0.;
                mgd_label[i][viennacl::row_major::mem_index(
                	j,1,vc_mgd_label[i].internal_size1(), vc_mgd_label[i].internal_size2())] = 1.;
            }
    		for(int k = 0; k < input_data->gradient_size; k++){
    			fscanf(file, ",%lf", &val);
    			mgd_data[i][viennacl::row_major::mem_index(
    				j,k,vc_mgd_y[i][0].internal_size1(),vc_mgd_y[i][0].internal_size2())] = val;
		    }
    		fgets(str, 1000, file);
	    }
	}
	fclose(file);
	printf("reading_time, %.10f\n", mgd_timer.GetTime());

	for(int i = 0; i < hyper_para->num_batches; i++){
        viennacl::fast_copy(&mgd_data[i][0], &mgd_data[i][0] + mgd_data[i].size(), vc_mgd_y[i][0]);
        viennacl::fast_copy(&mgd_label[i][0], &mgd_label[i][0] + mgd_label[i].size(), vc_mgd_label[i]);
    }

	delete[] mgd_data;
	delete[] mgd_label;
}

void MGD_VCL_MLP::init_model(int c_init){
    vector<vector<double> >* h_weight = new vector<vector<double> >[neural_nets->num_grad];
    double t_dist = 0.;
	
    switch(c_init){
	case 0:	// normal_distribution
		t_dist = int(sqrt(neural_nets->num_units[0] + neural_nets->num_units[1]))+1;
		//cout<<"t_dist: "<<t_dist<<endl;
		for(int i = 0; i < neural_nets->num_grad; i++){
			default_random_engine generator(hyper_para->seed);
			normal_distribution<double> distributions(
				0, sqrt(2.*t_dist/(neural_nets->num_units[i]+neural_nets->num_units[i+1])));
			//cout<<"layer: "<<i<<", normal_dist: 0;"<<
			//	sqrt(2.*t_dist/(neural_nets->num_units[i]+neural_nets->num_units[i+1]))<<endl;
			for(int j = 0; j < neural_nets->num_units[i]; j++){
				vector<double> t_weight(neural_nets->num_units[i+1]);
				for(int k = 0; k < neural_nets->num_units[i+1]; k++){
					t_weight[k] = distributions(generator);
				}
				h_weight[i].push_back(t_weight);
			}
		}
		break;
	case 1:	// uniform_distribution for sigmoid
		for(int i = 0; i < neural_nets->num_grad; i++){
			t_dist = sqrt(6./(neural_nets->num_units[i]+neural_nets->num_units[i+1]));
			default_random_engine generator(hyper_para->seed);
			uniform_real_distribution<double> distributions(-t_dist,t_dist);
			//cout<<"layer: "<<i<<", uniform_dist for sigmoid: "<<
			//	-t_dist<<";"<<t_dist<<endl;
			for(int j = 0; j < neural_nets->num_units[i]; j++){
				vector<double> t_weight(neural_nets->num_units[i+1]);
				for(int k = 0; k < neural_nets->num_units[i+1]; k++){
					t_weight[k] = distributions(generator);
				}
				h_weight[i].push_back(t_weight);
			}
		}
		break;
	case 2:	// uniform_distribution for tanh
		for(int i = 0; i < neural_nets->num_grad; i++){
			t_dist = 4 * sqrt(6./(neural_nets->num_units[i]+neural_nets->num_units[i+1]));
			default_random_engine generator(hyper_para->seed);
			uniform_real_distribution<double> distributions(-t_dist,t_dist);
			//cout<<"layer: "<<i<<", uniform_dist for tanh: "<<
			//	-t_dist<<";"<<t_dist<<endl;
			for(int j = 0; j < neural_nets->num_units[i]; j++){
				vector<double> t_weight(neural_nets->num_units[i+1]);
				for(int k = 0; k < neural_nets->num_units[i+1]; k++){
					t_weight[k] = distributions(generator);
				}
				h_weight[i].push_back(t_weight);
			}
		}
	}

	for(int i = 0; i < neural_nets->num_grad; i++){
        vc_mgd_weight[i] = viennacl::matrix<double, viennacl::row_major>(
        	neural_nets->num_units[i], neural_nets->num_units[i+1]);
        viennacl::copy(h_weight[i], vc_mgd_weight[i]);
    }

    for(int i = 0; i < hyper_para->num_batches; i++){
    	vc_mgd_gradient[i].resize(neural_nets->num_grad);
    	for(int j = 0; j < neural_nets->num_grad; j++){
    		vc_mgd_gradient[i][j] = viennacl::matrix<double, viennacl::row_major>(
    			neural_nets->num_units[j], neural_nets->num_units[j+1]);
    	}
    }

	delete[] h_weight;
}

void MGD_VCL_MLP::compute_gradient(int batch_idx, int processed_tuples){
	compute_gradient_bgd_vcl(vc_mgd_label[batch_idx], vc_mgd_y[batch_idx], vc_mgd_weight, neural_nets,
		processed_tuples, vc_mgd_gradient[batch_idx], dloss_dy[batch_idx], dloss_dx[batch_idx]);
}

void MGD_VCL_MLP::update_model(){
    update_model_bgd_vcl(vc_mgd_weight, vc_mgd_gradient[j], neural_nets->num_grad, stepsize);    
}

void MGD_VCL_MLP::forward(int batch_idx, int processed_tuples){
	forward_mgd_vcl(vc_mgd_y[batch_idx], vc_mgd_weight, neural_nets->num_units, processed_tuples);
}

double MGD_VCL_MLP::get_loss(){
    double loss = get_loss_mgd_vcl(vc_mgd_label, vc_mgd_y, vc_mgd_weight, neural_nets->num_units, hyper_para);
	return loss / input_data->num_tuples;
	
}

double MGD_VCL_MLP::get_loss_para(){
	double loss = get_loss_mgd_vcl_para(vc_mgd_label, vc_mgd_y, vc_mgd_weight, neural_nets->num_units, hyper_para);
	return loss / input_data->num_tuples;
}

void MGD_VCL_MLP::train(){
	loss = get_loss();
	printf("initial loss,%.10f\n",loss);
	//loss = get_loss_para();
	//printf("initial loss(para),%.10f\n",loss);

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
		loss = get_loss();
        //loss = get_loss_para();
	    forward_time += mgd_timer.GetTime();

	    printf("iter,%d,stepsize,%.10f,backprop_time,%.10f,forward_time,%.10f,loss,%.10f\n",
	    	i+1, stepsize, backprop_time, forward_time, loss);
	}
}

#endif /* ALGO_MGD_VCL_MLP_H_ */
