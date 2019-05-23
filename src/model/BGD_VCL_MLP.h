#ifndef MODEL_BGD_VCL_MLP_H_
#define MODEL_BGD_VCL_MLP_H_

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

class BGD_VCL_MLP{
public:
	InputData* input_data;
    HyperPara* hyper_para;
    NeuralNets* neural_nets;

    vector<viennacl::matrix<double, viennacl::row_major> > vc_bgd_weight;
    vector<viennacl::matrix<double, viennacl::row_major> > vc_bgd_gradient;
    // data maintained on each layer
    vector<viennacl::matrix<double, viennacl::row_major> > vc_bgd_y;
    viennacl::matrix<double, viennacl::row_major> vc_bgd_label;
    
    vector<viennacl::matrix<double> > dloss_dy;
    vector<viennacl::matrix<double> > dloss_dx;

    double stepsize;
    double loss;
    double forward_time;
    double backprop_time;

    Timer bgd_timer;

    BGD_VCL_MLP(int num_tuples, int gradient_size, int num_classes, char* filename,
            int num_threads, int batch_size, double decay, double N_0, int iterations, int seed,
            int num_layers, vector<int>& units);
    ~BGD_VCL_MLP();
    void load_data();
    void init_model(int c_init);
    void train();
    void compute_gradient();
    void update_model();
    double get_loss();
};

BGD_VCL_MLP::BGD_VCL_MLP(int n_tuples, int grad_size, int n_classes, char* fname,
    int n_threads, int b_size, double d, double N, int iter, int s,
    int n_layers, vector<int>& units){

    input_data = new InputData(n_tuples, grad_size, n_classes, fname);
    hyper_para = new HyperPara(n_threads, b_size, d, N, iter, s);
    neural_nets = new NeuralNets(n_layers, units);

    vc_bgd_weight.resize(neural_nets->num_grad);
    vc_bgd_gradient.resize(neural_nets->num_grad);
    vc_bgd_y.resize(neural_nets->num_layers);
    dloss_dy.resize(neural_nets->num_grad);
    dloss_dx.resize(neural_nets->num_grad);

    vc_bgd_y[0] = viennacl::matrix<double, viennacl::row_major>(input_data->num_tuples, input_data->gradient_size);
    vc_bgd_label = viennacl::matrix<double, viennacl::row_major>(input_data->num_tuples, input_data->num_classes);

    stepsize = N;
    loss = 0.;
    forward_time = 0.;
    backprop_time = 0.;
}

BGD_VCL_MLP::~BGD_VCL_MLP(){
    delete input_data;
    delete hyper_para;
    delete neural_nets;
}

void BGD_VCL_MLP::load_data(){
	char str[1000];
	double val, y_val;
    vector<double> bgd_data(vc_bgd_y[0].internal_size());
	vector<double> bgd_label(vc_bgd_label.internal_size());
	
    bgd_timer.Restart();
	FILE *file = fopen(input_data->filename, "r");
	for(int i = 0; i < input_data->num_tuples; i++){
		fscanf(file, "%lf", &y_val);
		if(y_val == -1.){ // [1,0]
			bgd_label[viennacl::row_major::mem_index(
				i,0,vc_bgd_label.internal_size1(), vc_bgd_label.internal_size2())] = 1.;
			bgd_label[viennacl::row_major::mem_index(
				i,1,vc_bgd_label.internal_size1(), vc_bgd_label.internal_size2())] = 0.;
		} else{ // [0,1]
			bgd_label[viennacl::row_major::mem_index(
				i,0,vc_bgd_label.internal_size1(), vc_bgd_label.internal_size2())] = 0.;
			bgd_label[viennacl::row_major::mem_index(
				i,1,vc_bgd_label.internal_size1(), vc_bgd_label.internal_size2())] = 1.;
		}
		for(int j = 0; j < input_data->gradient_size; j++){
			fscanf(file, ",%lf", &val);
			bgd_data[viennacl::row_major::mem_index(
				i, j, vc_bgd_y[0].internal_size1(), vc_bgd_y[0].internal_size2())] = val;
		}
		fgets(str, 1000, file);
	}
	fclose(file);
	printf("reading_time, %.10f\n", bgd_timer.GetTime());
	
    viennacl::fast_copy(&bgd_data[0], &bgd_data[0] + bgd_data.size(), vc_bgd_y[0]);
	viennacl::fast_copy(&bgd_label[0], &bgd_label[0] + bgd_label.size(), vc_bgd_label);
}

void BGD_VCL_MLP::init_model(int c_init){
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
	case -1:
		for(int i = 0; i < neural_nets->num_grad; i++){
			for(int j = 0; j < neural_nets->num_units[i]; j++){
				vector<double> t_weight(neural_nets->num_units[i+1]);
				for(int k = 0; k < neural_nets->num_units[i+1]; k++){
					t_weight[k] = 0.;
				}
				h_weight[i].push_back(t_weight);
			}
		}
	}

	for(int i = 0; i < neural_nets->num_grad; i++){
        vc_bgd_weight[i] = viennacl::matrix<double, viennacl::row_major>(
        	neural_nets->num_units[i], neural_nets->num_units[i+1]);
        vc_bgd_gradient[i] = viennacl::matrix<double, viennacl::row_major>(
        	neural_nets->num_units[i], neural_nets->num_units[i+1]);
        viennacl::copy(h_weight[i], vc_bgd_weight[i]);
    }
	delete[] h_weight;
}

void BGD_VCL_MLP::compute_gradient(){
	compute_gradient_bgd_vcl(vc_bgd_label, vc_bgd_y, vc_bgd_weight, neural_nets,
		input_data->num_tuples, vc_bgd_gradient, dloss_dy, dloss_dx);
}

double BGD_VCL_MLP::get_loss(){
	return get_loss_bgd_vcl(vc_bgd_label, vc_bgd_y, vc_bgd_weight, neural_nets->num_units, input_data->num_tuples);
}

void BGD_VCL_MLP::update_model(){
    update_model_bgd_vcl(vc_bgd_weight, vc_bgd_gradient, neural_nets->num_grad, stepsize);
}

void BGD_VCL_MLP::train(){
	loss = get_loss();
	printf("initial loss,%.10f\n",loss);
	for(int i = 0; i < hyper_para->iterations; i++){
		stepsize = hyper_para->N_0 * exp(-hyper_para->decay * i);

		bgd_timer.Restart();
	    compute_gradient();
        update_model();
	    backprop_time += bgd_timer.GetTime();

	    bgd_timer.Restart();
	    loss = get_loss();
	    forward_time += bgd_timer.GetTime();
	    printf("iter,%d,stepsize,%.10f,backprop_time,%.10f,forward_time,%.10f,loss,%.10f\n",
	    	i+1, stepsize, backprop_time, forward_time, loss);
	}
}

#endif /* MODEL_BGD_VCL_MLP_H_ */
