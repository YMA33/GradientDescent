#ifndef TRAIN_VCL_H_
#define TRAIN_VCL_H_
#include <vector>

#include "activation/activation_vcl.h"
#include "loss/loss_vcl.h"

#ifdef VIENNACL_WITH_OPENMP
	#include "activation/activation_vcl_cpu.h"
	#include "loss/loss_vcl_cpu.h"
#endif

#ifdef VIENNACL_WITH_CUDA
	#include "activation/activation_vcl_gpu.h"
	#include "loss/loss_vcl_gpu.h"
#endif

using std::cout;
using std::endl;

////////////////////////////////////////
void compute_gradient_bgd_vcl(
    viennacl::matrix<double>& vc_label,
    vector<viennacl::matrix<double> >& vc_y,
    vector<viennacl::matrix<double> >& vc_weight,
    NeuralNets* nn, int num_tuples,
    vector<viennacl::matrix<double> >& vc_gradient,
	vector<viennacl::matrix<double> >& dloss_dy,
	vector<viennacl::matrix<double> >& dloss_dx){
    
    int i = nn->num_grad;

    dloss_dy[i-1] = der_sce(vc_label, vc_y[i], num_tuples, nn->num_units[i]);
    dloss_dx[i-1] = dloss_dy[i-1];

    vc_gradient[i-1] = viennacl::linalg::prod(viennacl::trans(vc_y[i-1]), dloss_dx[i-1]);

    for(i = nn->num_grad-1; i > 0; i--){
        dloss_dy[i-1] = viennacl::linalg::prod(dloss_dx[i], viennacl::trans(vc_weight[i]));
		#ifdef SIGMOID
        dloss_dx[i-1] = viennacl::linalg::element_prod(dloss_dy[i-1], der_sigmoid(vc_y[i], num_tuples, nn->num_units[i]));
		#endif
		#ifdef TANH
        dloss_dx[i-1] = viennacl::linalg::element_prod(dloss_dy[i-1], der_tanh(vc_y[i], num_tuples, nn->num_units[i]));
		#endif
		#ifdef RELU
        dloss_dx[i-1] = viennacl::linalg::element_prod(dloss_dy[i-1], der_relu(vc_y[i], num_tuples, nn->num_units[i]));
		#endif
        vc_gradient[i-1] = viennacl::linalg::prod(viennacl::trans(vc_y[i-1]), dloss_dx[i-1]);
    }
}
////////////////////////////////////////
void update_model_bgd_vcl(
    vector<viennacl::matrix<double> >& vc_weight,
    vector<viennacl::matrix<double> >& vc_gradient,
    int num_grad, double learning_rate){
    for(int i = 0; i < num_grad; i++){
        vc_weight[i] -= learning_rate*vc_gradient[i];
    }
}
////////////////////////////////////////
void forward_mgd_vcl(
    vector<viennacl::matrix<double> >& vc_y,
    vector<viennacl::matrix<double> >& vc_weight,
    vector<int>& num_units,
    unsigned int processed_tuples){
    int i;
    for(i = 1; i < num_units.size()-1; i++){
		#ifdef SIGMOID
        vc_y[i] = sigmoid(viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]), processed_tuples, num_units[i]);
		#endif
		#ifdef TANH
    	vc_y[i] = tanh(viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]), processed_tuples, num_units[i]);
		#endif
		#ifdef RELU
    	vc_y[i] = relu(viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]), processed_tuples, num_units[i]);
		#endif
    }
    vc_y[i] = viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]);
    vc_y[i] = softmax(vc_y[i], processed_tuples, num_units[i]);
}
////////////////////////////////////////
double get_loss_bgd_vcl(
    viennacl::matrix<double>& vc_label,
    vector<viennacl::matrix<double> >& vc_y,
    vector<viennacl::matrix<double> >& vc_weight,
    vector<int>& num_units,
    int num_tuples){
    int i;
    for(i = 1; i < num_units.size()-1; i++){
		#ifdef SIGMOID
    	vc_y[i] = sigmoid(viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]), num_tuples, num_units[i]);
		#endif
		#ifdef TANH
        vc_y[i] = tanh(viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]), num_tuples, num_units[i]);
		#endif
		#ifdef RELU
        vc_y[i] = relu(viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]), num_tuples, num_units[i]);
		#endif
    }
    vc_y[i] = viennacl::linalg::prod(vc_y[i-1],vc_weight[i-1]);
    vc_y[i] = softmax(vc_y[i], num_tuples, num_units[i]);
    return cross_entropy(vc_y[i], vc_label, num_tuples);
}
////////////////////////////////////////
double get_loss_mgd_vcl(
    vector<viennacl::matrix<double> >& vc_label,
    vector<vector<viennacl::matrix<double> > >& vc_y,
    vector<viennacl::matrix<double> >& vc_weight,
    vector<int>& num_units, HyperPara* hpara){
    double loss = 0.;
    int i, j, processed_tuples = hpara->batch_size;
    for(i = 0; i < hpara->num_batches; i++){
        if(i == hpara->num_batches-1 && hpara->last_batch_processed)  processed_tuples = hpara->tuples_last_batch;
        for(j = 1; j < num_units.size()-1; j++){
			#ifdef SIGMOID
        	vc_y[i][j] = sigmoid(viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]), processed_tuples, num_units[j]);
			#endif
			#ifdef TANH
        	vc_y[i][j] = tanh(viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]), processed_tuples, num_units[j]);
			#endif
			#ifdef RELU
        	vc_y[i][j] = relu(viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]), processed_tuples, num_units[j]);
			#endif
        }
        vc_y[i][j] = viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]);
        vc_y[i][j] = softmax(vc_y[i][j], processed_tuples, num_units[j]);
        loss += processed_tuples * cross_entropy(vc_y[i][j], vc_label[i], processed_tuples);
    }
    return loss;
}
////////////////////////////////////////
double get_loss_mgd_vcl_para(
    vector<viennacl::matrix<double> >& vc_label,
    vector<vector<viennacl::matrix<double> > >& vc_y,
    vector<viennacl::matrix<double> >& vc_weight,
    vector<int>& num_units, HyperPara* hpara){
    double loss = 0.;
    int i;
    viennacl::vector<double> t_loss(hpara->num_batches);
    #pragma omp parallel for
    for(i = 0; i < hpara->num_batches; i++){
    	int processed_tuples = hpara->batch_size;
        if(i == hpara->num_batches-1 && hpara->last_batch_processed)	processed_tuples = hpara->tuples_last_batch;
        int j;
        for(j = 1; j < num_units.size()-1; j++){
			#ifdef SIGMOID
        	vc_y[i][j] = sigmoid(viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]), processed_tuples, num_units[j]);
			#endif
			#ifdef TANH
        	vc_y[i][j] = tanh(viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]), processed_tuples, num_units[j]);
			#endif
			#ifdef RELU
        	vc_y[i][j] = relu(viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]), processed_tuples, num_units[j]);
			#endif
        }
        vc_y[i][j] = viennacl::linalg::prod(vc_y[i][j-1],vc_weight[j-1]);
        vc_y[i][j] = softmax(vc_y[i][j], processed_tuples, num_units[j]);
        t_loss[i] = processed_tuples * cross_entropy(vc_y[i][j], vc_label[i], processed_tuples);
    }
    for(i = 0; i < hpara->num_batches; i++)    loss += t_loss[i];
    return loss;
}
////////////////////////////////////////
#endif
