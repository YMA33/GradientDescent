#ifndef NEURALNETS_H_
#define NEURALNETS_H_

#include <vector>

using std::vector;

class NeuralNets{
public:
    vector<int> num_units;
    int num_layers;
    int num_grad;
    
    NeuralNets(int num_layers, vector<int>& units);

};

NeuralNets::NeuralNets(int n_layers, vector<int>& units){
    num_layers = n_layers;
    num_grad = n_layers - 1;
    num_units = units;
}

#endif /* NEURALNETS_H_ */
