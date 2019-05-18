#ifndef HYPERPARA_H_
#define HYPERPARA_H_

class HyperPara{
public:
    int num_threads;
    int batch_size;
    double decay;
    double N_0;
    int iterations;
    int seed;
    int num_batches;
	bool last_batch_processed;  // if #tuples in the last batch is different
    int tuples_last_batch;  // #tuples in the last batch
    
    HyperPara(int num_threads, int batch_size, double decay, double N_0, int iterations, int seed);
    
};

HyperPara::HyperPara(int n_threads, int b_size, double d, double N, int iter, int s){
    num_threads = n_threads;
    batch_size = b_size;
    decay = d;
    N_0 = N;
    iterations = iter;
    seed = s;
    num_batches = 0;
    tuples_last_batch = 0;
    last_batch_processed = true;
}

#endif /* HYPERPARA_H_ */
