#ifndef HYPERPARA_GLM_H_
#define HYPERPARA_GLM_H_

class HyperParaGLM{
public:
    int num_blocks;
    int num_threads;
    int batch_size;
    double decay;
    double N_0;
    int iterations;
    int num_batches;
	//bool last_batch_processed;  // if #tuples in the last batch is different
    //int tuples_last_batch;  // #tuples in the last batch
    
    HyperParaGLM(int num_blocks, int num_threads, int batch_size, double decay, double N_0, int iterations);
    
};

HyperParaGLM::HyperParaGLM(int n_blocks, int n_threads, int b_size, double d, double N, int iter){
    num_batches = n_blocks;
    num_threads = n_threads;
    batch_size = b_size;
    decay = d;
    N_0 = N;
    iterations = iter;
    num_batches = 0;
    tuples_last_batch = 0;
    last_batch_processed = true;
}

#endif /* HYPERPARA_H_ */
