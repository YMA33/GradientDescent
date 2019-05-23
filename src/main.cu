#include <omp.h>
#include "algo/BGD_CUBLAS_MLP.h"

int main(int argc, char* argv[]){
	if(argc != 11){
		printf("<NUM_THREADS>, <BATCH_SIZE>, <NUM_TUPLES>, <GRADIENT_SIZE>, <NUM_CLASSES>, "
			"<DECAY>, <STEP_SIZE>, <FILE_NAME>, <ITERATIONS>, <SEED>\n");
		return 0;
	}
	int num_threads = atoi(argv[1]);
    int batch_size = atoi(argv[2]);
	int num_tuples = atoi(argv[3]);
	int gradient_size = atoi(argv[4]);
    int num_classes = atoi(argv[5]);
	double decay = atof(argv[6]);
	double N_0 = atof(argv[7]);
	char* filename = argv[8];
	int iterations = atoi(argv[9]);
    int seed = atoi(argv[10]);
	printf("num_threads: %d, batch_size: %d, num_tuples: %d, gradient_size: %d, num_classes:%d, "
		"stepsize: (%.10f*e^(-%.10f*i)), filename: %s, iterations: %d, seed, %d\n", num_threads,
		batch_size, num_tuples, gradient_size, num_classes, N_0, decay, filename, iterations, seed);
////////////////////////////////////////
    vector<int> num_units;
    int num_layers = 4;
    num_units.push_back(gradient_size);
    num_units.push_back(4096);
    num_units.push_back(4096);
    num_units.push_back(num_classes);
    printf("%d->4096->4096->%d\n", gradient_size, num_classes);
////////////////////////////////////////
    omp_set_num_threads(num_threads);
    Timer timer_tot;
    timer_tot.Restart();
////////////////////////////////////////
    BGD_CUBLAS_MLP* nn = new BGD_CUBLAS_MLP(num_tuples, gradient_size, num_classes, filename, num_threads,
    	batch_size, decay, N_0, iterations, seed, num_layers, num_units);
    nn->load_data();
    nn->init_model(0);
    nn->train();
    delete nn;
////////////////////////////////////////
	printf("Total time,%.10f\n", timer_tot.GetTime());
	return 0;
}
