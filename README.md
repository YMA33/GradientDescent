# Gradient Descent on Modern Hardware
We perform the first comprehensive experimental study of parallel gradient descent algorithms that investigates the combined impact of three axes -- computing architecture (multi-core CPU or GPU), model update strategy (synchronous or asynchronous), and data sparsity (dense or sparse) -- on three measures --- hardware efficiency, statistical efficiency, and overall time to convergence. Our work supports logistic regression (LR), support vector machines (SVM) and deep neural nets (MLP). We draw several interesting findings from our experiments:
- On synchronous SGD, GPU always outperforms parallel CPU in hardware efficiency and, consequently, in time to convergence. For LR and SVM, the difference is minimal for small low-dimensional datasets and increases with dimensionality and sparsity—for a maximum speedup of 5.66X. For MLP, the speedup is at least 4X in all the cases. However, this speedup is nowhere close to the degree of parallelism gap between CPU and GPU. 

![Synchronous SGD performance to 1% convergence error. The best values for each dataset are underlined.](/figures/sync-gd.png)
*Synchronous SGD performance to 1% convergence error. The best values for each dataset are underlined.*

- On asynchronous SGD, CPU is undoubtedly the optimal solution, outperforming GPU in time to convergence even when the GPU has a speedup of 10X or more. The main reason is the complex interaction between hardware and statistical efficiency under asynchronous parallelism. For MLP, the speedup of parallel CPU over GPU is always 6X or larger.

![Asynchronous SGD performance to 1% convergence error. The best values for each dataset are underlined.](/figures/async-gd.png)
*Asynchronous SGD performance to 1% convergence error. The best values for each dataset are underlined.*

- While GPU is the optimal architecture for synchronous SGD and CPU is optimal for asynchronous SGD, choosing the better of synchronous GPU and asynchronous CPU is task- and dataset-dependent. The choice between these two mirrors the comparison between BGD and SGD.

![Time to convergence comparison between synchronous GPU and asynchronous CPU.](/figures/comparison-sync_async.png)
*Time to convergence comparison between synchronous GPU and asynchronous CPU.*

- Our synchronous SGD implementations provide similar or better speedup than TensorFlow and BIDMach when executed on GPU and parallel CPU. This confirms that parallel CPU should be considered as a competitive alternative for training machine learning models with SGD.


See our [arXiv](https://arxiv.org/abs/1802.08800) paper and the conference paper accepted by [IPDPS2019](http://www.ipdps.org/) for more details. 

Now we are working to design an efficient hybrid gradient descent algorithm.

## Requirements
- NVIDIA's CUDA 9.1
- OpenMP 4.0

## Installation
- [Download and install](https://developer.nvidia.com/cublas) cuBLAS library 
- [Download](http://viennacl.sourceforge.net/viennacl-download.html) and [install](http://viennacl.sourceforge.net/doc/manual-installation.html) ViennaCL 1.7.1

## Build
To configure and build, do the following:
1. `cd ./GradientDescent`
2. edit 'VCL_PATH' in the Makefile, e.g., /home/user/ViennaCL
3. `make` 
 
## Train with our supporting GD algorithms
The datasets can be downloaded from [link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
- Example
