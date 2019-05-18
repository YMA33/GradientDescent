# Gradient Descent on Modern Hardware
We perform the first comprehensive experimental study of parallel gradient descent algorithms that investigates the combined impact of three axes -- computing architecture (multi-core CPU or GPU), model update strategy (synchronous or asynchronous), and data sparsity (dense or sparse) -- on three measures --- hardware efficiency, statistical efficiency, and overall time to convergence. Our work supports logistic regression (LR), support vector machines (SVM) and deep neural nets (MLP).

See our [arXiv](https://arxiv.org/abs/1802.08800) paper and the conference paper accepted by [IPDPS2019](http://www.ipdps.org/) for more details. 

Now we are working to design an efficient hybrid gradient descent algorithm.

## Requirements
- NVIDIA's CUDA 9.1
- OpenMP 4.0

## Installation
- [Download and install](https://developer.nvidia.com/cublas) cuBLAS library 
- [Download](http://viennacl.sourceforge.net/viennacl-download.html) and [install](http://viennacl.sourceforge.net/doc/manual-installation.html) ViennaCL 1.7.1

## Build


## Train with our supporting GD algorithms
The datasets can be downloaded from [link](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/).
- Example
