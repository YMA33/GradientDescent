export OMP_CPU_BIND=close
export GOMP_CPU_AFFINITY=0-55

# c_init = 0, normal_distribution
# 54->10->5->2

#./mlp_dense_sigmoid.cppout 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/bgd_dense_sigm_cpu
#./mlp_dense_sigmoid.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/bgd_dense_sigm_gpu
./mlp_dense_mgd_sigmoid.cppout 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/mgd_dense_sigm_cpu
#./mlp_dense_mgd_sigmoid.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/mgd_dense_sigm_gpu

#./mlp_dense_tanh.cppout 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/bgd_dense_tanh_cpu
#./mlp_dense_tanh.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/bgd_dense_tanh_gpu
#./mlp_dense_mgd_tanh.cppout 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/mgd_dense_tanh_cpu
#./mlp_dense_mgd_tanh.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/mgd_dense_tanh_gpu

#./mlp_dense_relu.cppout 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/bgd_dense_relu_cpu
#./mlp_dense_relu.out 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/bgd_dense_relu_gpu
#./mlp_dense_mgd_relu.cppout 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/mgd_dense_relu_cpu
#./mlp_dense_mgd_relu.out 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/mgd_dense_relu_gpu

#./mlp_dense_sigmoid_cublas.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/bgd_dense_sigm_cublas
#./mlp_dense_tanh_cublas.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/bgd_dense_tanh_cublas
#./mlp_dense_relu_cublas.out 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/bgd_dense_relu_cublas
#./mlp_dense_mgd_sigmoid_cublas.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/mgd_dense_sigm_cublas
#./mlp_dense_mgd_tanh_cublas.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/mgd_dense_tanh_cublas
#./mlp_dense_mgd_relu_cublas.out 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/mgd_dense_relu_cublas

#./mlp_dense_svrg_vcl_sigmoid.cppout 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_vcl_dense_sigm_cpu
#./mlp_dense_svrg_vcl_tanh.cppout 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/svrg_vcl_dense_tanh_cpu
#./mlp_dense_svrg_vcl_relu.cppout 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/svrg_vcl_dense_relu_cpu

# out of memory
#./mlp_dense_svrg_vcl_sigmoid.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_vcl_dense_sigm_gpu
#./mlp_dense_svrg_vcl_tanh.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/svrg_vcl_dense_tanh_gpu
#./mlp_dense_svrg_vcl_relu.out 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/svrg_vcl_dense_relu_gpu

#./mlp_dense_svrg_hybrid_sigmoid.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_hybrid_dense_sigm
#./mlp_dense_svrg_hybrid_tanh.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/svrg_hybrid_dense_tanh
#./mlp_dense_svrg_hybrid_relu.out 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/svrg_hybrid_dense_relu

#./mlp_dense_svrg_hybrid2_sigmoid.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_hybrid2_dense_sigm
#./mlp_dense_svrg_hybrid2_tanh.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/svrg_hybrid2_dense_tanh
#./mlp_dense_svrg_hybrid2_relu.out 56 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/svrg_hybrid2_dense_relu

#./mlp_dense_svrg_gpu_sigmoid.out 56 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_hybrid_dense_sigm


export GOMP_CPU_AFFINITY=0
#./mlp_dense_sigmoid.cppout 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 5 7890 > results/bgd_dense_sigm_cpu1
#./mlp_dense_mgd_sigmoid.cppout 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 5 7890 > results/mgd_dense_sigm_cpu1
#./mlp_dense_tanh.cppout 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 5 10 > results/bgd_dense_tanh_cpu1
#./mlp_dense_mgd_tanh.cppout 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 5 10 > results/mgd_dense_tanh_cpu1
#./mlp_dense_relu.cppout 1 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 5 0 > results/bgd_dense_relu_cpu1
#./mlp_dense_mgd_relu.cppout 1 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 5 0 > results/mgd_dense_relu_cpu1
#./mlp_dense_mgd_sigmoid_cublas.out 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 5 7890 > results/mgd_dense_sigm_cublas_cpu1
#./mlp_dense_mgd_tanh_cublas.out 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 5 10 > results/mgd_dense_tanh_cublas_cpu1
#./mlp_dense_mgd_relu_cublas.out 1 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 5 0 > results/mgd_dense_relu_cublas_cpu1

#./mlp_dense_svrg_vcl_sigmoid.cppout 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_vcl_dense_sigm_cpu1
#./mlp_dense_svrg_vcl_tanh.cppout 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/svrg_vcl_dense_tanh_cpu1
#./mlp_dense_svrg_vcl_relu.cppout 1 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/svrg_vcl_dense_relu_cpu1

#./mlp_dense_svrg_hybrid_sigmoid.out 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_hybrid_dense_sigm_cpu1
#./mlp_dense_svrg_hybrid_tanh.out 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/svrg_hybrid_dense_tanh_cpu1
#./mlp_dense_svrg_hybrid_relu.out 1 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/svrg_hybrid_dense_relu_cpu1

#./mlp_dense_svrg_hybrid2_sigmoid.out 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/svrg_hybrid2_dense_sigm_cpu1
#./mlp_dense_svrg_hybrid2_tanh.out 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 10 > results/svrg_hybrid2_dense_tanh_cpu1
#./mlp_dense_svrg_hybrid2_relu.out 1 512 581012 54 2 0 0.1 /home/yujing/fully-connected-nn/forest581012.csv 20 0 > results/svrg_hybrid2_dense_relu_cpu1

#./mlp_dense_mgd_sigmoid_cublas_layer.out 1 512 581012 54 2 0 10 /home/yujing/fully-connected-nn/forest581012.csv 20 7890 > results/mgd_dense_sigmoid_cublas_layer
