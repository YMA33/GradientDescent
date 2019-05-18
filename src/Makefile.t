all : mlp_dense_bgd mlp_dense_mgd mlp_dense_bgd_cublas mlp_dense_mgd_cublas mlp_dense_svrg_vcl mlp_dense_svrg_hybrid
mlp_dense_bgd : mlp_dense_sigmoid.cppout mlp_dense_sigmoid.out mlp_dense_tanh.cppout mlp_dense_tanh.out mlp_dense_relu.cppout mlp_dense_relu.out
mlp_dense_mgd : mlp_dense_mgd_sigmoid.cppout mlp_dense_mgd_sigmoid.out mlp_dense_mgd_tanh.cppout mlp_dense_mgd_tanh.out mlp_dense_mgd_relu.cppout mlp_dense_mgd_relu.out
mlp_dense_bgd_cublas : mlp_dense_sigmoid_cublas.out mlp_dense_tanh_cublas.out mlp_dense_relu_cublas.out
mlp_dense_mgd_cublas : mlp_dense_mgd_sigmoid_cublas.out mlp_dense_mgd_tanh_cublas.out mlp_dense_mgd_relu_cublas.out
mlp_dense_svrg_vcl : mlp_dense_svrg_vcl_sigmoid.cppout mlp_dense_svrg_vcl_sigmoid.out mlp_dense_svrg_vcl_tanh.cppout mlp_dense_svrg_vcl_tanh.out mlp_dense_svrg_vcl_relu.cppout mlp_dense_svrg_vcl_relu.out
mlp_dense_svrg_hybrid : mlp_dense_svrg_hybrid_sigmoid.out mlp_dense_svrg_hybrid_tanh.out mlp_dense_svrg_hybrid_relu.out
mlp_dense_svrg_hybrid2 : mlp_dense_svrg_hybrid2_sigmoid.out mlp_dense_svrg_hybrid2_tanh.out mlp_dense_svrg_hybrid2_relu.out
mlp_dense_svrg_gpu : mlp_dense_svrg_gpu_sigmoid.out mlp_dense_svrg_gpu_tanh.out mlp_dense_svrg_gpu_relu.out
mlp_dense_mgd_cublas_layer : mlp_dense_mgd_sigmoid_cublas_layer.out

clean :
	rm -f *out
mlp_dense_sigmoid.cppout : mlp_dense.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D SIGMOID mlp_dense.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_sigmoid.cppout -g -O3
mlp_dense_sigmoid.out : mlp_dense.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D SIGMOID mlp_dense.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_sigmoid.out -g -O3
mlp_dense_tanh.cppout : mlp_dense.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D TANH mlp_dense.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_tanh.cppout -g -O3
mlp_dense_tanh.out : mlp_dense.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D TANH mlp_dense.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_tanh.out -g -O3
mlp_dense_relu.cppout : mlp_dense.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D RELU mlp_dense.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_relu.cppout -g -O3
mlp_dense_relu.out : mlp_dense.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D RELU mlp_dense.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_relu.out -g -O3
mlp_dense_mgd_sigmoid.cppout : mlp_dense_mgd.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D SIGMOID mlp_dense_mgd.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_mgd_sigmoid.cppout -g -O3
mlp_dense_mgd_sigmoid.out : mlp_dense_mgd.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D SIGMOID mlp_dense_mgd.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_mgd_sigmoid.out -g -O3
mlp_dense_mgd_tanh.cppout : mlp_dense_mgd.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D TANH mlp_dense_mgd.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_mgd_tanh.cppout -g -O3
mlp_dense_mgd_tanh.out : mlp_dense_mgd.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D TANH mlp_dense_mgd.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_mgd_tanh.out -g -O3
mlp_dense_mgd_relu.cppout : mlp_dense_mgd.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D RELU mlp_dense_mgd.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_mgd_relu.cppout -g -O3
mlp_dense_mgd_relu.out : mlp_dense_mgd.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D RELU mlp_dense_mgd.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_mgd_relu.out -g -O3
mlp_dense_sigmoid_cublas.out : mlp_dense_cublas.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -D SIGMOID -lcublas mlp_dense_cublas.cu -I /usr/local/cuda/samples/common/inc/ -o mlp_dense_sigmoid_cublas.out -g -O3
mlp_dense_tanh_cublas.out : mlp_dense_cublas.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -D TANH -lcublas mlp_dense_cublas.cu -I /usr/local/cuda/samples/common/inc/ -o mlp_dense_tanh_cublas.out -g -O3
mlp_dense_relu_cublas.out : mlp_dense_cublas.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -D RELU -lcublas mlp_dense_cublas.cu -I /usr/local/cuda/samples/common/inc/  -o mlp_dense_relu_cublas.out -g -O3
mlp_dense_mgd_sigmoid_cublas.out : mlp_dense_mgd_cublas.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -D SIGMOID -lcublas mlp_dense_mgd_cublas.cu -I /usr/local/cuda/samples/common/inc/ -o mlp_dense_mgd_sigmoid_cublas.out -g -O3
mlp_dense_mgd_tanh_cublas.out : mlp_dense_mgd_cublas.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -D TANH -lcublas mlp_dense_mgd_cublas.cu -I /usr/local/cuda/samples/common/inc/ -o mlp_dense_mgd_tanh_cublas.out -g -O3
mlp_dense_mgd_relu_cublas.out : mlp_dense_mgd_cublas.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -D RELU -lcublas mlp_dense_mgd_cublas.cu -I /usr/local/cuda/samples/common/inc/  -o mlp_dense_mgd_relu_cublas.out -g -O3
mlp_dense_svrg_vcl_sigmoid.cppout : mlp_dense_svrg_vcl.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D SIGMOID mlp_dense_svrg_vcl.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_vcl_sigmoid.cppout -g -O3
mlp_dense_svrg_vcl_sigmoid.out : mlp_dense_svrg_vcl.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D SIGMOID mlp_dense_svrg_vcl.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_vcl_sigmoid.out -g -O3
mlp_dense_svrg_vcl_tanh.cppout : mlp_dense_svrg_vcl.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D TANH mlp_dense_svrg_vcl.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_vcl_tanh.cppout -g -O3
mlp_dense_svrg_vcl_tanh.out : mlp_dense_svrg_vcl.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D TANH mlp_dense_svrg_vcl.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_vcl_tanh.out -g -O3
mlp_dense_svrg_vcl_relu.cppout : mlp_dense_svrg_vcl.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D RELU mlp_dense_svrg_vcl.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_vcl_relu.cppout -g -O3
mlp_dense_svrg_vcl_relu.out : mlp_dense_svrg_vcl.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D RELU mlp_dense_svrg_vcl.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_vcl_relu.out -g -O3
mlp_dense_svrg_hybrid_sigmoid.out : mlp_dense_svrg_hybrid.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D SIGMOID -lcublas mlp_dense_svrg_hybrid.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_hybrid_sigmoid.out -g -O3
mlp_dense_svrg_hybrid_tanh.out : mlp_dense_svrg_hybrid.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D TANH -lcublas mlp_dense_svrg_hybrid.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_hybrid_tanh.out -g -O3
mlp_dense_svrg_hybrid_relu.out : mlp_dense_svrg_hybrid.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D RELU -lcublas mlp_dense_svrg_hybrid.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_hybrid_relu.out -g -O3
mlp_dense_svrg_hybrid2_sigmoid.out : mlp_dense_svrg_hybrid2.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D SIGMOID -lcublas mlp_dense_svrg_hybrid2.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_hybrid2_sigmoid.out -g -O3
mlp_dense_svrg_hybrid2_tanh.out : mlp_dense_svrg_hybrid2.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D TANH -lcublas mlp_dense_svrg_hybrid2.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_hybrid2_tanh.out -g -O3
mlp_dense_svrg_hybrid2_relu.out : mlp_dense_svrg_hybrid2.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_OPENMP -D RELU -lcublas mlp_dense_svrg_hybrid2.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_hybrid2_relu.out -g -O3
mlp_dense_svrg_gpu_sigmoid.out : mlp_dense_svrg_gpu.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D SIGMOID -lcublas mlp_dense_svrg_gpu.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_gpu_sigmoid.out -g -O3
mlp_dense_svrg_gpu_tanh.out : mlp_dense_svrg_gpu.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D TANH -lcublas mlp_dense_svrg_gpu.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_gpu_tanh.out -g -O3
mlp_dense_svrg_gpu_relu.out : mlp_dense_svrg_gpu.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -DVIENNACL_WITH_CUDA -D RELU -lcublas mlp_dense_svrg_gpu.cu -I /usr/local/cuda/samples/common/inc/  -I /home/yujing/ViennaCL-1.7.1/ -o mlp_dense_svrg_gpu_relu.out -g -O3
mlp_dense_mgd_sigmoid_cublas_layer.out : mlp_dense_mgd_cublas_layer.cu
	nvcc -std=c++11 -Xcompiler -fopenmp -D SIGMOID -lcublas mlp_dense_mgd_cublas_layer.cu -I /usr/local/cuda/samples/common/inc/ -o mlp_dense_mgd_sigmoid_cublas_layer.out -g -O3
	
