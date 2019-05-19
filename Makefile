# the compiler
CC = nvcc -std=c++11 -Xcompiler -fopenmp 

# the directory of ViennaCL library
VCL_PATH =

# the compiler flags that are used when object files are created
CC_CPUFLAGS = -DVIENNACL_WITH_OPENMP
CC_GPUFLAGS =  -DVIENNACL_WITH_CUDA

LDFLAGS_GPU = -lcublas
LDFLAGS_CPU = 

# activation function marco: SIGMOID, TANH, RELU
ACTIVATION_MACRO = SIGMOID 

SRC = ./src

TARGET = ./target

INCLUDE = -I$(VCL_PATH)

all : $(TARGET)/main_cpu.out $(TARGET)/main_gpu.out

clean:
	-rm -f $(TARGET)/*

$(TARGET)/main_cpu.out : $(SRC)/main.cc
	$(CC) $(CC_CPUFLAGS) -D $(ACTIVATION_MACRO) $(SRC)/main.cc -I $(INCLUDE) -o $(TARGET)/main_cpu.out -O3
$(TARGET)/main_gpu.out : $(SRC)/main.cu
	$(CC) $(CC_GPUFLAGS) -D $(ACTIVATION_MACRO) $(SRC)/main.cu -I $(INCLUDE) -o $(TARGET)/main_gpu.out -O3
    
    