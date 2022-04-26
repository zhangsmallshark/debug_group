CC = /usr/bin/g++

LD_FLAGS = -lrt

CUDA_PATH       ?= /data/lab/tao/chengming/cuda/cuda-11.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH   ?= $(CUDA_PATH)/lib

CUDNN_PATH     ?= /data/lab/tao/chengming/cudnn-8.0.4
CUDNN_INC_PATH ?= $(CUDNN_PATH)/include
CUDNN_LIB_PATH ?= $(CUDNN_PATH)/lib64

# CUDA code generation flags
GENCODE_FLAGS := -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_80,code=sm_80

# Common binaries
NVCC ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifeq ($(shell uname), Darwin)
	LDFLAGS := -Xlinker -rpath $(CUDA_LIB_PATH) -L$(CUDA_LIB_PATH) -lcudart -lcufft -lcublas
	CCFLAGS := -arch $(OS_ARCH)
else
	ifeq ($(OS_SIZE),32)
		LDFLAGS := -L $(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -L$(CUDNN_LIB_PATH) -lcudnn
		CCFLAGS := -m32 -std=c++11
	else
		CUDA_LIB_PATH := $(CUDA_LIB_PATH)64
		LDFLAGS       := -L $(CUDA_LIB_PATH) -lcudart -lcufft -lcublas -L$(CUDNN_LIB_PATH) -lcudnn
		CCFLAGS       := -m64 -std=c++11
	endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
	NVCCFLAGS := -m32 -std=c++11
else
	NVCCFLAGS := -m64 -std=c++11
endif

TARGETS = group-conv

all: $(TARGETS)

group-conv: group-conv.cu
	$(NVCC) $(CCFLAGS) $(GENCODE_FLAGS) -O3 $(LDFLAGS) -I$(CUDA_INC_PATH) -I$(CUDNN_INC_PATH) $^ -o $@

clean:
	rm -f *.o $(TARGETS)

again: clean $(TARGETS)