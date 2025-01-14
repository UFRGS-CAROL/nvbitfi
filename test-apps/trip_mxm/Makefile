CC=g++
NVCC=/usr/local/cuda/bin/nvcc

PRECISION=single
DEF_SIZE=256

LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

ifeq ($(LOWERCASED_PRECISION),half)
#ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
#ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
else
ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Tesla K40
#ARCH+= 	-gencode arch=compute_53,code=[sm_53,compute_53]	# Tegra X1
#ARCH+= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
#ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
endif

NVCC_FLAGS= $(ARCH)

RADDIR=/home/carol/radiation-benchmarks
NVIDIA_INC=/usr/local/cuda/include/,./
NVIDIA_LIB=/usr/local/cuda/lib64/
LOGHELPER_INC=$(RADDIR)/src/include/
LOGHELPER_LIB=$(RADDIR)/src/include/

SOURCES=./cuda_trip_mxm.cu
EXEC_RADIX=./cuda_trip_mxm_
EXEC=$(EXEC_RADIX)$(LOWERCASED_PRECISION)


DEF_A_INPUT=$(RADDIR)/data/gemm/mxm_$(LOWERCASED_PRECISION)_A_8192.matrix
DEF_B_INPUT=$(RADDIR)/data/gemm/mxm_$(LOWERCASED_PRECISION)_B_8192.matrix
DEF_GOLD=$(RADDIR)/data/gemm/mxm_$(LOWERCASED_PRECISION)_GOLD_$(DEF_TEST_SIZE).matrix

MATRIX_NAMES=$(DEF_A_INPUT) $(DEF_B_INPUT) $(DEF_GOLD) ./mxm*.matrix

all: $(EXEC)

check-benchtype:
	echo "syntax: make PRECISION=<double|half|single>"
	test $(PRECISION)

$(EXEC): check-benchtype $(RUN_SOURCES)
	$(NVCC) -I$(NVIDIA_INC),$(LOGHELPER_INC) -L$(NVIDIA_LIB),$(LOGHELPER_LIB) -DLOGS -Dtest_precision_$(LOWERCASED_PRECISION) -O3 -lLogHelper -std=c++11 -lcublas $(NVCC_FLAGS) -Xcompiler "-fopenmp -O3" $(LOGS) $(SOURCES) -o $(EXEC)

clean:
	rm -rf $(EXEC_RADIX)* $(MATRIX_NAMES)

generate: $(EXEC)
	$(EXEC) -generate -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD)

test: 
	$(EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=1 -verbose -no-warmup

golden:  
	$(EXEC) -size=$(DEF_SIZE) -input_a=$(DEF_A_INPUT) -input_b=$(DEF_B_INPUT) -gold=$(DEF_GOLD) -iterations=1 -verbose -no-warmup > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/Preparing/c\' golden_stdout.txt
	sed -i '/Average/c\' golden_stdout.txt
	sed -i '/time/c\' golden_stdout.txt
	sed -i '/SIZE/c\' golden_stdout.txt
	sed -i '/^$$/d' golden_stdout.txt

