CC=g++
NVCC=/usr/local/cuda/bin/nvcc
PRECISION?=half
DEF_SIZE?=1024
LOGS?=1
USE_OMP?=1
SAFE_MALLOC?=0

RADDIR=/home/carol/radiation-benchmarks

UPPERCASED_PRECISION=$(shell echo $(PRECISION) | tr a-z A-Z)
LOWERCASED_PRECISION=$(shell echo $(PRECISION) | tr A-Z a-z)

ifeq ($(UPPERCASED_PRECISION),HALF)
#ARCH= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
else
ARCH= 	-gencode arch=compute_35,code=[sm_35,compute_35]	# Tesla K40
#ARCH+= 	-gencode arch=compute_53,code=[sm_53,compute_53]	# Tegra X1
#ARCH+= 	-gencode arch=compute_60,code=[sm_60,compute_60]	# Tesla P100
#ARCH+= 	-gencode arch=compute_61,code=[sm_61,compute_61]	# Titan X | Titan Xp
#ARCH+= 	-gencode arch=compute_62,code=[sm_62,compute_62]	# Tegra X2
ARCH+= 	-gencode arch=compute_70,code=[sm_70,compute_70]	# Titan V | Tesla V100
endif

NVCC_FLAGS= $(ARCH) -O3 -std=c++11
INCLUDE= -I/usr/local/cuda/include
LIBRARY= -L/usr/local/cuda/lib64/
CXXFLAGS= -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -lcudart  -std=c++11

TARGET=cuda_trip_hotspot_$(LOWERCASED_PRECISION)
RUN_SOURCES=./cuda_trip_hotspot.cu

LOGHELPER_INC=$(RADDIR)/src/include/
LOGHELPER_LIB=$(RADDIR)/src/include/

ifeq ($(LOGS), 1)
INCLUDE+= -I$(LOGHELPER_INC)
NVCC_FLAGS+= -DLOGS -lLogHelper 
LIBRARY+= -L$(LOGHELPER_LIB) 
CXXFLAGS+= -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -DLOGS -lLogHelper 
endif

ifeq ($(USE_OMP), 1)
NVCC_FLAGS+= -DUSE_OMP -Xcompiler " -fopenmp -O3"
endif

ifeq ($(DEBUG), 1)
NVCC_FLAGS+= -g -G
endif

ifeq ($(SAFE_MALLOC), 1)
NVCC_FLAGS+= -I../../include/safe_memory -DSAFE_MALLOC -lSafeMemory 
LIBRARY+= -L../../include/safe_memory
endif

NVCC_FLAGS+= -DPRECISION_$(UPPERCASED_PRECISION)

TEST_SIM_TIME=100
DATA_DIR=$(RADDIR)/data/hotspot
DEF_INPUT_POWER=$(DATA_DIR)/power_$(DEF_SIZE)
DEF_INPUT_TEMP=$(DATA_DIR)/temp_$(DEF_SIZE)
DEF_GOLD=$(DATA_DIR)/hotspot_$(LOWERCASED_PRECISION)_$(DEF_SIZE)_$(TEST_SIM_TIME)

all: $(TARGET)

$(TARGET): $(RUN_SOURCES)
	$(NVCC) $(INCLUDE) $(LIBRARY) $(NVCC_FLAGS) $(RUN_SOURCES) -o $(TARGET)

clean:
	rm -rf $(TARGET) $(DEF_GOLD)

generate: $(TARGET)
	./$(TARGET) -generate -size=$(DEF_SIZE) -input_power=$(DEF_INPUT_POWER) -input_temp=$(DEF_INPUT_TEMP) -gold=$(DEF_GOLD) -sim_time=$(TEST_SIM_TIME) -verbose -iterations=1

test: $(TARGET) 
	./$(TARGET) -size=$(DEF_SIZE) -input_power=$(DEF_INPUT_POWER) -input_temp=$(DEF_INPUT_TEMP) -gold=$(DEF_GOLD) -sim_time=$(TEST_SIM_TIME) -iterations=1 -verbose

golden: $(TARGET) 
	./$(TARGET) -size=$(DEF_SIZE) -input_power=$(DEF_INPUT_POWER) -input_temp=$(DEF_INPUT_TEMP) -gold=$(DEF_GOLD) -sim_time=$(TEST_SIM_TIME) -iterations=1 -verbose > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/readInput time/c\' golden_stdout.txt 
	sed -i '/Iteration/c\' golden_stdout.txt 
