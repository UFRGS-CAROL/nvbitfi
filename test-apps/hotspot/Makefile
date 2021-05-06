CXX=g++-7
CUDA_BASE_DIR=/usr/local/cuda
NVCC=$(CUDA_BASE_DIR)/bin/nvcc


PRECISION=-DPRECISION_SINGLE
TARGET = hotspot
RAD=/home/carol/radiation-benchmarks

LOGHELPER_INC=$(RAD)/src/include 
LOGHELPER_LIB=$(RAD)/src/include 

GENCODE=-gencode arch=compute_35,code=[sm_35,compute_35]
GENCODE+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

LOG_FLAGS=-DLOGS -I$(RAD)/src/cuda/common  -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) -lLogHelper

NVCC_FLAGS = $(PRECISION) $(LOG_FLAGS) -ccbin $(CXX) -Xptxas -v -I$(CUDA_BASE_DIR)/include/ -Xcompiler " -fopenmp"
LINK_FLAGS = -std=c++11 $(GENCODE) -lcudart -lcudadevrt $(EXTRA_LINK_FLAGS)

NVCC_FLAGS+= -Xptxas -v $(NVCCOPTFLAGS) $(OPT)
# TODO: verify this code
NVCC_FLAGS+= -DNVCCOPTFLAGS="$(strip $(patsubst %-Xptxas,%,$(NVCCOPTFLAGS)))"  
# Host must be always at maximum opt
NVCC_FLAGS+= --optimize 3

CUDA_LIB_DIR=$(CUDA_BASE_DIR)/lib64

INPUTTEMP=$(RAD)/data/hotspot/temp_1024
INPUTPOWER=$(RAD)/data/hotspot/power_1024
GOLD=$(RAD)/data/hotspot/gold_1024_1

all: $(TARGET)

$(TARGET):
	$(NVCC) -o hotspot cuda_hotspot.cu $(CFLAGS) $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)
	
golden:
	./$(TARGET) -verbose -size=1024 -sim_time=10 -streams=1 -input_temp=$(INPUTTEMP) -input_power=$(INPUTPOWER) -gold_temp=$(GOLD) -iterations=1 > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/LOGFILENAME/c\' golden_stdout.txt 
	sed -i '/Time/c\' golden_stdout.txt 
	sed -i '/time/c\' golden_stdout.txt 
	sed -i '/^$$/d' golden_stdout.txt
	sed -i '/Performance/c\' golden_stdout.txt

test:
	./$(TARGET) -size=1024 -sim_time=10 -streams=1 -input_temp=$(INPUTTEMP) -input_power=$(INPUTPOWER) -gold_temp=$(GOLD) -iterations=1

generate:
	./$(TARGET) -size=1024 -sim_time=10 -streams=1 -input_temp=$(INPUTTEMP) -input_power=$(INPUTPOWER) -gold_temp=$(GOLD) -iterations=1 -generate

clean:
	rm -f *.o *~ hotspot

