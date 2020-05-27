CXX=g++-7

TARGET = mergesort
RAD=/home/carol/radiation-benchmarks
LOGHELPER_INC=$(RAD)/src/include -I$(RAD)/src/cuda/common 
LOGHELPER_LIB=$(RAD)/src/include 
CUDA_BASE_DIR=/usr/local/cuda
NVCC=$(CUDA_BASE_DIR)/bin/nvcc

GENCODE=-gencode arch=compute_35,code=[sm_35,compute_35]
GENCODE+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

NVCC_FLAGS =  -DLOGS -I$(LOGHELPER_INC) -L$(LOGHELPER_LIB) 
NVCC_FLAGS += -lLogHelper -Xptxas -v -I$(CUDA_BASE_DIR)/include/
NVCC_FLAGS += -Xcompiler " -fopenmp" -Xcompiler "-O3" -ccbin $(CXX)
NVCC_FLAGS += -I$(CUDA_BASE_DIR)/samples/common/inc
LINK_FLAGS = -std=c++11 $(GENCODE) -lcudart -lcudadevrt

all: clean $(TARGET)

$(TARGET):
	$(NVCC) -o mergesort bitonic.cu main.cpp mergeSort.cu mergeSort_host.cpp mergeSort_validate.cpp  $(NVCC_FLAGS) -L$(CUDA_LIB_DIR) $(LINK_FLAGS)

#SIZE=134217728
SIZE=1048576 

INPUT=$(RAD)/data/mergesort/mergesort_input_134217728
GOLD=$(RAD)/data/mergesort/mergesort_gold_$(SIZE)

test:
	./$(TARGET) -size=$(SIZE)  -input=$(INPUT) -gold=$(GOLD) -iterations=1 -verbose

golden:
	./$(TARGET) -size=$(SIZE)  -input=$(INPUT) -gold=$(GOLD) -iterations=1 -verbose > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/LOGFILENAME/c\' golden_stdout.txt 
	sed -i '/Time/c\' golden_stdout.txt 
	sed -i '/time/c\' golden_stdout.txt 
	sed -i '/^$$/d' golden_stdout.txt
	sed -i '/Perf/c\' golden_stdout.txt
	sed -i '/Starting/c\' golden_stdout.txt

generate:
	./$(TARGET) -generate -size=$(SIZE) -input=$(INPUT) -gold=$(GOLD) -iterations=1

clean:
	rm -f *.o *~ $(TARGET)

