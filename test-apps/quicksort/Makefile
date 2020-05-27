TARGET = quicksort
RAD=/home/carol/radiation-benchmarks
LOG_FLAGS= -I$(RAD)/src/cuda/common

#SIZE=134217728
SIZE=1048576 
INPUT=$(RAD)/data/quicksort/quicksort_input_134217728
GOLD=$(RAD)/data/quicksort/quicksort_gold_$(SIZE)

CUDA_BASE_DIR=/usr/local/cuda
NVCC=$(CUDA_BASE_DIR)/bin/nvcc

GENCODE=-gencode arch=compute_35,code=[sm_35,compute_35]
GENCODE+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

LOGHELPER_LIB= -DLOGS -I$(RAD)/src/include -L$(RAD)/src/include -I$(RAD)/src/cuda/common

CXX=g++-4.8
NVCC_FLAGS = -ccbin $(CXX) $(LOGHELPER_LIB) -lLogHelper -Xptxas -v -I$(CUDA_BASE_DIR)/include/ -Xcompiler " -fopenmp" -rdc=true
LINK_FLAGS = -L$(CUDA_BASE_DIR)/lib64 -std=c++11 $(GENCODE) -lcudart -lcudadevrt 

all: clean $(TARGET)

$(TARGET):
	$(NVCC) -o $(TARGET) cdpAdvancedQuicksort.cu cdpBitonicSort.cu  $(NVCC_FLAGS) $(LINK_FLAGS)


golden:
	./quicksort -size=$(SIZE) -input=$(INPUT) -gold=$(GOLD) -iterations=1 -verbose > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/LOGFILENAME/c\' golden_stdout.txt 
	sed -i '/Time/c\' golden_stdout.txt 
	sed -i '/time/c\' golden_stdout.txt 
	sed -i '/^$$/d' golden_stdout.txt
	sed -i '/Perf/c\' golden_stdout.txt
	sed -i '/Starting/c\' golden_stdout.txt
	sed -i '/Done in/c\' golden_stdout.txt

test:
	./quicksort -size=$(SIZE) -input=$(INPUT) -gold=$(GOLD) -iterations=1

generate:
	./quicksort -size=$(SIZE) -input=$(INPUT) -gold=$(GOLD) -generate -iterations=1

clean:
	rm -f *.o *~ $(TARGET)

