CXX=g++-7
CXXFLAGS= -std=c++11 -o3 -fopenmp 
TARGET=nw
LOGS=1
ITERATIONS=1

GPU?=1
DEBUG?=0
#for radiation setup
OMP?=1

SIZE?=16384
#6384 #8192 #4096 
ITERATIONS=1

RADDIR=/home/carol/radiation-benchmarks
DATA=$(RADDIR)/data/nw

INPUT=$(DATA)/input_$(SIZE)
GOLD=$(DATA)/gold_$(SIZE)
PENALTY=10

VPATH=./src/

CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc 

NVCCFLAGS= -std=c++11 -O3 -Xptxas -v  -Wno-deprecated-gpu-targets


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RADDIR)/src/include -I$(RADDIR)/src/cuda/common

OBJDIR=./obj/
OBJ= needle.o needle_kernel.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(RADDIR)/src/include  -lLogHelper
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt  -Wno-deprecated-gpu-targets

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


all: mkdir $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(COMMON) $(CFLAGS) $^ $(INCLUDE) -o $@  $(LDFLAGS)  

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@ $(INCLUDE) 

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) -ccbin $(CXX) $(ARCH) $(COMMON) $(NVCCFLAGS) --compiler-options "$(CFLAGS)" $(INCLUDE) -c $< -o $@ 


mkdir:
	mkdir -p $(OBJDIR) $(DATA)

clean:
	rm -f $(OBJDIR)* $(TARGET)

generate:
	./$(TARGET) $(SIZE) $(PENALTY) $(INPUT) $(GOLD) $(ITERATIONS) 1

test:
	./$(TARGET) $(SIZE) $(PENALTY) $(INPUT) $(GOLD) $(ITERATIONS) 0

golden:
	./$(TARGET) $(SIZE) $(PENALTY) $(INPUT) $(GOLD) $(ITERATIONS) 0 > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/LOGFILE/c\REPLACED.' golden_stdout.txt 
	sed -i '/read../c\REPLACED.' golden_stdout.txt 
	sed -i '/^$$/d' golden_stdout.txt
