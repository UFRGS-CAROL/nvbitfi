CXX=g++-7
COMMON= -std=c++11 -O3 
CXXFLAGS= -fPIC -fopenmp 

EXEC=cudaGaussian
LOGS=1
BUILDPROFILER=0
ITERATIONS=1
SIZE=512

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

RADDIR=/home/carol/radiation-benchmarks

NVCCFLAGS= -Xptxas -v --compiler-options "$(CXXFLAGS)" 

ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RADDIR)/src/include -I$(RADDIR)/src/cuda/common/include

OBJDIR=./obj/
OBJ=  gaussian.o  main.o Parameters.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(RADDIR)/src/include  -lLogHelper
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand
NVCCFLAGS+= -ccbin $(CXX)

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

DIR=$(RADDIR)/data/gaussian


all: clean mkdir $(EXEC)

$(EXEC): $(OBJS)  
	$(CXX) $(COMMON) $^ -o $@ $(CXXFLAGS) $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(COMMON) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR) $(DIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)


generate:
	./$(EXEC) --size $(SIZE) --input $(DIR)/input_$(SIZE).data --gold $(DIR)/gold_$(SIZE).data --verbose --generate

test:
	./$(EXEC) --size $(SIZE) --input $(DIR)/input_$(SIZE).data --gold $(DIR)/gold_$(SIZE).data --verbose --iterations 1


golden:
	./$(EXEC) --size $(SIZE) --input $(DIR)/input_$(SIZE).data --gold $(DIR)/gold_$(SIZE).data --verbose --iterations 1 > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/time/c\' golden_stdout.txt 
	sed -i '/LOGFILENAME/c\' golden_stdout.txt
