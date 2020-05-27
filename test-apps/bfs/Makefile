CXX=g++-7
CXXFLAGS= -std=c++11 -o3 -fopenmp 
EXEC=cudaBFS
LOGS=1
BUILDPROFILER=0
ITERATIONS=1

RAD=/home/carol/radiation-benchmarks
GOLD=$(RAD)/data/bfs/gold.data
INPUT=$(RAD)/data/bfs/graph1MW_6.txt

VPATH=./src/

CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc 

NVCCFLAGS= -std=c++11 -O3 -Xptxas -v  -Wno-deprecated-gpu-targets

ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RAD)/src/include -I$(RAD)/src/cuda/common/include

OBJDIR=./obj/
OBJ= bfs.o main.o Parameters.o kernel.o


ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(RAD)/src/include  -lLogHelper
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand -lcudadevrt  -Wno-deprecated-gpu-targets

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile


all: clean mkdir $(EXEC)

$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE) $(EXTRA_LINK_FLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE) 

	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC)   -ccbin $(CXX) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE)

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)

generate:
	./$(EXEC) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose --generate


test:
	./$(EXEC) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose

golden:
	./$(EXEC) --input $(INPUT) --gold $(GOLD) --iterations $(ITERATIONS) --verbose > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/LOGFILENAME/c\' golden_stdout.txt 
	sed -i '/Time/c\' golden_stdout.txt 
	sed -i '/time/c\' golden_stdout.txt 
	sed -i '/^$$/d' golden_stdout.txt
