CXX=g++-7
COMMON= -std=c++11 -O3 
CXXFLAGS= -fPIC -fopenmp 
EXEC=cudaLUD
LOGS=1
BUILDPROFILER=0
ITERATIONS=1
#SIZE=8192
SIZE=2048

VPATH=./src/
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc

RADDIR=/home/carol/radiation-benchmarks

NVCCFLAGS= -Xptxas -v --compiler-options "$(CXXFLAGS)" 


ARCH= -gencode arch=compute_35,code=[sm_35,compute_35] #Kepler
ARCH+= -gencode arch=compute_70,code=[sm_70,compute_70] #Titan V

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RADDIR)/src/include -I$(RADDIR)/src/cuda/common/include

OBJDIR=./obj/
OBJ=  lud_kernel.o  main.o Parameters.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(RADDIR)/src/include  -lLogHelper
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand $(EXTRA_LINK_FLAGS)
NVCCFLAGS+= -ccbin $(CXX) $(EXTRA_NVCC_FLAGS) 

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

DATA=$(RADDIR)/data/lud

all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(CXX) $(COMMON) $^ -o $@ $(CXXFLAGS) $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(COMMON) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR) $(DATA)
	
clean:
	rm -f $(OBJDIR)* $(EXEC)
	

generate:
	./$(EXEC) --size $(SIZE) --generate --input $(DATA)/input_$(SIZE).data --gold $(DATA)/gold_$(SIZE).data --iterations $(ITERATIONS) --verbose;

test:
	./$(EXEC) --size $(SIZE) --input $(DATA)/input_$(SIZE).data --gold $(DATA)/gold_$(SIZE).data --iterations $(ITERATIONS) --verbose;
	
golden:
	./$(EXEC) --size $(SIZE) --input $(DATA)/input_$(SIZE).data --gold $(DATA)/gold_$(SIZE).data --iterations $(ITERATIONS) --verbose > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/Array set time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Gold check time/c\REPLACED.' golden_stdout.txt 
	sed -i '/OUTPUT/c\REPLACED.' golden_stdout.txt 
	sed -i '/Iteration 0 overall time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Total kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Average kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Device kernel time/c\REPLACED.' golden_stdout.txt 
	sed -i '/Copy time/c\REPLACED.' golden_stdout.txt 

