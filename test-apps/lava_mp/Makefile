CXX=g++
EXEC=cuda_lava
LOGS=1
BUILDPROFILER=0
OPERATIONNUM=194
PRECISION=float
DMR=none
RADDIR=/home/carol/radiation-benchmarks
DATA_DIR=$(RADDIR)/data/lava
SIZE=2
STREAMS=1
CHARGES_INPUT=$(DATA_DIR)/lava_$(PRECISION)_charges_$(SIZE)
DISTANCES_INPUT=$(DATA_DIR)/lava_$(PRECISION)_distances_$(SIZE)
GOLD=$(DATA_DIR)/lava_$(PRECISION)_gold_$(SIZE)
ITERATIONS=1
RELATIVEERROR=0

VPATH=./src
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc
NVCCFLAGS= -std=c++11 -Xptxas -v
CXXFLAGS= -std=c++11 -o3  -fPIC -pthread 

# Kepler, Volta
SMS= 35 53 70 86
$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

INCLUDE= -I./src -I$(CUDAPATH)/include -I$(RADDIR)/src/include -I$(RADDIR)/src/cuda/common

OBJDIR=./obj/
OBJ= setup_double.o  setup_float.o Parameters.o cuda_lava.o 

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(RELATIVEERROR), 1)
NVCCFLAGS+= -DBUILDRELATIVEERROR=1
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L$(RADDIR)/src/include  -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L$(RADDIR)/src/cuda/common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif

LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(INCLUDE)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) -ccbin $(CXX) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR)
	mkdir -p $(DATA_DIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	
generate:
	./$(EXEC) -boxes $(SIZE) -generate -streams $(STREAMS) \
				-iterations 1 -verbose \
				-input_distances $(DISTANCES_INPUT) \
				-input_charges $(CHARGES_INPUT)	\
				-output_gold $(GOLD) \
				-opnum $(OPERATIONNUM) -precision $(PRECISION) -redundancy $(DMR)

golden:
	./$(EXEC) -boxes $(SIZE) -streams $(STREAMS) \
				-iterations $(ITERATIONS) -verbose \
				-input_distances $(DISTANCES_INPUT) \
				-input_charges $(CHARGES_INPUT) \
				-output_gold $(GOLD) \
				-opnum $(OPERATIONNUM) -precision $(PRECISION) \
				-redundancy $(DMR) >golden_stdout.txt 2>golden_stderr.txt
				sed -i '/LOGFILENAME/c\' golden_stdout.txt 
				sed -i '/time/c\' golden_stdout.txt 
				sed -i '/TESTE/c\' golden_stdout.txt 
				sed -i '/========/c\' golden_stdout.txt

test:
	./$(EXEC) -boxes $(SIZE) -streams $(STREAMS) \
				-iterations $(ITERATIONS) -verbose \
				-input_distances $(DISTANCES_INPUT) \
				-input_charges $(CHARGES_INPUT) \
				-output_gold $(GOLD) \
				-opnum $(OPERATIONNUM) -precision $(PRECISION) -redundancy $(DMR)

