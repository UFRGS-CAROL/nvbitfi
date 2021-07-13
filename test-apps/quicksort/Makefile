CXX=g++
EXEC=quicksort
LOGS=1
BUILDPROFILER=0
DEBUG=0
#SIZE=134217728
SIZE=1048576

VPATH=./
CUDAPATH=/usr/local/cuda
NVCC=$(CUDAPATH)/bin/nvcc
RADDIR=/home/fernando/radiation-benchmarks
INPUT=$(RADDIR)/data/quicksort/quicksort_input_134217728
GOLD=$(RADDIR)/data/quicksort/quicksort_gold_$(SIZE)

NVCCFLAGS= -std=c++14 -O3 -Xptxas -v -rdc=true -Xcompiler -fopenmp
CXXFLAGS= -std=c++14 -O3  -Xcompiler -fPIC  -Xcompiler  -pthread -Xcompiler -fopenmp

INCLUDE= -I$(CUDAPATH)/include -I$(CUDAPATH)/samples/common/inc -I$(RADDIR)/src/include -I$(RADDIR)/src/cuda/common
# Kepler, Volta
SMS= 35 70 86
$(foreach sm,$(SMS),$(eval ARCH += -gencode arch=compute_$(sm),code=sm_$(sm)))

OBJDIR=./obj/
OBJ= cdpBitonicSort.o cdpAdvancedQuicksort.o

ifeq ($(DEBUG), 1) 
CXXFLAGS+=-O0 -g
NVCCFLAGS+= -g -G
endif

ifeq ($(LOGS), 1)
CXXFLAGS+= -DLOGS
NVCCFLAGS+= -DLOGS
LDFLAGS+= -L../../include  -lLogHelper
endif

ifeq ($(BUILDPROFILER), 1)
CXXFLAGS+= -DBUILDPROFILER
NVCCFLAGS+= -DBUILDPROFILER
LDFLAGS+= -L../common/lib -lNVMLWrapper -L$(CUDAPATH)/lib64/stubs -lnvidia-ml 
endif


LDFLAGS+= -L$(CUDAPATH)/lib64  -lcudart  -lcurand

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

# It is necessary to multi compiler analysis
PASSHOSTCC= -DNVCCOPTFLAGS="$(strip $(NVCCOPTFLAGS))"
NVCCFLAGS+= -Xptxas -v $(NVCCOPTFLAGS) --optimize 3 $(PASSHOSTCC)

all: mkdir $(EXEC)


$(EXEC): $(OBJS)  
	$(NVCC) $(ARCH) $^ -o $@ $(LDFLAGS) $(INCLUDE) $(CXXFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INCLUDE)
	
$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(NVCCFLAGS) -c $< -o $@ $(INCLUDE) 

mkdir:
	mkdir -p $(OBJDIR)

clean:
	rm -f $(OBJDIR)* $(EXEC)
	
	
generate:
	 ./$(EXEC) -size=$(SIZE) -generate -verbose \
	 		   -input=$(INPUT)  -gold=$(GOLD) -iterations=1
	 		   #-noinputensurance
	 		   	
test:
	 ./$(EXEC) -size=$(SIZE) -verbose \
	 		   -input=$(INPUT)  -gold=$(GOLD) -iterations=1
	 		   #-noinputensurance

golden:
        ./quicksort -size=$(SIZE) -input=$(INPUT) -gold=$(GOLD) -iterations=1 -verbose > golden_stdout.txt 2>golden_stderr.txt
        sed -i '/LOGFILENAME/c\' golden_stdout.txt 
        sed -i '/Time/c\' golden_stdout.txt 
        sed -i '/time/c\' golden_stdout.txt 
        sed -i '/^$$/d' golden_stdout.txt
        sed -i '/Perf/c\' golden_stdout.txt
        sed -i '/Starting/c\' golden_stdout.txt
        sed -i '/Done in/c\' golden_stdout.txt

