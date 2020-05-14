GPU=1
CUDNN=0
OPENCV=0
DEBUG=0

#for radiation setup
LOGS=1
GEN_IMG?=0 #if you want to generate images on gold generator execution
TARGET=darknet_v2

ARCH= -gencode arch=compute_35,code=sm_35
#      -gencode arch=compute_50,code=[sm_50,compute_50] \
#      -gencode arch=compute_52,code=[sm_52,compute_52] \
#      -gencode arch=compute_60,code=[sm_60,compute_60] \
#      -gencode arch=compute_62,code=[sm_62,compute_62]

# This is what I use, uncomment if you know your arch and want to specify
# ARCH=  -gencode arch=compute_52,code=compute_52

VPATH=./src/
OBJDIR=./obj/

RAD_DIR=/home/carol/radiation-benchmarks
CUDA_BASE_DIR=/usr/local/cuda
NVCC=$(CUDA_BASE_DIR)/bin/nvcc

CC=gcc-7
CXX=g++-7
OPTS=-O3
LDFLAGS= -lm -pthread 
COMMON= 
CFLAGS= 

ifeq ($(GEN_IMG), 1)
CFLAGS+= -DGEN_IMG=1
endif

ifeq ($(DEBUG), 1) 
OPTS=-O0 -g
endif

CFLAGS+=$(OPTS)

ifeq ($(OPENCV), 1) 
COMMON+= -DOPENCV
CFLAGS+= -DOPENCV
LDFLAGS+= `pkg-config --libs opencv` 
COMMON+= `pkg-config --cflags opencv` 
endif

ifeq ($(GPU), 1) 
COMMON+= -DGPU -I$(CUDA_BASE_DIR)/include/
CFLAGS+= -DGPU
LDFLAGS+= -L$(CUDA_BASE_DIR)/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(CUDNN), 1) 
COMMON+= -DCUDNN 
CFLAGS+= -DCUDNN
LDFLAGS+= -lcudnn
endif

OBJ=gemm.o utils.o cuda.o deconvolutional_layer.o convolutional_layer.o \
list.o image.o activations.o im2col.o col2im.o blas.o crop_layer.o \
dropout_layer.o maxpool_layer.o softmax_layer.o data.o matrix.o network.o \
connected_layer.o cost_layer.o parser.o option_list.o darknet.o detection_layer.o \
captcha.o route_layer.o writing.o box.o nightmare.o normalization_layer.o avgpool_layer.o \
coco.o dice.o yolo.o detector.o layer.o compare.o regressor.o classifier.o local_layer.o \
swag.o shortcut_layer.o activation_layer.o rnn_layer.o gru_layer.o rnn.o rnn_vid.o \
crnn_layer.o demo.o tag.o cifar.o go.o batchnorm_layer.o art.o region_layer.o reorg_layer.o \
lsd.o super.o voxel.o tree.o args.o log_processing.o abft_host.o
ifeq ($(GPU), 1) 
LDFLAGS+= -lstdc++ 
OBJ+=convolutional_kernels.o deconvolutional_kernels.o activation_kernels.o \
im2col_kernels.o col2im_kernels.o blas_kernels.o crop_layer_kernels.o dropout_layer_kernels.o \
maxpool_layer_kernels.o network_kernels.o avgpool_layer_kernels.o maxpool_layer_hardened_kernels.o
endif

ifeq ($(LOGS), 1)
LOGHELPERINC=-I$(RAD_DIR)/src/include/
LOGHELPERLIB= -L$(RAD_DIR)/src/include/ -lLogHelper -DLOGS=1
HELPFUL=-I$(RAD_DIR)/src/include/
endif

OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) Makefile

############################################################################################
#-------------------------------------------------------------------------------------------
#-rdc=true
NVCC_FLAGS += -ccbin $(CXX) -Xptxas -v -I$(CUDA_BASE_DIR)/include/ $(ARCH)

LINK_FLAGS =  -L$(CUDA_LIB_DIR) -lcudart -lcudadevrt -lcuda -lcudart -lcublas -lcurand

ALL_FLAGS=$(NVCC_FLAGS) $(LINK_FLAGS)

#-------------------------------------------------------------------------------------------
############################################################################################

all: clean obj backup results $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $^ $(NVCC_FLAGS) $(LOGHELPERLIB) $(LOGHELPERINC) $(LINK_FLAGS) -o $@  -lstdc++ 

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@  $(HELPFUL) $(LOGHELPERLIB) $(LOGHELPERINC) $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CXX) $(COMMON) $(CFLAGS) -c $< -o $@  $(HELPFUL) $(LOGHELPERLIB) $(LOGHELPERINC) $(LDFLAGS) -std=c++11

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH)  $(CFLAGS) -c $< -o $@ $(ALL_FLAGS)


obj:
	mkdir -p obj
backup:
	mkdir -p backup
results:
	mkdir -p results

.PHONY: clean

clean:
	rm -rf $(OBJS) $(TARGET)

test_pure_yolo:
	./$(TARGET) detect cfg/yolo.cfg $(RAD_DIR)/data/darknet/yolo_v2.weights data/dog.jpg
	
test:
	./$(TARGET) test_radiation -d $(RAD_DIR)/data/darknet/fault_injection.csv -n 1 -s 0 -a 0 
	
golden:
	./$(TARGET) test_radiation -d $(RAD_DIR)/data/darknet/fault_injection.csv -n 1 -s 0 -a 0 > golden_stdout.txt 2>golden_stderr.txt
	sed -i '/Iteration 0 - image/c\REPLACED.' golden_stdout.txt 


generate:
	./$(TARGET) test_radiation  -c $(RAD_DIR)/src/cuda/darknet_v2/cfg/yolo.cfg \
								  -w $(RAD_DIR)/data/darknet/yolo_v2.weights \
								  -g 1 -d $(RAD_DIR)/data/darknet/fault_injection.csv -s 0 -l   \
								  $(RAD_DIR)/data/networks_img_list/fault_injection.txt -a 0
