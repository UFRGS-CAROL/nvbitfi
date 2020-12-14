#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

// extern "C" {
#include "dropout_layer.h"
#include "cuda.h"
#include "utils.h"
// }

__global__ void yoloswag420blazeit360noscope(real_device *input, int size, real_device *rand, real_device prob, real_device scale)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id < size) input[id] = (rand[id] < prob) ? CAST_DEV(0) : input[id]*scale;
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    cuda_random(layer.rand_gpu, size);

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>((real_device*)net.input_gpu, size, (real_device*)layer.rand_gpu, (real_device)layer.probability, (real_device)layer.scale);
    check_error(cudaPeekAtLastError());
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta_gpu) return;
    int size = layer.inputs*layer.batch;

    yoloswag420blazeit360noscope<<<cuda_gridsize(size), BLOCK>>>((real_device*)net.delta_gpu, size, (real_device*)layer.rand_gpu, (real_device)layer.probability, (real_device)layer.scale);
    check_error(cudaPeekAtLastError());
}
