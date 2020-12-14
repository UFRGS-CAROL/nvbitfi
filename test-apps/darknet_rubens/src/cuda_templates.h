#ifndef CUDA_TEMPLATES
#define CUDA_TEMPLATES

#ifdef GPU

// > Dependencies declaration

#include "cuda.h"
void check_error(cudaError_t status);

// > Templates

template<typename T>
void cuda_push_array(T *x_gpu, T *x, size_t n) {
    size_t size = sizeof(T)*n;
    cudaError_t status = cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status);
}

template<typename T>
void cuda_free(T *x_gpu) {
    cudaError_t status = cudaFree(x_gpu);
    check_error(status);
}

template<typename T>
void cuda_pull_array(T *x_gpu, T *x, size_t n) {
    size_t size = sizeof(T)*n;
    cudaError_t status = cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status);
}

#endif

#endif