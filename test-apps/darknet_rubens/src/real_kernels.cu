#include "real.h"

#include <cuda_fp16.h>
#include <cuda.h>

template<typename T1, typename T2>
__global__ void array_cast_kernel(T1 *src, T2 *dst, int n) {
	int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
	if (i < n) {
		dst[i] = src[i];
	}
}

void half2real_array_gpu(half_host* src, real* dst, int n) {
	array_cast_kernel<<<cuda_gridsize(n), BLOCK>>>((half_device*)src, (real_device*)dst, n);
	check_error(cudaPeekAtLastError());
}
void real2half_array_gpu(real* src, half_host* dst, int n) {
	array_cast_kernel<<<cuda_gridsize(n), BLOCK>>>((real_device*)src, (half_device*)dst, n);
	check_error(cudaPeekAtLastError());
}

void float2real_array_gpu(float* src, real* dst, int n) {
	array_cast_kernel<<<cuda_gridsize(n), BLOCK>>>(src, (real_device*)dst, n);
	check_error(cudaPeekAtLastError());
}

void real2float_array_gpu(real* src, float* dst, int n) {
	array_cast_kernel<<<cuda_gridsize(n), BLOCK>>>((real_device*)src, dst, n);
	check_error(cudaPeekAtLastError());
}


template<typename T1, typename T2>
void generic_copy_array_gpu_template(T1 *src, T2 *dst, int n) {
	array_cast_kernel<<<cuda_gridsize(n), BLOCK>>>(src, dst, n);
	check_error(cudaPeekAtLastError());
}
void generic_copy_array_gpu(half_host *src, half_host *dst, int n) {
	generic_copy_array_gpu_template((half_device*)src, (half_device*)dst, n);
}
void generic_copy_array_gpu(half_host *src, float *dst, int n) {
	generic_copy_array_gpu_template((half_device*)src, dst, n);
}
void generic_copy_array_gpu(float *src, half_host *dst, int n) {
	generic_copy_array_gpu_template(src, (half_device*)dst, n);
}
void generic_copy_array_gpu(float *src, float *dst, int n) {
	generic_copy_array_gpu_template(src, dst, n);
}



