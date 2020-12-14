#ifndef REAL_H
#define REAL_H

#define FLOAT       32
#define DOUBLE      64
#define HALF        16

#define CAST(v)     real(v)
#define CAST_DEV(v) real_device(v)

#define IS_MIX_PRECISION_FLOAT_LAYER(r)		REAL != FLOAT && r == FLOAT
#define IS_MIX_PRECISION_HALF_LAYER(r)		REAL != HALF && r == HALF

// Indicates which data type must be supported for mixed precision
#if REAL == FLOAT
	#define MIX_PRECISION_SUPPORT				HALF
#elif REAL == HALF
	#define MIX_PRECISION_SUPPORT				FLOAT
#endif

// > Half for mixed precision

#ifdef GPU
	#include <cuda_fp16.h>
#endif

#define HALF_ROUND_STYLE 1  // 1: nearest, -1: truncate (fastest, default)
#include "half.hpp"
using half_float::half;
using namespace half_float::literal;
typedef half_float::half half_host;
#if GPU
	typedef __half half_device;
#endif

// > Real type definitions

#if REAL == DOUBLE
    typedef double real;
    #define REAL_MAX __DBL_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_DOUBLE
	#if GPU
		typedef double real_device;
	#endif
#elif REAL == HALF
	typedef half_float::half real;
    #define REAL_MAX CAST(65504)
    #define CUDNN_DATA_REAL CUDNN_DATA_HALF
	#if GPU
		typedef __half real_device;
	#endif
#else // REAL == FLOAT
    typedef float real;
    #define REAL_MAX __FLT_MAX__
    #define CUDNN_DATA_REAL CUDNN_DATA_FLOAT
	#if GPU
		typedef float real_device;
	#endif
#endif

// > Real type conversions

#ifdef GPU
	void float2real_array_gpu(float* src, real* dst, int n);
	void real2float_array_gpu(real* src, float* dst, int n);

	void half2real_array_gpu(half_host* src, real* dst, int n);
	void real2half_array_gpu(real* src, half_host* dst, int n);

	void generic_copy_array_gpu(half_host *src, half_host *dst, int n);
	void generic_copy_array_gpu(half_host *src, float *dst, int n);
	void generic_copy_array_gpu(float *src, half_host *dst, int n);
	void generic_copy_array_gpu(float *src, float *dst, int n);
#endif

void float2real_array(float* src, real* dst, int n);
void real2float_array(real* src, float* dst, int n);

void half2real_array(half_host* src, real* dst, int n);
void real2half_array(real* src, half_host* dst, int n);


float* cast_array_real2float(real *src, int n, float *dst_gpu);
real* cast_array_float2real(float *src, int n, real *dst_gpu);

// > Real type functions

const char *get_default_real_string();
const char *get_real_string(int real);

#ifdef __NVCC__

	/* real3
	* Based on vector_types.h
	*/
	typedef struct __device_builtin__ {
		real_device x;
		real_device y;
		real_device z;
	} real_device3;

	/* Math functions */

	__device__ __forceinline__ real_device exp_real(real_device x) {
	#if REAL == HALF
		return hexp(x);
	#elif REAL == FLOAT
		return expf(x);
	#elif REAL == DOUBLE
		return exp(x);
	#endif
	}

	__device__ __forceinline__ real_device floor_real(real_device x) {
	#if REAL == HALF
		return hfloor(x);
	#elif REAL == FLOAT
		return floorf(x);
	#elif REAL == DOUBLE
		return floor(x);
	#endif
	}

	__device__ __forceinline__ real_device pow_real(real_device x, real_device y) {
	#if REAL == HALF
		return CAST_DEV(powf(float(x), float(y)));
	#elif REAL == FLOAT
		return powf(x, y);
	#elif REAL == DOUBLE
		return pow(x, y);
	#endif
	}

	__device__ __forceinline__ real_device sqrt_real(real_device x) {
	#if REAL == HALF
		return hsqrt(x);
	#elif REAL == FLOAT
		return sqrtf(x);
	#elif REAL == DOUBLE
		return sqrt(x);
	#endif
	}

	__device__ __forceinline__ real_device min_real(real_device x, real_device y) {
	#if REAL == HALF
		return fminf(x, y);
	#elif REAL == FLOAT
		return fminf(x, y);
	#elif REAL == DOUBLE
		return fmin(x, y);
	#endif
	}

	__device__ __forceinline__ real_device max_real(real_device x, real_device y) {
	#if REAL == HALF
		return fmaxf(x, y);
	#elif REAL == FLOAT
		return fmaxf(x, y);
	#elif REAL == DOUBLE
		return fmax(x, y);
	#endif
	}

	__device__ __forceinline__ real_device fabs_real(real_device x) {
	#if REAL == HALF
		return fabsf(x);
	#elif REAL == FLOAT
		return fabsf(x);
	#elif REAL == DOUBLE
		return fabs(x);
	#endif
	}

	__device__ __forceinline__ real_device log_real(real_device x) {
	#if REAL == HALF
		return hlog(x);
	#elif REAL == FLOAT
		return logf(x);
	#elif REAL == DOUBLE
		return log(x);
	#endif
	}

	__device__ __forceinline__ real_device cos_real(real_device x) {
	#if REAL == HALF
		return hcos(x);
	#elif REAL == FLOAT
		return cosf(x);
	#elif REAL == DOUBLE
		return cos(x);
	#endif
	}

	__device__ __forceinline__ real_device sin_real(real_device x) {
	#if REAL == HALF
		return hsin(x);
	#elif REAL == FLOAT
		return sinf(x);
	#elif REAL == DOUBLE
		return sin(x);
	#endif
	}

	__device__ __forceinline__ half_device atomicAdd_half(half_device *x, half_device val) {
		#if __CUDA_ARCH__ > 700
			return atomicAdd(x, val);
		#endif

		half_device old = *x;
		*x += val;
		return old;
	}

#endif


#endif