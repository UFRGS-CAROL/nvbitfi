#include "real.h"
#include "darknet.h"

template<typename T1, typename T2>
void array_cast(T1* src, T2* dst, int n) {
    int i;
    for (i = 0; i < n; i++)
        dst[i] = src[i];
}

void float2real_array(float* src, real* dst, int n) {
    array_cast(src, dst, n);
}

void real2float_array(real* src, float* dst, int n) {
    array_cast(src, dst, n);
}

void half2real_array(half_host* src, real* dst, int n) {
    array_cast(src, dst, n);
}

void real2half_array(real* src, half_host* dst, int n) {
    array_cast(src, dst, n);
}

/*
 *  @param src - CPU array
 *  @param dst_gpu - if NULL    -> 'src' will be converted to float and the output array will be available in CPU,
 *                   otherwise  -> if GPU && dst != NULL, the output array will be will be kept in GPU.
 */
float* cast_array_real2float(real *src, int n, float *dst_gpu) {
    #if REAL == FLOAT
        #ifdef GPU
            if (dst_gpu) {
                // Push 'src' to 'dst_gpu'
                cuda_push_array(dst_gpu, src, n);
                return dst_gpu;
            } else {
                // Nothing to do with src
                return src;
            }
        #else
            // Nothing to do with src
            return src;
        #endif
    #else
        #ifdef GPU
            // Cast on GPU
            real *src_gpu = cuda_make_array(src, n);
            if (dst_gpu) {
                // Keep output in GPU
                real2float_array_gpu(src_gpu, dst_gpu, n);
                cuda_free(src_gpu);
                return dst_gpu;
            } else {
                // Bring output back from GPU
                dst_gpu = cuda_make_float_array(NULL, n);
                float *dst = (float*)calloc(n, sizeof(float));
                real2float_array_gpu(src_gpu, dst_gpu, n);
                cuda_pull_array(dst_gpu, dst, n);
                cuda_free(src_gpu);
                cuda_free(dst_gpu);
                return dst;
            }
        #else
            // Cast and output in CPU
            float *dst = (float*)calloc(n, sizeof(float));
            real2float_array(src, dst, n);
            return dst;
        #endif
    #endif
}


/*
 *  @param src - CPU array
 *  @param dst_gpu - if NULL    -> 'src' will be converted to 'real' and the output array will be available in CPU
 *                   otherwise  -> if GPU && dst != NULL, the output array will be will be kept in GPU
 */
real* cast_array_float2real(float *src, int n, real *dst_gpu) {
    #if REAL == FLOAT
        #ifdef GPU
            if (dst_gpu) {
                // Push 'src' to 'dst_gpu'
                cuda_push_array(dst_gpu, src, n);
                return dst_gpu;
            } else {
                // Nothing to do with src
                return src;
            }
        #else
            // Nothing to do with src
            return src;
        #endif
    #else
        #ifdef GPU
            // Cast on GPU
            float *src_gpu = cuda_make_float_array(src, n);
            if (dst_gpu) {
                // Keep output in GPU
                float2real_array_gpu(src_gpu, dst_gpu, n);
                cuda_free(src_gpu);
                return dst_gpu;
            } else {
                // Bring output back from GPU
                dst_gpu = cuda_make_array(NULL, n);
                real *dst = (real*)calloc(n, sizeof(real));
                float2real_array_gpu(src_gpu, dst_gpu, n);
                cuda_pull_array(dst_gpu, dst, n);
                cuda_free(src_gpu);
                cuda_free(dst_gpu);
                return dst;
            }
        #else
            // Cast and output in CPU
            real *dst = (real*)calloc(n, sizeof(real));
            float2real_array(src, dst, n);
            return dst;
        #endif
    #endif
}


const char *get_default_real_string() {
    switch (REAL) {
        case HALF: return "HALF";
        case FLOAT: return "FLOAT";
        case DOUBLE: return "DOUBLE";
    }
    return "";
}

const char *get_real_string(int real) {
    switch (real) {
        case HALF: return "HALF";
        case FLOAT: return "FLOAT";
        case DOUBLE: return "DOUBLE";
    }
    return "";
}