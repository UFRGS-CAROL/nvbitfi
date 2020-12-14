#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"
#include <assert.h>

#include "blas.h"
#include "cuda.h"
#include "utils.h"

// > Mixed precision kernels

template<typename T>
__global__ void scale_bias_kernel(T *output, T *biases, int n, int size) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    int filter = blockIdx.y;
    int batch = blockIdx.z;

    if(offset < size) output[(batch*n+filter)*size + offset] *= biases[filter];
}

template<typename T>
__global__ void add_bias_kernel(T *output, T *biases, int batch, int n, int size) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n*size*batch) return;
    int i = index % size;
    index /= size;
    int j = index % n;
    index /= n;
    int k = index;

    output[(k*n+j)*size + i] += biases[j];
}

template<typename T>
__global__ void normalize_kernel(int N, T *x, T *mean, T *variance, int batch, int filters, int spatial) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    x[index] = (x[index] - mean[f]) / (sqrt(variance[f] + T(.00001f)));
}
__global__ void normalize_kernel(int N, half_device *x, half_device *mean, half_device *variance, int batch, int filters, int spatial) {
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    x[index] = (x[index] - mean[f]) / (hsqrt(variance[f] + half_device(.00001f)));
}

template<typename T>
__global__ void scal_kernel(int N, T ALPHA, T *X, int INCX) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] *= ALPHA;
}

template<typename T>
__global__ void fill_kernel(int N, T ALPHA, T *X, int INCX) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

template<typename T>
__global__ void copy_kernel(int N,  T *X, int OFFX, int INCX, T *Y, int OFFY, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

template<typename T>
__global__ void axpy_kernel(int N, T ALPHA, T *X, int OFFX, int INCX,  T *Y, int OFFY, int INCY) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

template<typename T>
__global__ void  fast_mean_kernel(T *x, int batch, int filters, int spatial, T *mean) {
    const int threads = BLOCK;
    __shared__ T local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : T(0);
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

template<typename T>
__global__ void  fast_variance_kernel(T *x, T *mean, int batch, int filters, int spatial, T *variance) {
    const int threads = BLOCK;
    __shared__ T local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? pow((x[index] - mean[filter]), 2) : T(0);
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}
__global__ void  fast_variance_kernel(half_device *x, half_device *mean, int batch, int filters, int spatial, half_device *variance) {
    const int threads = BLOCK;
    __shared__ half_device local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? half_device(pow((x[index] - mean[filter]), (float)2)) : half_device(0);
        }
    }

    __syncthreads();

    if(id == 0){
        variance[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance[filter] += local[i];
        }
        variance[filter] /= (spatial * batch - 1);
    }
}

template<typename T1, typename T2>
__global__ void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, T1 *add, int w2, int h2, int c2, float s1, float s2, T2 *out) {
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    out[out_index] = T2(s1)*out[out_index] + T2(T1(s2)*add[add_index]);
}

// > Mixed precision kernels callers

template<typename T>
void scale_bias_gpu_template(T *output, T *biases, int batch, int n, int size) {
    dim3 dimGrid((size-1)/BLOCK + 1, n, batch);
    dim3 dimBlock(BLOCK, 1, 1);

    scale_bias_kernel<<<dimGrid, dimBlock>>>(output, biases, n, size);
    check_error(cudaPeekAtLastError());
}
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size) {
    scale_bias_gpu_template(output, biases, batch, n, size);
}
void scale_bias_gpu(double *output, double *biases, int batch, int n, int size) {
    scale_bias_gpu_template(output, biases, batch, n, size);
}
void scale_bias_gpu(half_host *output, half_host *biases, int batch, int n, int size) {
    scale_bias_gpu_template((half_device*)output, (half_device*)biases, batch, n, size);
}

template<typename T>
void add_bias_gpu_template(T *output, T *biases, int batch, int n, int size) {
    int num = n*size*batch;

    add_bias_kernel<<<cuda_gridsize(num), BLOCK>>>(output, biases, batch, n, size);
    check_error(cudaPeekAtLastError());
}
void add_bias_gpu(float *output, float *biases, int batch, int n, int size) {
    add_bias_gpu_template(output, biases, batch, n, size);
}
void add_bias_gpu(double *output, double *biases, int batch, int n, int size) {
    add_bias_gpu_template(output, biases, batch, n, size);
}
void add_bias_gpu(half_host *output, half_host *biases, int batch, int n, int size) {
    add_bias_gpu_template((half_device*)output, (half_device*)biases, batch, n, size);
}

template<typename T>
void normalize_gpu_template(T *x, T *mean, T *variance, int batch, int filters, int spatial) {
    size_t N = batch*filters*spatial;
    normalize_kernel<<<cuda_gridsize(N), BLOCK>>>(N, x, mean, variance, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}
void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial) {
    normalize_gpu_template(x, mean, variance, batch, filters, spatial);
}
void normalize_gpu(double *x, double *mean, double *variance, int batch, int filters, int spatial) {
    normalize_gpu_template(x, mean, variance, batch, filters, spatial);
}
void normalize_gpu(half_host *x, half_host *mean, half_host *variance, int batch, int filters, int spatial) {
    normalize_gpu_template((half_device*)x, (half_device*)mean, (half_device*)variance, batch, filters, spatial);
}

template<typename T>
void scal_gpu_template(int N, float ALPHA, T *X, int INCX) {
    scal_kernel<<<cuda_gridsize(N), BLOCK>>>(N, T(ALPHA), X, INCX);
    check_error(cudaPeekAtLastError());
}
void scal_gpu(int N, float ALPHA, float *X, int INCX) {
    scal_gpu_template(N, ALPHA, X, INCX);
}
void scal_gpu(int N, float ALPHA, double *X, int INCX) {
    scal_gpu_template(N, ALPHA, X, INCX);
}
void scal_gpu(int N, float ALPHA, half_host *X, int INCX) {
    scal_gpu_template(N, ALPHA, (half_device*)X, INCX);
}

template<typename T>
void fill_gpu_template(int N, float ALPHA, T *X, int INCX) {
    fill_kernel<<<cuda_gridsize(N), BLOCK>>>(N, T(ALPHA), X, INCX);
    check_error(cudaPeekAtLastError());
}
void fill_gpu(int N, float ALPHA, float *X, int INCX) {
    fill_gpu_template(N, ALPHA, X, INCX);
}
void fill_gpu(int N, float ALPHA, double *X, int INCX) {
    fill_gpu_template(N, ALPHA, X, INCX);
}
void fill_gpu(int N, float ALPHA, half_host *X, int INCX) {
    fill_gpu_template(N, ALPHA, (half_device*)X, INCX);
}

template<typename T>
void copy_gpu_offset_template(int N, T *X, int OFFX, int INCX, T *Y, int OFFY, int INCY) {
    copy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}
void copy_gpu_offset(int N, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY) {
    copy_gpu_offset_template(N, X, OFFX, INCX, Y, OFFY, INCY);
}
void copy_gpu_offset(int N, double *X, int OFFX, int INCX, double *Y, int OFFY, int INCY) {
    copy_gpu_offset_template(N, X, OFFX, INCX, Y, OFFY, INCY);
}
void copy_gpu_offset(int N, half_host *X, int OFFX, int INCX, half_host *Y, int OFFY, int INCY) {
    copy_gpu_offset_template(N, (half_device*)X, OFFX, INCX, (half_device*)Y, OFFY, INCY);
}

template<typename T>
void copy_gpu_template(int N, T *X, int INCX, T *Y, int INCY) {
    copy_gpu_offset(N, X, 0, INCX, Y, 0, INCY);
}
void copy_gpu(int N, float *X, int INCX, float *Y, int INCY) {
    copy_gpu_template(N, X, INCX, Y, INCY);
}
void copy_gpu(int N, double *X, int INCX, double *Y, int INCY) {
    copy_gpu_template(N, X, INCX, Y, INCY);
}
void copy_gpu(int N, half_host *X, int INCX, half_host *Y, int INCY) {
    copy_gpu_template(N, X, INCX, Y, INCY);
}

template<typename T>
void axpy_gpu_offset_template(int N, float ALPHA, T *X, int OFFX, int INCX, T *Y, int OFFY, int INCY) {
    axpy_kernel<<<cuda_gridsize(N), BLOCK>>>(N, T(ALPHA), X, OFFX, INCX, Y, OFFY, INCY);
    check_error(cudaPeekAtLastError());
}
void axpy_gpu_offset(int N, float ALPHA, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY) {
    axpy_gpu_offset_template(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
}
void axpy_gpu_offset(int N, float ALPHA, double *X, int OFFX, int INCX, double *Y, int OFFY, int INCY) {
    axpy_gpu_offset_template(N, ALPHA, X, OFFX, INCX, Y, OFFY, INCY);
}
void axpy_gpu_offset(int N, float ALPHA, half_host *X, int OFFX, int INCX, half_host *Y, int OFFY, int INCY) {
    axpy_gpu_offset_template(N, ALPHA, (half_device*)X, OFFX, INCX, (half_device*)Y, OFFY, INCY);
}

template<typename T>
void axpy_gpu_template(int N, float ALPHA, T *X, int INCX, T *Y, int INCY) {
    axpy_gpu_offset(N, ALPHA, X, 0, INCX, Y, 0, INCY);
}
void axpy_gpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY) {
    axpy_gpu_template(N, ALPHA, X, INCX, Y, INCY);
}
void axpy_gpu(int N, float ALPHA, double *X, int INCX, double *Y, int INCY) {
    axpy_gpu_template(N, ALPHA, X, INCX, Y, INCY);
}
void axpy_gpu(int N, float ALPHA, half_host *X, int INCX, half_host *Y, int INCY) {
    axpy_gpu_template(N, ALPHA, X, INCX, Y, INCY);
}

template<typename T>
void fast_mean_gpu_template(T *x, int batch, int filters, int spatial, T *mean) {
    fast_mean_kernel<<<filters, BLOCK>>>(x, batch, filters, spatial, mean);
    check_error(cudaPeekAtLastError());
}
void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean) {
    fast_mean_gpu_template(x, batch, filters, spatial, mean);
}
void fast_mean_gpu(double *x, int batch, int filters, int spatial, double *mean) {
    fast_mean_gpu_template(x, batch, filters, spatial, mean);
}
void fast_mean_gpu(half_host *x, int batch, int filters, int spatial, half_host *mean) {
    fast_mean_gpu_template((half_device*)x, batch, filters, spatial, (half_device*)mean);
}

template<typename T>
void fast_variance_gpu_template(T *x, T *mean, int batch, int filters, int spatial, T *variance) {
    fast_variance_kernel<<<filters, BLOCK>>>(x, mean, batch, filters, spatial, variance);
    check_error(cudaPeekAtLastError());
}
void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance) {
    fast_variance_gpu_template(x, mean, batch, filters, spatial, variance);
}
void fast_variance_gpu(double *x, double *mean, int batch, int filters, int spatial, double *variance) {
    fast_variance_gpu_template(x, mean, batch, filters, spatial, variance);
}
void fast_variance_gpu(half_host *x, half_host *mean, int batch, int filters, int spatial, half_host *variance) {
    fast_variance_gpu_template((half_device*)x, (half_device*)mean, batch, filters, spatial, (half_device*)variance);
}

template<typename T1, typename T2>
void shortcut_gpu_template(int batch, int w1, int h1, int c1, T1 *add, int w2, int h2, int c2, float s1, float s2, T2 *out) {
    int minw = (w1 < w2) ? w1 : w2;
    int minh = (h1 < h2) ? h1 : h2;
    int minc = (c1 < c2) ? c1 : c2;

    int stride = w1/w2;
    int sample = w2/w1;
    assert(stride == h1/h2);
    assert(sample == h2/h1);
    if(stride < 1) stride = 1;
    if(sample < 1) sample = 1;

    int size = batch * minw * minh * minc;
    shortcut_kernel<<<cuda_gridsize(size), BLOCK>>>(size, minw, minh, minc, stride, sample, batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
    check_error(cudaPeekAtLastError());
}
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out) {
    shortcut_gpu_template(batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
}
void shortcut_gpu(int batch, int w1, int h1, int c1, double *add, int w2, int h2, int c2, float s1, float s2, double *out) {
    shortcut_gpu_template(batch, w1, h1, c1, add, w2, h2, c2, s1, s2, out);
}
void shortcut_gpu(int batch, int w1, int h1, int c1, half_host *add, int w2, int h2, int c2, float s1, float s2, half_host *out) {
    shortcut_gpu_template(batch, w1, h1, c1, (half_device*)add, w2, h2, c2, s1, s2, (half_device*)out);
}
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, half_host *out) {
    shortcut_gpu_template(batch, w1, h1, c1, add, w2, h2, c2, s1, s2, (half_device*)out);
}
void shortcut_gpu(int batch, int w1, int h1, int c1, half_host *add, int w2, int h2, int c2, float s1, float s2, float *out) {
    shortcut_gpu_template(batch, w1, h1, c1, (half_device*)add, w2, h2, c2, s1, s2, out);
}

// > General functions

__global__ void backward_scale_kernel(real_device *x_norm, real_device *delta, int batch, int n, int size, real_device *scale_updates)
{
    __shared__ real_device part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    real_device sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index]*x_norm[index] : CAST_DEV(0);
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) scale_updates[filter] += part[i];
    }
}

void backward_scale_gpu(real *x_norm, real *delta, int batch, int n, int size, real *scale_updates)
{
    backward_scale_kernel<<<n, BLOCK>>>((real_device*)x_norm, (real_device*)delta, batch, n, size, (real_device*)scale_updates);
    check_error(cudaPeekAtLastError());
}

__global__ void backward_bias_conn_kernel(real_device *bias_updates, real_device *delta, int batch, int n)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= n) return;
    int b;
    real_device sum = 0;
    for(b = 0; b < batch; ++b){
        int i = b*n + index;
        sum += delta[i];
    }
    bias_updates[index] += sum;
}

__global__ void backward_bias_kernel(real_device *bias_updates, real_device *delta, int batch, int n, int size)
{
    __shared__ real_device part[BLOCK];
    int i,b;
    int filter = blockIdx.x;
    int p = threadIdx.x;
    real_device sum = 0;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < size; i += BLOCK){
            int index = p + i + size*(filter + n*b);
            sum += (p+i < size) ? delta[index] : CAST_DEV(0);
        }
    }
    part[p] = sum;
    __syncthreads();
    if (p == 0) {
        for(i = 0; i < BLOCK; ++i) bias_updates[filter] += part[i];
    }
}

void backward_bias_gpu(real *bias_updates, real *delta, int batch, int n, int size)
{
    if(size == 1){
        backward_bias_conn_kernel<<<cuda_gridsize(n), BLOCK>>>((real_device*)bias_updates, (real_device*)delta, batch, n);
    }else{
        backward_bias_kernel<<<n, BLOCK>>>((real_device*)bias_updates, (real_device*)delta, batch, n, size);
    }
    check_error(cudaPeekAtLastError());
}

__global__ void adam_kernel(int N, real_device *x, real_device *m, real_device *v, real_device B1, real_device B2, real_device rate, real_device eps, int t)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;

    real_device mhat = m[index] / (CAST_DEV(1.f) - pow_real(B1, t));
    real_device vhat = v[index] / (CAST_DEV(1.f) - pow_real(B2, t));
    
    x[index] = x[index] + rate * mhat / (sqrt_real(vhat) + eps);
}

void adam_gpu(int n, real *x, real *m, real *v, float B1, float B2, float rate, float eps, int t)
{
    adam_kernel<<<cuda_gridsize(n), BLOCK>>>(n, (real_device*)x, (real_device*)m, (real_device*)v, (real_device)B1, (real_device)B2, (real_device)rate, (real_device)eps, t);
    check_error(cudaPeekAtLastError());
}

void adam_update_gpu(real *w, real *d, real *m, real *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t)
{
    scal_gpu(n, B1, m, 1);
    scal_gpu(n, B2, v, 1);
    axpy_gpu(n, -decay*batch, w, 1, d, 1);

    axpy_gpu(n, 1-B1, d, 1, m, 1);
    mul_gpu(n, d, 1, d, 1);
    axpy_gpu(n, 1-B2, d, 1, v, 1);

    adam_gpu(n, w, m, v, B1, B2, rate, eps, t);
    fill_gpu(n, 0, d, 1);
}

__global__ void normalize_delta_kernel(int N, real_device *x, real_device *mean, real_device *variance, real_device *mean_delta, real_device *variance_delta, int batch, int filters, int spatial, real_device *delta)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int f = (index/spatial)%filters;
    
    delta[index] = delta[index] * CAST_DEV(1.f) / (sqrt_real(variance[f] + CAST_DEV(.00001f))) + variance_delta[f] * CAST_DEV(2.f) * (x[index] - mean[f]) / CAST_DEV(spatial * batch) + mean_delta[f]/CAST_DEV(spatial*batch);
}

void normalize_delta_gpu(real *x, real *mean, real *variance, real *mean_delta, real *variance_delta, int batch, int filters, int spatial, real *delta)
{
    size_t N = batch*filters*spatial;
    normalize_delta_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device*)x, (real_device*)mean, (real_device*)variance, (real_device*)mean_delta, (real_device*)variance_delta, batch, filters, spatial, (real_device*)delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  variance_delta_kernel(real_device *x, real_device *delta, real_device *mean, real_device *variance, int batch, int filters, int spatial, real_device *variance_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    variance_delta[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance_delta[i] += delta[index]*(x[index] - mean[i]);
        }
    }
    variance_delta[i] *= CAST_DEV(-.5f) * pow_real(variance[i] + CAST_DEV(.00001f), (-3.f/2.f));
}

__global__ void accumulate_kernel(real_device *x, int n, int groups, real_device *sum)
{
    int k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}

__global__ void fast_mean_delta_kernel(real_device *delta, real_device *variance, int batch, int filters, int spatial, real_device *mean_delta)
{
    const int threads = BLOCK;
    __shared__ real_device local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? delta[index] : CAST_DEV(0);
        }
    }

    __syncthreads();

    if(id == 0){
        mean_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean_delta[filter] += local[i];
        }
        mean_delta[filter] *= (CAST_DEV(-1.f) / sqrt_real(variance[filter] + CAST_DEV(.00001f)));
    }
}

__global__ void  fast_variance_delta_kernel(real_device *x, real_device *delta, real_device *mean, real_device *variance, int batch, int filters, int spatial, real_device *variance_delta)
{
    const int threads = BLOCK;
    __shared__ real_device local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;

            local[id] += (i+id < spatial) ? delta[index]*(x[index] - mean[filter]) : CAST_DEV(0);
        }
    }

    __syncthreads();

    if(id == 0){
        variance_delta[filter] = 0;
        for(i = 0; i < threads; ++i){
            variance_delta[filter] += local[i];
        }
        variance_delta[filter] *= CAST_DEV(-.5f) * pow_real(variance[filter] + CAST_DEV(.00001f), (-3.f/2.f));
    }
}


__global__ void mean_delta_kernel(real_device *delta, real_device *variance, int batch, int filters, int spatial, real_device *mean_delta)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean_delta[i] = 0;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j*filters*spatial + i*spatial + k;
            mean_delta[i] += delta[index];
        }
    }
    mean_delta[i] *= (CAST_DEV(-1.f) / sqrt_real(variance[i] + CAST_DEV(.00001f)));
}

void mean_delta_gpu(real *delta, real *variance, int batch, int filters, int spatial, real *mean_delta)
{
    mean_delta_kernel<<<cuda_gridsize(filters), BLOCK>>>((real_device*)delta, (real_device*)variance, batch, filters, spatial, (real_device*)mean_delta);
    check_error(cudaPeekAtLastError());
}

void fast_mean_delta_gpu(real *delta, real *variance, int batch, int filters, int spatial, real *mean_delta)
{
    fast_mean_delta_kernel<<<filters, BLOCK>>>((real_device*)delta, (real_device*)variance, batch, filters, spatial, (real_device*)mean_delta);
    check_error(cudaPeekAtLastError());
}

void fast_variance_delta_gpu(real *x, real *delta, real *mean, real *variance, int batch, int filters, int spatial, real *variance_delta)
{
    fast_variance_delta_kernel<<<filters, BLOCK>>>((real_device*)x, (real_device*)delta, (real_device*)mean, (real_device*)variance, batch, filters, spatial, (real_device*)variance_delta);
    check_error(cudaPeekAtLastError());
}

__global__ void  mean_kernel(real_device *x, int batch, int filters, int spatial, real_device *mean)
{
    real_device scale = 1.f/(batch * spatial);
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    int j,k;
    mean[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}

__global__ void variance_kernel(real_device *x, real_device *mean, int batch, int filters, int spatial, real_device *variance)
{
    real_device scale = 1.f/(batch * spatial - 1);
    int j,k;
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (i >= filters) return;
    variance[i] = 0;
    for(j = 0; j < batch; ++j){
        for(k = 0; k < spatial; ++k){
            int index = j*filters*spatial + i*spatial + k;
            variance[i] += pow_real((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}

__global__ void reorg_kernel(int N, real_device *x, int w, int h, int c, int batch, int stride, int forward, real_device *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;
    //printf("%d\n", offset);
    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

   // printf("%d %d %d\n", w2, h2, c2);
    //printf("%d %d\n", in_index, out_index);
    //if(out_index >= N || out_index < 0) printf("bad bad bad \n");

    if(forward) out[out_index] = x[in_index];
    else out[in_index] = x[out_index];
    //if(forward) out[1] = x[1];
    //else out[0] = x[0];
}

__global__ void axpy_kernel(int N, real_device ALPHA, real_device *X, int OFFX, int INCX,  real_device *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[OFFY+i*INCY] += ALPHA*X[OFFX+i*INCX];
}

__global__ void pow_kernel(int N, real_device ALPHA, real_device *X, int INCX, real_device *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] = pow_real(X[i*INCX], ALPHA);
}

__global__ void const_kernel(int N, real_device ALPHA, real_device *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void constrain_kernel(int N, real_device ALPHA, real_device *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = min_real(ALPHA, max_real(-ALPHA, X[i*INCX]));
}

__global__ void supp_kernel(int N, real_device ALPHA, real_device *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}

__global__ void add_kernel(int N, real_device ALPHA, real_device *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] += ALPHA;
}

__global__ void fill_kernel(int N, real_device ALPHA, real_device *X, int INCX)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) X[i*INCX] = ALPHA;
}

__global__ void copy_kernel(int N,  real_device *X, int OFFX, int INCX, real_device *Y, int OFFY, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}

__global__ void mul_kernel(int N, real_device *X, int INCX, real_device *Y, int INCY)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < N) Y[i*INCY] *= X[i*INCX];
}

__global__ void l2norm_kernel(int N, real_device *x, real_device *dx, int batch, int filters, int spatial)
{
    int index = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    real_device sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += pow_real(x[index], 2);
    }
    sum = sqrt_real(sum);
    if(sum == CAST_DEV(0)) sum = CAST_DEV(1);
    //printf("%f\n", sum);
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
        dx[index] = (CAST_DEV(1) - x[index]) / sum;
    }
}

void l2normalize_gpu(real *x, real *dx, int batch, int filters, int spatial)
{
    size_t N = batch*spatial;
    l2norm_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device*)x, (real_device*)dx, batch, filters, spatial);
    check_error(cudaPeekAtLastError());
}

__global__ void  fast_mean_kernel(real_device *x, int batch, int filters, int spatial, real_device *mean)
{
    const int threads = BLOCK;
    __shared__ real_device local[threads];

    int id = threadIdx.x;
    local[id] = 0;

    int filter = blockIdx.x;

    int i, j;
    for(j = 0; j < batch; ++j){
        for(i = 0; i < spatial; i += threads){
            int index = j*spatial*filters + filter*spatial + i + id;
            local[id] += (i+id < spatial) ? x[index] : CAST_DEV(0);
        }
    }

    __syncthreads();

    if(id == 0){
        mean[filter] = 0;
        for(i = 0; i < threads; ++i){
            mean[filter] += local[i];
        }
        mean[filter] /= spatial * batch;
    }
}

void mean_gpu(real *x, int batch, int filters, int spatial, real *mean)
{
    mean_kernel<<<cuda_gridsize(filters), BLOCK>>>((real_device*)x, batch, filters, spatial, (real_device*)mean);
    check_error(cudaPeekAtLastError());
}

void variance_gpu(real *x, real *mean, int batch, int filters, int spatial, real *variance)
{
    variance_kernel<<<cuda_gridsize(filters), BLOCK>>>((real_device*)x, (real_device*)mean, batch, filters, spatial, (real_device*)variance);
    check_error(cudaPeekAtLastError());
}

void pow_gpu(int N, float ALPHA, real *X, int INCX, real  *Y, int INCY)
{
    pow_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device)ALPHA, (real_device*)X, INCX, (real_device*)Y, INCY);
    check_error(cudaPeekAtLastError());
}

void mul_gpu(int N, real *X, int INCX, real *Y, int INCY)
{
    mul_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device*)X, INCX, (real_device*)Y, INCY);
    check_error(cudaPeekAtLastError());
}

__global__ void flatten_kernel(int N, real_device *x, int spatial, int layers, int batch, int forward, real_device *out)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward) out[i2] = x[i1];
    else out[i1] = x[i2];
}

void flatten_gpu(real *x, int spatial, int layers, int batch, int forward, real *out)
{
    int size = spatial*batch*layers;
    flatten_kernel<<<cuda_gridsize(size), BLOCK>>>(size, (real_device*)x, spatial, layers, batch, forward, (real_device*)out);
    check_error(cudaPeekAtLastError());
}

void reorg_gpu(real *x, int w, int h, int c, int batch, int stride, int forward, real *out)
{
    int size = w*h*c*batch;
    reorg_kernel<<<cuda_gridsize(size), BLOCK>>>(size, (real_device*)x, w, h, c, batch, stride, forward, (real_device*)out);
    check_error(cudaPeekAtLastError());
}

__global__ void mask_kernel(int n, real_device *x, real_device mask_num, real_device *mask, real_device val)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] = val;
}

void mask_gpu(int N, real * X, float mask_num, real *mask, float val)
{
    mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device*)X, (real_device)mask_num, (real_device*)mask, (real_device)val);
    check_error(cudaPeekAtLastError());
}

__global__ void scale_mask_kernel(int n,  real_device *x, real_device mask_num, real_device *mask, real_device scale)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}

void scale_mask_gpu(int N, real *X, float mask_num, real * mask, float scale)
{
    scale_mask_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device*)X, (real_device)mask_num, (real_device*)mask, (real_device)scale);
    check_error(cudaPeekAtLastError());
}

void const_gpu(int N, float ALPHA, real *X, int INCX)
{
    const_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device)ALPHA, (real_device*)X, INCX);
    check_error(cudaPeekAtLastError());
}

void constrain_gpu(int N, float ALPHA, real *X, int INCX)
{
    constrain_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device)ALPHA, (real_device*)X, INCX);
    check_error(cudaPeekAtLastError());
}


void add_gpu(int N, float ALPHA, real *X, int INCX)
{
    add_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device)ALPHA, (real_device*)X, INCX);
    check_error(cudaPeekAtLastError());
}

void supp_gpu(int N, float ALPHA, real *X, int INCX)
{
    supp_kernel<<<cuda_gridsize(N), BLOCK>>>(N, (real_device)ALPHA, (real_device*)X, INCX);
    check_error(cudaPeekAtLastError());
}

__global__ void smooth_l1_kernel(int n, real_device *pred, real_device *truth, real_device *delta, real_device *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        real_device diff = truth[i] - pred[i];
        real_device abs_val = fabs_real(diff);
        if(abs_val < CAST_DEV(1)) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = CAST_DEV(2)*abs_val - CAST_DEV(1);
            delta[i] = (diff > CAST_DEV(0)) ? 1 : -1;
        }
    }
}

void smooth_l1_gpu(int n, real *pred, real *truth, real *delta, real *error)
{
    smooth_l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, (real_device*)pred, (real_device*)truth, (real_device*)delta, (real_device*)error);
    check_error(cudaPeekAtLastError());
}

__global__ void softmax_x_ent_kernel(int n, real_device *pred, real_device *truth, real_device *delta, real_device *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        real_device t = truth[i];
        real_device p = pred[i];
        error[i] = (t) ? -log_real(p) : CAST_DEV(0);
        delta[i] = t-p;
    }
}

void softmax_x_ent_gpu(int n, real *pred, real *truth, real *delta, real *error)
{
    softmax_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, (real_device*)pred, (real_device*)truth, (real_device*)delta, (real_device*)error);
    check_error(cudaPeekAtLastError());
}

__global__ void logistic_x_ent_kernel(int n, real_device *pred, real_device *truth, real_device *delta, real_device *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        real_device t = truth[i];
        real_device p = pred[i];
        error[i] = -t*log_real(p + CAST_DEV(.0000001)) - (CAST_DEV(1)-t)*log_real(CAST_DEV(1) - p + CAST_DEV(.0000001));
        delta[i] = t-p;
    }
}

void logistic_x_ent_gpu(int n, real *pred, real *truth, real *delta, real *error)
{
    logistic_x_ent_kernel<<<cuda_gridsize(n), BLOCK>>>(n, (real_device*)pred, (real_device*)truth, (real_device*)delta, (real_device*)error);
    check_error(cudaPeekAtLastError());
}

__global__ void l2_kernel(int n, real_device *pred, real_device *truth, real_device *delta, real_device *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        real_device diff = truth[i] - pred[i];
        error[i] = diff * diff; //I know this is technically wrong, deal with it.
        delta[i] = diff;
    }
}

void l2_gpu(int n, real *pred, real *truth, real *delta, real *error)
{
    l2_kernel<<<cuda_gridsize(n), BLOCK>>>(n, (real_device*)pred, (real_device*)truth, (real_device*)delta, (real_device*)error);
    check_error(cudaPeekAtLastError());
}

__global__ void l1_kernel(int n, real_device *pred, real_device *truth, real_device *delta, real_device *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        real_device diff = truth[i] - pred[i];
        error[i] = fabs_real(diff);
        delta[i] = (diff > CAST_DEV(0)) ? 1 : -1;
    }
}

void l1_gpu(int n, real *pred, real *truth, real *delta, real *error)
{
    l1_kernel<<<cuda_gridsize(n), BLOCK>>>(n, (real_device*)pred, (real_device*)truth, (real_device*)delta, (real_device*)error);
    check_error(cudaPeekAtLastError());
}

__global__ void wgan_kernel(int n, real_device *pred, real_device *truth, real_device *delta, real_device *error)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        error[i] = truth[i] ? -pred[i] : pred[i];
        delta[i] = (truth[i] > CAST_DEV(0)) ? 1 : -1;
    }
}

void wgan_gpu(int n, real *pred, real *truth, real *delta, real *error)
{
    wgan_kernel<<<cuda_gridsize(n), BLOCK>>>(n, (real_device*)pred, (real_device*)truth, (real_device*)delta, (real_device*)error);
    check_error(cudaPeekAtLastError());
}




__global__ void weighted_sum_kernel(int n, real_device *a, real_device *b, real_device *s, real_device *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = s[i]*a[i] + (CAST_DEV(1)-s[i])*(b ? b[i] : CAST_DEV(0));
    }
}

__global__ void deinter_kernel(int NX, real_device *X, int NY, real_device *Y, int B, real_device *OUT)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            if(X) X[b*NX + j] += OUT[i];
        } else {
            if(Y) Y[b*NY + j - NX] += OUT[i];
        }
    }
}

void deinter_gpu(int NX, real *X, int NY, real *Y, int B, real *OUT)
{
    deinter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, (real_device*)X, NY, (real_device*)Y, B, (real_device*)OUT);
    check_error(cudaPeekAtLastError());
}

__global__ void inter_kernel(int NX, real_device *X, int NY, real_device *Y, int B, real_device *OUT)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            OUT[i] = X[b*NX + j];
        } else {
            OUT[i] = Y[b*NY + j - NX];
        }
    }
}

void inter_gpu(int NX, real *X, int NY, real *Y, int B, real *OUT)
{
    inter_kernel<<<cuda_gridsize((NX+NY)*B), BLOCK>>>(NX, (real_device*)X, NY, (real_device*)Y, B, (real_device*)OUT);
    check_error(cudaPeekAtLastError());
}

void weighted_sum_gpu(real *a, real *b, real *s, int num, real *c)
{
    weighted_sum_kernel<<<cuda_gridsize(num), BLOCK>>>(num, (real_device*)a, (real_device*)b, (real_device*)s, (real_device*)c);
    check_error(cudaPeekAtLastError());
}

__global__ void weighted_delta_kernel(int n, real_device *a, real_device *b, real_device *s, real_device *da, real_device *db, real_device *ds, real_device *dc)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        if(db) db[i] += dc[i] * (CAST_DEV(1)-s[i]);
        ds[i] += dc[i] * (a[i] - b[i]);
    }
}

void weighted_delta_gpu(real *a, real *b, real *s, real *da, real *db, real *ds, int num, real *dc)
{
    weighted_delta_kernel<<<cuda_gridsize(num), BLOCK>>>(num, (real_device*)a, (real_device*)b, (real_device*)s, (real_device*)da, (real_device*)db, (real_device*)ds, (real_device*)dc);
    check_error(cudaPeekAtLastError());
}

__global__ void mult_add_into_kernel(int n, real_device *a, real_device *b, real_device *c)
{
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] += a[i]*b[i];
    }
}

void mult_add_into_gpu(int num, real *a, real *b, real *c)
{
    mult_add_into_kernel<<<cuda_gridsize(num), BLOCK>>>(num, (real_device*)a, (real_device*)b, (real_device*)c);
    check_error(cudaPeekAtLastError());
}


__device__ void softmax_device(real_device *input, int n, real_device temp, int stride, real_device *output)
{
    int i;
    real_device sum = 0;
    real_device largest = -INFINITY;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (CAST_DEV(val) > largest) ? CAST_DEV(val) : largest;
    }
    for(i = 0; i < n; ++i){
        real_device e = exp_real(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


__global__ void softmax_tree_kernel(real_device *input, int spatial, int batch, int stride, real_device temp, real_device *output, int groups, int *group_size, int *group_offset)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + goff + boff + s, group_size[g], temp, spatial, output + goff + boff + s);
}

void softmax_tree(real *input, int spatial, int batch, int stride, float temp, real *output, tree hier)
{
    int *tree_groups_size = cuda_make_int_array(hier.group_size, hier.groups);
    int *tree_groups_offset = cuda_make_int_array(hier.group_offset, hier.groups);
    int num = spatial*batch*hier.groups;
    softmax_tree_kernel<<<cuda_gridsize(num), BLOCK>>>((real_device*)input, spatial, batch, stride, (real_device)temp, (real_device*)output, hier.groups, tree_groups_size, tree_groups_offset);
    check_error(cudaPeekAtLastError());
    cuda_free((real *)tree_groups_size);
    cuda_free((real *)tree_groups_offset);
}

__global__ void softmax_kernel(real_device *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, real_device temp, real_device *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset, n, temp, stride, output + b*batch_offset + g*group_offset);
}

void softmax_gpu(real *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, real *output)
{
    softmax_kernel<<<cuda_gridsize(batch*groups), BLOCK>>>((real_device*)input, n, batch, batch_offset, groups, group_offset, stride, (real_device)temp, (real_device*)output);
    check_error(cudaPeekAtLastError());
}

template<typename T>
__global__ void upsample_kernel(size_t N, T *x, int w, int h, int c, int batch, int stride, int forward, T scale, T *out) {
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

    if (forward) out[out_index] += scale * x[in_index];
    else atomicAdd(x+in_index, scale * out[out_index]);
}
__global__ void upsample_kernel(size_t N, half_device *x, int w, int h, int c, int batch, int stride, int forward, half_device scale, half_device *out) {
    size_t i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

    if (forward) out[out_index] += scale * x[in_index];
    else atomicAdd_half(x+in_index, scale * out[out_index]);
}

void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out) {
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, scale, out);
    check_error(cudaPeekAtLastError());
}
void upsample_gpu(double *in, int w, int h, int c, int batch, int stride, int forward, float scale, double *out) {
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, in, w, h, c, batch, stride, forward, (double)scale, out);
    check_error(cudaPeekAtLastError());
}
void upsample_gpu(half_host *in, int w, int h, int c, int batch, int stride, int forward, float scale, half_host *out) {
    size_t size = w*h*c*batch*stride*stride;
    upsample_kernel<<<cuda_gridsize(size), BLOCK>>>(size, (half_device*)in, w, h, c, batch, stride, forward, (half_device)scale, (half_device*)out);
    check_error(cudaPeekAtLastError());
}


// > Extra/Test functions

template<typename T1, typename T2>
__global__ void relative_error_kernel(T1 *arr1, T2 *arr2, int N, T1 *out) {
    int i = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(i >= N) return;

    if (abs(arr1[i]) == 0)
        out[i] = 0;
    else
        out[i] = abs(arr1[i] - T1(arr2[i]))/abs(arr1[i]);
}

void relative_error_gpu(float *arr1, half_host* arr2, int N, float *out) {
    relative_error_kernel<<<cuda_gridsize(N), BLOCK>>>(arr1, (half_device*)arr2, N, out);
    check_error(cudaPeekAtLastError());
}
void relative_error_gpu(float *arr1, float* arr2, int N, float *out) {
    relative_error_kernel<<<cuda_gridsize(N), BLOCK>>>(arr1, arr2, N, out);
    check_error(cudaPeekAtLastError());
}