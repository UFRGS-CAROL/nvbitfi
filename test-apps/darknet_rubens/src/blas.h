#ifndef BLAS_H
#define BLAS_H
#include "darknet.h"

void flatten(real *x, int size, int layers, int batch, int forward);
void pm(int M, int N, real *A);
real *random_matrix(int rows, int cols);
void time_random_matrix(int TA, int TB, int m, int k, int n);
void reorg_cpu(real *x, int w, int h, int c, int batch, int stride, int forward, real *out);

void test_blas();

void inter_cpu(int NX, real *X, int NY, real *Y, int B, real *OUT);
void deinter_cpu(int NX, real *X, int NY, real *Y, int B, real *OUT);
void mult_add_into_cpu(int N, real *X, real *Y, real *Z);

void const_cpu(int N, float ALPHA, real *X, int INCX);
void constrain_gpu(int N, float ALPHA, real * X, int INCX);
void pow_cpu(int N, float ALPHA, real *X, int INCX, real *Y, int INCY);
void mul_cpu(int N, real *X, int INCX, real *Y, int INCY);

int test_gpu_blas();
void shortcut_cpu(int batch, int w1, int h1, int c1, real *add, int w2, int h2, int c2, float s1, float s2, real *out);

void mean_cpu(real *x, int batch, int filters, int spatial, real *mean);
void variance_cpu(real *x, real *mean, int batch, int filters, int spatial, real *variance);

void scale_bias(real *output, real *scales, int batch, int n, int size);
void backward_scale_cpu(real *x_norm, real *delta, int batch, int n, int size, real *scale_updates);
void mean_delta_cpu(real *delta, real *variance, int batch, int filters, int spatial, real *mean_delta);
void  variance_delta_cpu(real *x, real *delta, real *mean, real *variance, int batch, int filters, int spatial, real *variance_delta);
void normalize_delta_cpu(real *x, real *mean, real *variance, real *mean_delta, real *variance_delta, int batch, int filters, int spatial, real *delta);
void l2normalize_cpu(real *x, real *dx, int batch, int filters, int spatial);

void smooth_l1_cpu(int n, real *pred, real *truth, real *delta, real *error);
void l2_cpu(int n, real *pred, real *truth, real *delta, real *error);
void l1_cpu(int n, real *pred, real *truth, real *delta, real *error);
void logistic_x_ent_cpu(int n, real *pred, real *truth, real *delta, real *error);
void softmax_x_ent_cpu(int n, real *pred, real *truth, real *delta, real *error);
void weighted_sum_cpu(real *a, real *b, real *s, int num, real *c);
void weighted_delta_cpu(real *a, real *b, real *s, real *da, real *db, real *ds, int n, real *dc);

void softmax(real *input, int n, float temp, int stride, real *output);
void softmax_cpu(real *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, real *output);
void upsample_cpu(real *in, int w, int h, int c, int batch, int stride, int forward, float scale, real *out);

#ifdef GPU
#include "cuda.h"
#include "tree.h"

// > Mixed precision functions
void scale_bias_gpu(float *output, float *biases, int batch, int n, int size);
void scale_bias_gpu(double *output, double *biases, int batch, int n, int size);
void scale_bias_gpu(half_host *output, half_host *biases, int batch, int n, int size);

void add_bias_gpu(float *output, float *biases, int batch, int n, int size);
void add_bias_gpu(double *output, double *biases, int batch, int n, int size);
void add_bias_gpu(half_host *output, half_host *biases, int batch, int n, int size);

void normalize_gpu(float *x, float *mean, float *variance, int batch, int filters, int spatial);
void normalize_gpu(double *x, double *mean, double *variance, int batch, int filters, int spatial);
void normalize_gpu(half_host *x, half_host *mean, half_host *variance, int batch, int filters, int spatial);

void copy_gpu_offset(int N, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY);
void copy_gpu_offset(int N, double *X, int OFFX, int INCX, double *Y, int OFFY, int INCY);
void copy_gpu_offset(int N, half_host *X, int OFFX, int INCX, half_host *Y, int OFFY, int INCY);

void axpy_gpu_offset(int N, float ALPHA, float *X, int OFFX, int INCX, float *Y, int OFFY, int INCY);
void axpy_gpu_offset(int N, float ALPHA, double *X, int OFFX, int INCX, double *Y, int OFFY, int INCY);
void axpy_gpu_offset(int N, float ALPHA, half_host *X, int OFFX, int INCX, half_host *Y, int OFFY, int INCY);

void fast_mean_gpu(float *x, int batch, int filters, int spatial, float *mean);
void fast_mean_gpu(double *x, int batch, int filters, int spatial, double *mean);
void fast_mean_gpu(half_host *x, int batch, int filters, int spatial, half_host *mean);

void fast_variance_gpu(float *x, float *mean, int batch, int filters, int spatial, float *variance);
void fast_variance_gpu(double *x, double *mean, int batch, int filters, int spatial, double *variance);
void fast_variance_gpu(half_host *x, half_host *mean, int batch, int filters, int spatial, half_host *variance);

void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, float *out);
void shortcut_gpu(int batch, int w1, int h1, int c1, double *add, int w2, int h2, int c2, float s1, float s2, double *out);
void shortcut_gpu(int batch, int w1, int h1, int c1, half_host *add, int w2, int h2, int c2, float s1, float s2, half_host *out);
void shortcut_gpu(int batch, int w1, int h1, int c1, float *add, int w2, int h2, int c2, float s1, float s2, half_host *out);
void shortcut_gpu(int batch, int w1, int h1, int c1, half_host *add, int w2, int h2, int c2, float s1, float s2, float *out);

// > General functions
void add_gpu(int N, float ALPHA, real * X, int INCX);
void supp_gpu(int N, float ALPHA, real * X, int INCX);
void mask_gpu(int N, real * X, float mask_num, real * mask, float val);
void scale_mask_gpu(int N, real * X, float mask_num, real * mask, float scale);
void const_gpu(int N, float ALPHA, real *X, int INCX);
void pow_gpu(int N, float ALPHA, real *X, int INCX, real *Y, int INCY);
void mul_gpu(int N, real *X, int INCX, real *Y, int INCY);
void mean_gpu(real *x, int batch, int filters, int spatial, real *mean);
void variance_gpu(real *x, real *mean, int batch, int filters, int spatial, real *variance);
void l2normalize_gpu(real *x, real *dx, int batch, int filters, int spatial);
void normalize_delta_gpu(real *x, real *mean, real *variance, real *mean_delta, real *variance_delta, int batch, int filters, int spatial, real *delta);
void fast_mean_delta_gpu(real *delta, real *variance, int batch, int filters, int spatial, real *mean_delta);
void fast_variance_delta_gpu(real *x, real *delta, real *mean, real *variance, int batch, int filters, int spatial, real *variance_delta);
void backward_scale_gpu(real *x_norm, real *delta, int batch, int n, int size, real *scale_updates);
void backward_bias_gpu(real *bias_updates, real *delta, int batch, int n, int size);
void logistic_x_ent_gpu(int n, real *pred, real *truth, real *delta, real *error);
void softmax_x_ent_gpu(int n, real *pred, real *truth, real *delta, real *error);
void smooth_l1_gpu(int n, real *pred, real *truth, real *delta, real *error);
void l2_gpu(int n, real *pred, real *truth, real *delta, real *error);
void l1_gpu(int n, real *pred, real *truth, real *delta, real *error);
void wgan_gpu(int n, real *pred, real *truth, real *delta, real *error);
void weighted_delta_gpu(real *a, real *b, real *s, real *da, real *db, real *ds, int num, real *dc);
void weighted_sum_gpu(real *a, real *b, real *s, int num, real *c);
void mult_add_into_gpu(int num, real *a, real *b, real *c);
void inter_gpu(int NX, real *X, int NY, real *Y, int B, real *OUT);
void deinter_gpu(int NX, real *X, int NY, real *Y, int B, real *OUT);
void reorg_gpu(real *x, int w, int h, int c, int batch, int stride, int forward, real *out);
void softmax_gpu(real *input, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, real *output);
void adam_update_gpu(real *w, real *d, real *m, real *v, float B1, float B2, float eps, float decay, float rate, int n, int batch, int t);
void adam_gpu(int n, real *x, real *m, real *v, float B1, float B2, float rate, float eps, int t);
void flatten_gpu(real *x, int spatial, int layers, int batch, int forward, real *out);
void softmax_tree(real *input, int spatial, int batch, int stride, float temp, real *output, tree hier);

void upsample_gpu(float *in, int w, int h, int c, int batch, int stride, int forward, float scale, float *out);
void upsample_gpu(double *in, int w, int h, int c, int batch, int stride, int forward, float scale, double *out);
void upsample_gpu(half_host *in, int w, int h, int c, int batch, int stride, int forward, float scale, half_host *out);

// > Extra/Test functions
void relative_error_gpu(float *arr1, half_host* arr2, int N, float *out);
void relative_error_gpu(float *arr1, float* arr2, int N, float *out);

#endif
#endif
