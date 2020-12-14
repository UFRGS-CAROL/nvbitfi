#ifndef GEMM_H
#define GEMM_H

#include "darknet.h"

void gemm_bin(int M, int N, int K, float ALPHA, 
        char  *A, int lda, 
        real *B, int ldb,
        real *C, int ldc);
        
void gemm(int TA, int TB, int M, int N, int K, float ALPHA, 
                    real *A, int lda, 
                    real *B, int ldb,
                    float BETA,
                    real *C, int ldc);

void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA, 
        real *A, int lda, 
        real *B, int ldb,
        float BETA,
        real *C, int ldc);

#ifdef GPU

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        float *A_gpu, int lda,
        float *B_gpu, int ldb,
        float BETA,
        float *C_gpu, int ldc);

void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        double *A_gpu, int lda,
        double *B_gpu, int ldb,
        float BETA,
        double *C_gpu, int ldc);
        
void gemm_gpu(int TA, int TB, int M, int N, int K, float ALPHA,
        half_host *A_gpu, int lda,
        half_host *B_gpu, int ldb,
        float BETA,
        half_host *C_gpu, int ldc);

#endif
#endif
