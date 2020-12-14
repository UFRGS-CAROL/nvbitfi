#ifndef BLAS_TEMPLATES
#define BLAS_TEMPLATES

template<typename T>
void axpy_cpu(int N, float ALPHA, T *X, int INCX, T *Y, int INCY) {
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

template<typename T>
void copy_cpu(int N, T *X, int INCX, T *Y, int INCY) {
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}

template<typename T>
void scal_cpu(int N, float ALPHA, T *X, int INCX) {
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}

template<typename T>
void fill_cpu(int N, float ALPHA, T *X, int INCX) {
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] = ALPHA;
}

template<typename T>
void normalize_cpu(T *x, T *mean, T *variance, int batch, int filters, int spatial) {
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f]) + T(.000001f));
            }
        }
    }
}

#endif