#ifndef UTILS_TEMPLATES
#define UTILS_TEMPLATES

template<typename T>
void top_k(T *a, int n, int k, int *index) {
    int i,j;
    for(j = 0; j < k; ++j) index[j] = -1;
    for(i = 0; i < n; ++i){
        int curr = i;
        for(j = 0; j < k; ++j){
            if((index[j] < 0) || a[curr] > a[index[j]]){
                int swap = curr;
                curr = index[j];
                index[j] = swap;
            }
        }
    }
}

template<typename T>
void scale_array(T *a, int n, float s) {
    int i;
    for(i = 0; i < n; ++i){
        a[i] *= s;
    }
}

template<typename T>
float dot_cpu(int N, T *X, int INCX, T *Y, int INCY) {
    int i;
    float dot = 0;
    for(i = 0; i < N; ++i) dot += X[i*INCX] * Y[i*INCY];
    return dot;
}

template<typename T>
int max_index(T *a, int n) {
    if(n <= 0) return -1;
    int i, max_i = 0;
    float max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

template<typename T>
float sum_array(T *a, int n) {
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}

template<typename T>
int sample_array(T *a, int n) {
    float sum = sum_array(a, n);
    scale_array(a, n, 1.0/sum);
    float r = rand_uniform(0.0, 1.0);
    int i;
    for(i = 0; i < n; ++i){
        r = r - a[i];
        if (r <= 0) return i;
    }
    return n-1;
}

template<typename T>
float mean_array(T *a, int n) {
    return sum_array(a,n)/n;
}

template<typename T>
float variance_array(T *a, int n) {
    int i;
    float sum = 0;
    float mean = mean_array(a, n);
    for(i = 0; i < n; ++i) sum += (a[i] - mean)*(a[i]-mean);
    float variance = sum/n;
    return variance;
}

template<typename T>
float mag_array(T *a, int n) {
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        sum += a[i]*a[i];   
    }
    return sqrt(sum);
}

#endif