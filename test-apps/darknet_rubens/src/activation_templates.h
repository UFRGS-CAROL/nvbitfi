#ifndef ACTIVATIONS_TEMPLATES
#define ACTIVATIONS_TEMPLATES

// > Dependencies declaration

#include "activations.h"

// > Templates

template<typename T>
void activate_array_gpu(T *x, int n, ACTIVATION a);
void activate_array_gpu(half_host *x, int n, ACTIVATION a);

template<typename T>
void gradient_array_gpu(T *x, int n, ACTIVATION a, T *delta);
void gradient_array_gpu(half_host *x, int n, ACTIVATION a, half_host *delta);

#endif