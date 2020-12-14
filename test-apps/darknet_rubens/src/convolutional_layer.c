#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

// > Mixed precision functions

// #if MIX_PRECISION_SUPPORT == FLOAT

    void swap_binary_float(convolutional_layer *l) {
        float *swap = l->weights_float;
        l->weights_float = l->binary_weights_float;
        l->binary_weights_float = swap;

        #ifdef GPU
            swap = l->weights_float_gpu;
            l->weights_float_gpu = l->binary_weights_float_gpu;
            l->binary_weights_float_gpu = swap;
        #endif
    }

// #elif MIX_PRECISION_SUPPORT == HALF

    void swap_binary_half(convolutional_layer *l) {
        half_host *swap = l->weights_half;
        l->weights_half = l->binary_weights_half;
        l->binary_weights_half = swap;

        #ifdef GPU
            swap = l->weights_half_gpu;
            l->weights_half_gpu = l->binary_weights_half_gpu;
            l->binary_weights_half_gpu = swap;
        #endif
    }

// #endif

// > General functions

void swap_binary(convolutional_layer *l)
{
    real *swap = l->weights;
    l->weights = l->binary_weights;
    l->binary_weights = swap;

#ifdef GPU
    swap = l->weights_gpu;
    l->weights_gpu = l->binary_weights_gpu;
    l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(real *weights, int n, int size, real *binary)
{
    int i, f;
    for(f = 0; f < n; ++f){
        float mean = 0;
        for(i = 0; i < size; ++i){
            mean += fabs(weights[f*size + i]);
        }
        mean = mean / size;
        for(i = 0; i < size; ++i){
            binary[f*size + i] = (weights[f*size + i] > 0) ? mean : -mean;
        }
    }
}

void binarize_cpu(real *input, int n, real *binary)
{
    int i;
    for(i = 0; i < n; ++i){
        binary[i] = (input[i] > 0) ? 1 : -1;
    }
}

void binarize_input(real *input, int n, int size, real *binary)
{
    int i, s;
    for(s = 0; s < size; ++s){
        float mean = 0;
        for(i = 0; i < n; ++i){
            mean += fabs(input[i*size + s]);
        }
        mean = mean / n;
        for(i = 0; i < n; ++i){
            binary[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
        }
    }
}

int convolutional_out_height(convolutional_layer l)
{
    return (l.h + 2*l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l)
{
    return (l.w + 2*l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l)
{
    return real_to_image(l.out_w,l.out_h,l.out_c,l.output);
}

image get_convolutional_delta(convolutional_layer l)
{
    return real_to_image(l.out_w,l.out_h,l.out_c,l.delta);
}

static size_t get_workspace_size(layer l){
#ifdef CUDNN
    if(gpu_index >= 0){
        size_t most = 0;
        size_t s = 0;
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.weightDesc,
                l.convDesc,
                l.dstTensorDesc,
                l.fw_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(),
                l.srcTensorDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dweightDesc,
                l.bf_algo,
                &s);
        if (s > most) most = s;
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(),
                l.weightDesc,
                l.ddstTensorDesc,
                l.convDesc,
                l.dsrcTensorDesc,
                l.bd_algo,
                &s);
        if (s > most) most = s;
        return most;
    }
#endif
    switch (l.real_type) {
        case FLOAT: return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(float);
        case DOUBLE: return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(double);
        case HALF: return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(half_host);
        default: fprintf(stderr, "Unexpected data type (convolutional_layer.c:%d)\n", __LINE__);
            break;
    }
    
    return (size_t)l.out_h*l.out_w*l.size*l.size*l.c/l.groups*sizeof(real);
}

#ifdef GPU
#ifdef CUDNN
void cudnn_convolutional_setup(layer *l)
{
    cudnnDataType_t data_type;
    switch (l.real_type) {
        case FLOAT: data_type = CUDNN_DATA_FLOAT; break;
        case DOUBLE: data_type = CUDNN_DATA_FLOAT; break;
        case HALF: data_type = CUDNN_DATA_HALF; break;
        default: data_type = CUDNN_DATA_REAL; break;
    }

    cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w); 

    cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->c, l->h, l->w); 
    cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l->batch, l->out_c, l->out_h, l->out_w); 
    cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW, data_type, 1, l->out_c, 1, 1); 

    cudnnSetFilter4dDescriptor(l->dweightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    cudnnSetFilter4dDescriptor(l->weightDesc, data_type, CUDNN_TENSOR_NCHW, l->n, l->c/l->groups, l->size, l->size); 
    #if CUDNN_MAJOR >= 6
        cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION, data_type);
    #else
        cudnnSetConvolution2dDescriptor(l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1, CUDNN_CROSS_CORRELATION);
    #endif

    #if CUDNN_MAJOR >= 7
        cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
    #else
        if(l->groups > 1){
            error("CUDNN < 7 doesn't support groups, please upgrade!");
        }
    #endif

    cudnnGetConvolutionForwardAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->weightDesc,
            l->convDesc,
            l->dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->fw_algo);
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(),
            l->weightDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bd_algo);
    cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(),
            l->srcTensorDesc,
            l->ddstTensorDesc,
            l->convDesc,
            l->dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
            2000000000,
            &l->bf_algo);
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c, int n, int groups, int size, int stride, int padding, ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam, int real_type)
{
    int i;
    convolutional_layer l = {}; // zero init
    l.type = CONVOLUTIONAL;

    l.groups = groups;
    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.xnor = xnor;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = padding;
    l.batch_normalize = batch_normalize;

    l.real_type = real_type;

    if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
        l.weights_float = (float*)calloc(c/groups*n*size*size, sizeof(float));
        l.weight_updates_float = (float*)calloc(c/groups*n*size*size, sizeof(float));

        l.biases_float = (float*)calloc(n, sizeof(float));
        l.bias_updates_float = (float*)calloc(n, sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
        l.weights_half = (half_host*)calloc(c/groups*n*size*size, sizeof(half_host));
        l.weight_updates_half = (half_host*)calloc(c/groups*n*size*size, sizeof(half_host));

        l.biases_half = (half_host*)calloc(n, sizeof(half_host));
        l.bias_updates_half = (half_host*)calloc(n, sizeof(half_host));
    } else {
        l.weights = (real*)calloc(c/groups*n*size*size, sizeof(real));
        l.weight_updates = (real*)calloc(c/groups*n*size*size, sizeof(real));

        l.biases = (real*)calloc(n, sizeof(real));
        l.bias_updates = (real*)calloc(n, sizeof(real));
    }

    l.nweights = c/groups*n*size*size;
    l.nbiases = n;

    float scale = sqrt(2./(size*size*c/l.groups));

    if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
        for(i = 0; i < l.nweights; ++i) l.weights_float[i] = scale*rand_normal();
    } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
        for(i = 0; i < l.nweights; ++i) l.weights_half[i] = scale*rand_normal();
    } else {
        for(i = 0; i < l.nweights; ++i) l.weights[i] = scale*rand_normal();
    }

    int out_w = convolutional_out_width(l);
    int out_h = convolutional_out_height(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.output = (real*)calloc(l.batch*l.outputs, sizeof(real));
    l.delta  = (real*)calloc(l.batch*l.outputs, sizeof(real));

    if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
        l.output_float = (float*)calloc(l.batch*l.outputs, sizeof(float));
        l.delta_float = (float*)calloc(l.batch*l.outputs, sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
        l.output_half = (half_host*)calloc(l.batch*l.outputs, sizeof(half_host));
        l.delta_half = (half_host*)calloc(l.batch*l.outputs, sizeof(half_host));
    }

    l.forward = forward_convolutional_layer;
    l.backward = backward_convolutional_layer;
    l.update = update_convolutional_layer;
    if(binary){
        l.cweights = (char*)calloc(l.nweights, sizeof(char));

        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.binary_weights_float = (float*)calloc(l.nweights, sizeof(float));
            l.scales_float = (float*)calloc(n, sizeof(float));
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.binary_weights_half = (half_host*)calloc(l.nweights, sizeof(half_host));
            l.scales_half = (half_host*)calloc(n, sizeof(half_host));
        } else {
            l.binary_weights = (real*)calloc(l.nweights, sizeof(real));
            l.scales = (real*)calloc(n, sizeof(real));
        }
    }
    if(xnor){
        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.binary_weights_float = (float*)calloc(l.nweights, sizeof(float));
            l.binary_input_float = (float*)calloc(l.inputs*l.batch, sizeof(float));
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.binary_weights_half = (half_host*)calloc(l.nweights, sizeof(half_host));
            l.binary_input_half = (half_host*)calloc(l.inputs*l.batch, sizeof(half_host));
        } else {
            l.binary_weights = (real*)calloc(l.nweights, sizeof(real));
            l.binary_input = (real*)calloc(l.inputs*l.batch, sizeof(real));
        }
    }

    if(batch_normalize){
        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.scales_float = (float*)calloc(n, sizeof(float));
            l.scale_updates_float = (float*)calloc(n, sizeof(float));
            for(i = 0; i < n; ++i){
                l.scales_float[i] = 1;
            }

            l.mean_float = (float*)calloc(n, sizeof(float));
            l.variance_float = (float*)calloc(n, sizeof(float));

            l.mean_delta_float = (float*)calloc(n, sizeof(float));
            l.variance_delta_float = (float*)calloc(n, sizeof(float));

            l.rolling_mean_float = (float*)calloc(n, sizeof(float));
            l.rolling_variance_float = (float*)calloc(n, sizeof(float));
            l.x_float = (float*)calloc(l.batch*l.outputs, sizeof(float));
            l.x_norm_float = (float*)calloc(l.batch*l.outputs, sizeof(float));
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.scales_half = (half_host*)calloc(n, sizeof(half_host));
            l.scale_updates_half = (half_host*)calloc(n, sizeof(half_host));
            for(i = 0; i < n; ++i){
                l.scales_half[i] = 1.0_h;
            }

            l.mean_half = (half_host*)calloc(n, sizeof(half_host));
            l.variance_half = (half_host*)calloc(n, sizeof(half_host));

            l.mean_delta_half = (half_host*)calloc(n, sizeof(half_host));
            l.variance_delta_half = (half_host*)calloc(n, sizeof(half_host));

            l.rolling_mean_half = (half_host*)calloc(n, sizeof(half_host));
            l.rolling_variance_half = (half_host*)calloc(n, sizeof(half_host));
            l.x_half = (half_host*)calloc(l.batch*l.outputs, sizeof(half_host));
            l.x_norm_half = (half_host*)calloc(l.batch*l.outputs, sizeof(half_host));
        } else {
            l.scales = (real*)calloc(n, sizeof(real));
            l.scale_updates = (real*)calloc(n, sizeof(real));
            for(i = 0; i < n; ++i){
                l.scales[i] = 1;
            }

            l.mean = (real*)calloc(n, sizeof(real));
            l.variance = (real*)calloc(n, sizeof(real));

            l.mean_delta = (real*)calloc(n, sizeof(real));
            l.variance_delta = (real*)calloc(n, sizeof(real));

            l.rolling_mean = (real*)calloc(n, sizeof(real));
            l.rolling_variance = (real*)calloc(n, sizeof(real));
            l.x = (real*)calloc(l.batch*l.outputs, sizeof(real));
            l.x_norm = (real*)calloc(l.batch*l.outputs, sizeof(real));
        }
    }
    if(adam){
        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.m_float = (float*)calloc(l.nweights, sizeof(float));
            l.v_float = (float*)calloc(l.nweights, sizeof(float));
            l.bias_m_float = (float*)calloc(n, sizeof(float));
            l.scale_m_float = (float*)calloc(n, sizeof(float));
            l.bias_v_float = (float*)calloc(n, sizeof(float));
            l.scale_v_float = (float*)calloc(n, sizeof(float));
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.m_half = (half_host*)calloc(l.nweights, sizeof(half_host));
            l.v_half = (half_host*)calloc(l.nweights, sizeof(half_host));
            l.bias_m_half = (half_host*)calloc(n, sizeof(half_host));
            l.scale_m_half = (half_host*)calloc(n, sizeof(half_host));
            l.bias_v_half = (half_host*)calloc(n, sizeof(half_host));
            l.scale_v_half = (half_host*)calloc(n, sizeof(half_host));
        } else {
            l.m = (real*)calloc(l.nweights, sizeof(real));
            l.v = (real*)calloc(l.nweights, sizeof(real));
            l.bias_m = (real*)calloc(n, sizeof(real));
            l.scale_m = (real*)calloc(n, sizeof(real));
            l.bias_v = (real*)calloc(n, sizeof(real));
            l.scale_v = (real*)calloc(n, sizeof(real));
        }
    }

#ifdef GPU
    if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
        l.forward_gpu = forward_convolutional_layer_float_gpu;
    } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
        l.forward_gpu = forward_convolutional_layer_half_gpu;
    } else {
        l.forward_gpu = forward_convolutional_layer_gpu;
    }
    l.backward_gpu = backward_convolutional_layer_gpu;
    l.update_gpu = update_convolutional_layer_gpu;

    if(gpu_index >= 0){
        if (adam) {
            if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
                l.m_float_gpu = cuda_make_float_array(l.m_float, l.nweights);
                l.v_float_gpu = cuda_make_float_array(l.v_float, l.nweights);
                l.bias_m_float_gpu = cuda_make_float_array(l.bias_m_float, n);
                l.bias_v_float_gpu = cuda_make_float_array(l.bias_v_float, n);
                l.scale_m_float_gpu = cuda_make_float_array(l.scale_m_float, n);
                l.scale_v_float_gpu = cuda_make_float_array(l.scale_v_float, n);
            } if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
                l.m_half_gpu = cuda_make_half_array(l.m_half, l.nweights);
                l.v_half_gpu = cuda_make_half_array(l.v_half, l.nweights);
                l.bias_m_half_gpu = cuda_make_half_array(l.bias_m_half, n);
                l.bias_v_half_gpu = cuda_make_half_array(l.bias_v_half, n);
                l.scale_m_half_gpu = cuda_make_half_array(l.scale_m_half, n);
                l.scale_v_half_gpu = cuda_make_half_array(l.scale_v_half, n);
            } else {
                l.m_gpu = cuda_make_array(l.m, l.nweights);
                l.v_gpu = cuda_make_array(l.v, l.nweights);
                l.bias_m_gpu = cuda_make_array(l.bias_m, n);
                l.bias_v_gpu = cuda_make_array(l.bias_v, n);
                l.scale_m_gpu = cuda_make_array(l.scale_m, n);
                l.scale_v_gpu = cuda_make_array(l.scale_v, n);
            }
        }

        l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
        l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.weights_float_gpu = cuda_make_float_array(l.weights_float, l.nweights);
            l.weight_updates_float_gpu = cuda_make_float_array(l.weight_updates_float, l.nweights);

            l.biases_float_gpu = cuda_make_float_array(l.biases_float, n);
            l.bias_updates_float_gpu = cuda_make_float_array(l.bias_updates_float, n);

            l.delta_float_gpu = cuda_make_float_array(l.delta_float, l.batch*out_h*out_w*n);
            l.output_float_gpu = cuda_make_float_array(l.output_float, l.batch*out_h*out_w*n);
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.weights_half_gpu = cuda_make_half_array(l.weights_half, l.nweights);
            l.weight_updates_half_gpu = cuda_make_half_array(l.weight_updates_half, l.nweights);

            l.biases_half_gpu = cuda_make_half_array(l.biases_half, n);
            l.bias_updates_half_gpu = cuda_make_half_array(l.bias_updates_half, n);

            l.delta_half_gpu = cuda_make_half_array(l.delta_half, l.batch*out_h*out_w*n);
            l.output_half_gpu = cuda_make_half_array(l.output_half, l.batch*out_h*out_w*n);
        } else {
            l.weights_gpu = cuda_make_array(l.weights, l.nweights);
            l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

            l.biases_gpu = cuda_make_array(l.biases, n);
            l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);
        }

        if(binary){
            if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
                l.binary_weights_float_gpu = cuda_make_float_array(l.weights_float, l.nweights);
            } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
                l.binary_weights_half_gpu = cuda_make_half_array(l.weights_half, l.nweights);
            } else {
                l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
            }
        }
        if(xnor){
            if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
                l.binary_weights_float_gpu = cuda_make_float_array(l.weights_float, l.nweights);
                l.binary_input_float_gpu = cuda_make_float_array(0, l.inputs*l.batch);
            } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
                l.binary_weights_half_gpu = cuda_make_half_array(l.weights_half, l.nweights);
                l.binary_input_half_gpu = cuda_make_half_array(0, l.inputs*l.batch);
            } else {
                l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
                l.binary_input_gpu = cuda_make_array(0, l.inputs*l.batch);
            }
        }

        if(batch_normalize){
            if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
                l.mean_float_gpu = cuda_make_float_array(l.mean_float, n);
                l.variance_float_gpu = cuda_make_float_array(l.variance_float, n);

                l.rolling_mean_float_gpu = cuda_make_float_array(l.mean_float, n);
                l.rolling_variance_float_gpu = cuda_make_float_array(l.variance_float, n);

                l.mean_delta_float_gpu = cuda_make_float_array(l.mean_float, n);
                l.variance_delta_float_gpu = cuda_make_float_array(l.variance_float, n);

                l.scales_float_gpu = cuda_make_float_array(l.scales_float, n);
                l.scale_updates_float_gpu = cuda_make_float_array(l.scale_updates_float, n);

                l.x_float_gpu = cuda_make_float_array(l.output_float, l.batch*out_h*out_w*n);
                l.x_norm_float_gpu = cuda_make_float_array(l.output_float, l.batch*out_h*out_w*n);
            } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
                l.mean_half_gpu = cuda_make_half_array(l.mean_half, n);
                l.variance_half_gpu = cuda_make_half_array(l.variance_half, n);

                l.rolling_mean_half_gpu = cuda_make_half_array(l.mean_half, n);
                l.rolling_variance_half_gpu = cuda_make_half_array(l.variance_half, n);

                l.mean_delta_half_gpu = cuda_make_half_array(l.mean_half, n);
                l.variance_delta_half_gpu = cuda_make_half_array(l.variance_half, n);

                l.scales_half_gpu = cuda_make_half_array(l.scales_half, n);
                l.scale_updates_half_gpu = cuda_make_half_array(l.scale_updates_half, n);

                l.x_half_gpu = cuda_make_half_array(l.output_half, l.batch*out_h*out_w*n);
                l.x_norm_half_gpu = cuda_make_half_array(l.output_half, l.batch*out_h*out_w*n);
            } else {
                l.mean_gpu = cuda_make_array(l.mean, n);
                l.variance_gpu = cuda_make_array(l.variance, n);

                l.rolling_mean_gpu = cuda_make_array(l.mean, n);
                l.rolling_variance_gpu = cuda_make_array(l.variance, n);

                l.mean_delta_gpu = cuda_make_array(l.mean, n);
                l.variance_delta_gpu = cuda_make_array(l.variance, n);

                l.scales_gpu = cuda_make_array(l.scales, n);
                l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

                l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
                l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
            }
        }
#ifdef CUDNN
        cudnnCreateTensorDescriptor(&l.normTensorDesc);
        cudnnCreateTensorDescriptor(&l.srcTensorDesc);
        cudnnCreateTensorDescriptor(&l.dstTensorDesc);
        cudnnCreateFilterDescriptor(&l.weightDesc);
        cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
        cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
        cudnnCreateFilterDescriptor(&l.dweightDesc);
        cudnnCreateConvolutionDescriptor(&l.convDesc);
        cudnn_convolutional_setup(&l);
#endif
    }
#endif
    l.workspace_size = get_workspace_size(l);
    l.activation = activation;

    fprintf(stderr, "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d  %5.3f BFLOPs - %s\n", n, size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c, (2.0 * l.n * l.size*l.size*l.c/l.groups * l.out_h*l.out_w)/1000000000., get_real_string(real_type));

    return l;
}

void denormalize_convolutional_layer(convolutional_layer l)
{
    int i, j;
    for(i = 0; i < l.n; ++i){
        float scale = l.scales[i]/sqrt(l.rolling_variance[i] + .00001);
        for(j = 0; j < l.c/l.groups*l.size*l.size; ++j){
            l.weights[i*l.c/l.groups*l.size*l.size + j] *= scale;
        }
        l.biases[i] -= l.rolling_mean[i] * scale;
        l.scales[i] = 1;
        l.rolling_mean[i] = 0;
        l.rolling_variance[i] = 1;
    }
}

/*
void test_convolutional_layer()
{
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 2, 5, 2, 1, LEAKY, 1, 0, 0, 0);
    l.batch_normalize = 1;
    float data[] = {1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        1,1,1,1,1,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        2,2,2,2,2,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3,
        3,3,3,3,3};
    //net.input = data;
    //forward_convolutional_layer(l);
}
*/

void resize_convolutional_layer(convolutional_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    int out_w = convolutional_out_width(*l);
    int out_h = convolutional_out_height(*l);

    l->out_w = out_w;
    l->out_h = out_h;

    l->outputs = l->out_h * l->out_w * l->out_c;
    l->inputs = l->w * l->h * l->c;

    l->output = (real*)realloc(l->output, l->batch*l->outputs*sizeof(real));
    l->delta  = (real*)realloc(l->delta,  l->batch*l->outputs*sizeof(real));

    if (IS_MIX_PRECISION_FLOAT_LAYER(l->real_type)) {
        l->output_float = (float*)realloc(l->output_float, l->batch*l->outputs*sizeof(float));
        l->delta_float  = (float*)realloc(l->delta_float,  l->batch*l->outputs*sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(l->real_type)) {
        l->output_half = (half_host*)realloc(l->output_half, l->batch*l->outputs*sizeof(half_host));
        l->delta_half  = (half_host*)realloc(l->delta_half,  l->batch*l->outputs*sizeof(half_host));
    }

    if(l->batch_normalize){
        if (IS_MIX_PRECISION_FLOAT_LAYER(l->real_type)) {
            l->x_float = (float*)realloc(l->x_float, l->batch*l->outputs*sizeof(float));
            l->x_norm_float  = (float*)realloc(l->x_norm_float, l->batch*l->outputs*sizeof(float));
        } else if (IS_MIX_PRECISION_HALF_LAYER(l->real_type)) {
            l->x_half = (half_host*)realloc(l->x_half, l->batch*l->outputs*sizeof(half_host));
            l->x_norm_half  = (half_host*)realloc(l->x_norm_half, l->batch*l->outputs*sizeof(half_host));
        } else {
            l->x = (real*)realloc(l->x, l->batch*l->outputs*sizeof(real));
            l->x_norm  = (real*)realloc(l->x_norm, l->batch*l->outputs*sizeof(real));
        }
    }

#ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);

    l->delta_gpu =  cuda_make_array(l->delta,  l->batch*l->outputs);
    l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);

    if (IS_MIX_PRECISION_FLOAT_LAYER(l->real_type)) {
        cuda_free(l->delta_float_gpu);
        cuda_free(l->output_float_gpu);

        l->delta_float_gpu =  cuda_make_float_array(l->delta_float,  l->batch*l->outputs);
        l->output_float_gpu = cuda_make_float_array(l->output_float, l->batch*l->outputs);
    } else if (IS_MIX_PRECISION_HALF_LAYER(l->real_type)) {
        cuda_free(l->delta_half_gpu);
        cuda_free(l->output_half_gpu);

        l->delta_half_gpu =  cuda_make_half_array(l->delta_half,  l->batch*l->outputs);
        l->output_half_gpu = cuda_make_half_array(l->output_half, l->batch*l->outputs);
    }

    if(l->batch_normalize){
        if (IS_MIX_PRECISION_FLOAT_LAYER(l->real_type)) {
            cuda_free(l->x_float_gpu);
            cuda_free(l->x_norm_float_gpu);

            l->x_float_gpu = cuda_make_float_array(l->output_float, l->batch*l->outputs);
            l->x_norm_float_gpu = cuda_make_float_array(l->output_float, l->batch*l->outputs);
        } else if (IS_MIX_PRECISION_HALF_LAYER(l->real_type)) {
            cuda_free(l->x_half_gpu);
            cuda_free(l->x_norm_half_gpu);

            l->x_half_gpu = cuda_make_half_array(l->output_half, l->batch*l->outputs);
            l->x_norm_half_gpu = cuda_make_half_array(l->output_half, l->batch*l->outputs);
        } else {
            cuda_free(l->x_gpu);
            cuda_free(l->x_norm_gpu);

            l->x_gpu = cuda_make_array(l->output, l->batch*l->outputs);
            l->x_norm_gpu = cuda_make_array(l->output, l->batch*l->outputs);
        }
    }
#ifdef CUDNN
    cudnn_convolutional_setup(l);
#endif
#endif
    l->workspace_size = get_workspace_size(*l);
}

void add_bias(real *output, real *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }
        }
    }
}

void scale_bias(real *output, real *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(real *bias_updates, real *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}

void forward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(l.xnor){
        binarize_weights(l.weights, l.n, l.c/l.groups*l.size*l.size, l.binary_weights);
        swap_binary(&l);
        binarize_cpu(net.input, l.c*l.h*l.w*l.batch, l.binary_input);
        net.input = l.binary_input;
    }

    int m = l.n/l.groups;
    int k = l.size*l.size*l.c/l.groups;
    int n = l.out_w*l.out_h;
    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            real *a = l.weights + j*l.nweights/l.groups;
            real *b = net.workspace;
            real *c = l.output + (i*l.groups + j)*n*m;
            real *im =  net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if (l.size == 1) {
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, b);
            }
            gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        }
    }

    if(l.batch_normalize){
        forward_batchnorm_layer(l, net);
    } else {
        add_bias(l.output, l.biases, l.batch, l.n, l.out_h*l.out_w);
    }

    activate_array(l.output, l.outputs*l.batch, l.activation);
    if(l.binary || l.xnor) swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net)
{
    int i, j;
    int m = l.n/l.groups;
    int n = l.size*l.size*l.c/l.groups;
    int k = l.out_w*l.out_h;

    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);

    if(l.batch_normalize){
        backward_batchnorm_layer(l, net);
    } else {
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    }

    for(i = 0; i < l.batch; ++i){
        for(j = 0; j < l.groups; ++j){
            real *a = l.delta + (i*l.groups + j)*m*k;
            real *b = net.workspace;
            real *c = l.weight_updates + j*l.nweights/l.groups;

            real *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
            real *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

            if(l.size == 1){
                b = im;
            } else {
                im2col_cpu(im, l.c/l.groups, l.h, l.w, 
                        l.size, l.stride, l.pad, b);
            }

            gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

            if (net.delta) {
                a = l.weights + j*l.nweights/l.groups;
                b = l.delta + (i*l.groups + j)*m*k;
                c = net.workspace;
                if (l.size == 1) {
                    c = imd;
                }

                gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

                if (l.size != 1) {
                    col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
                }
            }
        }
    }
}

void update_convolutional_layer(convolutional_layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    if(l.scales){
        axpy_cpu(l.n, learning_rate/batch, l.scale_updates, 1, l.scales, 1);
        scal_cpu(l.n, momentum, l.scale_updates, 1);
    }

    axpy_cpu(l.nweights, -decay*batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate/batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}


image get_convolutional_weight(convolutional_layer l, int i)
{
    int h = l.size;
    int w = l.size;
    int c = l.c/l.groups;
    return real_to_image(w,h,c,l.weights+i*h*w*c);
}

void rgbgr_weights(convolutional_layer l)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            rgbgr_image(im);
        }
    }
}

void rescale_weights(convolutional_layer l, float scale, float trans)
{
    int i;
    for(i = 0; i < l.n; ++i){
        image im = get_convolutional_weight(l, i);
        if (im.c == 3) {
            scale_image(im, scale);
            float sum = sum_array(im.data, im.w*im.h*im.c);
            l.biases[i] += sum*trans;
        }
    }
}

image *get_weights(convolutional_layer l)
{
    image *weights = (image*)calloc(l.n, sizeof(image));
    int i;
    for(i = 0; i < l.n; ++i){
        weights[i] = copy_image(get_convolutional_weight(l, i));
        normalize_image(weights[i]);
        /*
           char buff[256];
           sprintf(buff, "filter%d", i);
           save_image(weights[i], buff);
         */
    }
    //error("hey");
    return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window, image *prev_weights)
{
    image *single_weights = get_weights(l);
    show_images(single_weights, l.n, window);

    image delta = get_convolutional_image(l);
    image dc = collapse_image_layers(delta, 1);
    char buff[256];
    sprintf(buff, "%s: Output", window);
    //show_image(dc, buff);
    //save_image(dc, buff);
    free_image(dc);
    return single_weights;
}

