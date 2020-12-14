#include "maxpool_layer.h"
#include "cuda.h"
#include <stdio.h>

image get_maxpool_image(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return real_to_image(w,h,c,l.output);
}

image get_maxpool_delta(maxpool_layer l)
{
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    return real_to_image(w,h,c,l.delta);
}

maxpool_layer make_maxpool_layer(int batch, int h, int w, int c, int size, int stride, int padding, int real_type)
{
    maxpool_layer l = {}; // zero init
    l.type = MAXPOOL;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.pad = padding;
    l.out_w = (w + padding - size)/stride + 1;
    l.out_h = (h + padding - size)/stride + 1;
    l.out_c = c;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = h*w*c;
    l.size = size;
    l.stride = stride;

    l.real_type = real_type;

    int output_size = l.out_h * l.out_w * l.out_c * batch;
    l.indexes = (int*)calloc(output_size, sizeof(int));

    l.output = (real*)calloc(output_size, sizeof(real));
    l.delta = (real*)calloc(output_size, sizeof(real));

    if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
        l.output_float = (float*)calloc(output_size, sizeof(float));
        l.delta_float = (float*)calloc(output_size, sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
        l.output_half = (half_host*)calloc(output_size, sizeof(half_host));
        l.delta_half = (half_host*)calloc(output_size, sizeof(half_host));
    }

    l.forward = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;

    #ifdef GPU
        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.forward_gpu = forward_maxpool_layer_float_gpu;
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.forward_gpu = forward_maxpool_layer_half_gpu;
        } else {
            l.forward_gpu = forward_maxpool_layer_gpu;
        }
        l.backward_gpu = backward_maxpool_layer_gpu;

        l.indexes_gpu = cuda_make_int_array(0, output_size);

        l.output_gpu  = cuda_make_array(l.output, output_size);
        l.delta_gpu   = cuda_make_array(l.delta, output_size);

        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.output_float_gpu  = cuda_make_float_array(l.output_float, output_size);
            l.delta_float_gpu   = cuda_make_float_array(l.delta_float, output_size);
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.output_half_gpu  = cuda_make_half_array(l.output_half, output_size);
            l.delta_half_gpu   = cuda_make_half_array(l.delta_half, output_size);
        }
    #endif
    fprintf(stderr, "max          %d x %d / %d  %4d x%4d x%4d   ->  %4d x%4d x%4d", size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);
    fprintf(stderr, "%14s - %s\n", "", get_real_string(real_type));
    return l;
}

void resize_maxpool_layer(maxpool_layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = h*w*l->c;

    l->out_w = (w + l->pad - l->size)/l->stride + 1;
    l->out_h = (h + l->pad - l->size)/l->stride + 1;
    l->outputs = l->out_w * l->out_h * l->c;
    int output_size = l->outputs * l->batch;

    l->indexes = (int*)realloc(l->indexes, output_size * sizeof(int));
    l->output = (real*)realloc(l->output, output_size * sizeof(real));
    l->delta = (real*)realloc(l->delta, output_size * sizeof(real));

    if (IS_MIX_PRECISION_FLOAT_LAYER(l->real_type)) {
        l->output_float = (float*)realloc(l->output_float, output_size * sizeof(float));
        l->delta_float = (float*)realloc(l->delta_float, output_size * sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(l->real_type)) {
        l->output_half = (half_host*)realloc(l->output_half, output_size * sizeof(half_host));
        l->delta_half = (half_host*)realloc(l->delta_half, output_size * sizeof(half_host));
    }

    #ifdef GPU
        cuda_free((real *)l->indexes_gpu);
        cuda_free(l->output_gpu);
        cuda_free(l->delta_gpu);
        l->indexes_gpu = cuda_make_int_array(0, output_size);
        l->output_gpu  = cuda_make_array(l->output, output_size);
        l->delta_gpu   = cuda_make_array(l->delta,  output_size);

        if (IS_MIX_PRECISION_FLOAT_LAYER(l->real_type)) {
            cuda_free(l->output_float_gpu);
            cuda_free(l->delta_float_gpu);
            l->output_float_gpu  = cuda_make_float_array(l->output_float, output_size);
            l->delta_float_gpu   = cuda_make_float_array(l->delta_float, output_size);
        } else if (IS_MIX_PRECISION_HALF_LAYER(l->real_type)) {
            cuda_free(l->output_half_gpu);
            cuda_free(l->delta_half_gpu);
            l->output_half_gpu  = cuda_make_half_array(l->output_half, output_size);
            l->delta_half_gpu   = cuda_make_half_array(l->delta_half, output_size);
        }
    #endif
}

void forward_maxpool_layer(const maxpool_layer l, network net)
{
    int b,i,j,k,m,n;
    int w_offset = -l.pad/2;
    int h_offset = -l.pad/2;

    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < c; ++k){
            for(i = 0; i < h; ++i){
                for(j = 0; j < w; ++j){
                    int out_index = j + w*(i + h*(k + c*b));
                    float max = -REAL_MAX;
                    int max_i = -1;
                    for(n = 0; n < l.size; ++n){
                        for(m = 0; m < l.size; ++m){
                            int cur_h = h_offset + i*l.stride + n;
                            int cur_w = w_offset + j*l.stride + m;
                            int index = cur_w + l.w*(cur_h + l.h*(k + b*l.c));
                            int valid = (cur_h >= 0 && cur_h < l.h &&
                                         cur_w >= 0 && cur_w < l.w);
                            float val = (valid != 0) ? net.input[index] : -REAL_MAX;
                            max_i = (val > max) ? index : max_i;
                            max   = (val > max) ? val   : max;
                        }
                    }
                    l.output[out_index] = max;
                    l.indexes[out_index] = max_i;
                }
            }
        }
    }
}

void backward_maxpool_layer(const maxpool_layer l, network net)
{
    int i;
    int h = l.out_h;
    int w = l.out_w;
    int c = l.c;
    for(i = 0; i < h*w*c*l.batch; ++i){
        int index = l.indexes[i];
        net.delta[index] += l.delta[i];
    }
}

