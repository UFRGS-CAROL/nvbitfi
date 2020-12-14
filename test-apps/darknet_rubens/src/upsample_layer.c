#include "upsample_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

layer make_upsample_layer(int batch, int w, int h, int c, int stride, int real_type) {
    layer l = {};
    l.type = UPSAMPLE;
    l.batch = batch;
    l.w = w;
    l.h = h;
    l.c = c;
    l.out_w = w*stride;
    l.out_h = h*stride;
    l.out_c = c;
    if (stride < 0) {
        stride = -stride;
        l.reverse=1;
        l.out_w = w/stride;
        l.out_h = h/stride;
    }
    l.stride = stride;
    l.outputs = l.out_w*l.out_h*l.out_c;
    l.inputs = l.w*l.h*l.c;

    l.delta = (real*)calloc(l.outputs*batch, sizeof(real));
    l.output = (real*)calloc(l.outputs*batch, sizeof(real));;

    if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
        l.output_float = (float*)calloc(l.outputs*batch, sizeof(float));
        l.delta_float = (float*)calloc(l.outputs*batch, sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
        l.output_half = (half_host*)calloc(l.outputs*batch, sizeof(half_host));
        l.delta_half = (half_host*)calloc(l.outputs*batch, sizeof(half_host));
    }

    l.forward = forward_upsample_layer;
    l.backward = backward_upsample_layer;

    #ifdef GPU
        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type))
            l.forward_gpu = forward_upsample_layer_float_gpu;
        else if (IS_MIX_PRECISION_HALF_LAYER(real_type))
            l.forward_gpu = forward_upsample_layer_half_gpu;
        else
            l.forward_gpu = forward_upsample_layer_gpu;
            
        l.backward_gpu = backward_upsample_layer_gpu;

        l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
        l.output_gpu = cuda_make_array(l.output, l.outputs*batch);

        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.output_float_gpu  = cuda_make_float_array(l.output_float, l.outputs*batch);
            l.delta_float_gpu   = cuda_make_float_array(l.delta_float, l.outputs*batch);
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.output_half_gpu  = cuda_make_half_array(l.output_half, l.outputs*batch);
            l.delta_half_gpu   = cuda_make_half_array(l.delta_half, l.outputs*batch);
        }
    #endif

    if (l.reverse)
        fprintf(stderr, "downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    else
        fprintf(stderr, "upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d", stride, w, h, c, l.out_w, l.out_h, l.out_c);
    fprintf(stderr, "%14s - %s\n", "", get_real_string(real_type));
    return l;
}

void resize_upsample_layer(layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->out_w = w*l->stride;
    l->out_h = h*l->stride;
    if(l->reverse){
        l->out_w = w/l->stride;
        l->out_h = h/l->stride;
    }
    l->outputs = l->out_w*l->out_h*l->out_c;
    l->inputs = l->h*l->w*l->c;
    l->delta = (real*)realloc(l->delta, l->outputs*l->batch*sizeof(real));
    l->output = (real*)realloc(l->output, l->outputs*l->batch*sizeof(real));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

void forward_upsample_layer(const layer l, network net)
{
    fill_cpu(l.outputs*l.batch, 0, l.output, 1);
    if(l.reverse){
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input);
    }else{
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output);
    }
}

void backward_upsample_layer(const layer l, network net)
{
    if(l.reverse){
        upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta);
    }else{
        upsample_cpu(net.delta, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta);
    }
}

#ifdef GPU
void forward_upsample_layer_gpu(const layer l, network net) {
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);
    if (l.reverse)
        upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu);
    else
        upsample_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu);
}
void forward_upsample_layer_float_gpu(const layer l, network net) {
    fill_gpu(l.outputs*l.batch, 0, l.output_float_gpu, 1);
    if (l.reverse)
        upsample_gpu(l.output_float_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_float_gpu);
    else
        upsample_gpu(net.input_float_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_float_gpu);
}
void forward_upsample_layer_half_gpu(const layer l, network net) {
    fill_gpu(l.outputs*l.batch, 0, l.output_half_gpu, 1);
    if (l.reverse)
        upsample_gpu(l.output_half_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_half_gpu);
    else
        upsample_gpu(net.input_half_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_half_gpu);
}


void backward_upsample_layer_gpu(const layer l, network net) {
    if (l.reverse)
        upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta_gpu);
    else
        upsample_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu);
}
#endif
