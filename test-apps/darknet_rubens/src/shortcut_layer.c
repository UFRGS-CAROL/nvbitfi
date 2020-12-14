#include "shortcut_layer.h"
#include "cuda.h"
#include "blas.h"
#include "activations.h"

#include <stdio.h>
#include <assert.h>

layer make_shortcut_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2, int real_type) {
    layer l = {};
    l.type = SHORTCUT;
    l.batch = batch;
    l.w = w2;
    l.h = h2;
    l.c = c2;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = w*h*c;
    l.inputs = l.outputs;

    l.index = index;

    l.delta = (real*)calloc(l.outputs*batch, sizeof(real));
    l.output = (real*)calloc(l.outputs*batch, sizeof(real));

    int output_size = l.outputs*batch;

    if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
        l.output_float = (float*)calloc(output_size, sizeof(float));
        l.delta_float = (float*)calloc(output_size, sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
        l.output_half = (half_host*)calloc(output_size, sizeof(half_host));
        l.delta_half = (half_host*)calloc(output_size, sizeof(half_host));
    }

    l.forward = forward_shortcut_layer;
    l.backward = backward_shortcut_layer;

    #ifdef GPU
        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.forward_gpu = forward_shortcut_layer_float_gpu;
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.forward_gpu = forward_shortcut_layer_half_gpu;
        } else {
            l.forward_gpu = forward_shortcut_layer_gpu;
        }
        l.backward_gpu = backward_shortcut_layer_gpu;

        l.delta_gpu =  cuda_make_array(l.delta, l.outputs*batch);
        l.output_gpu = cuda_make_array(l.output, l.outputs*batch);

        if (IS_MIX_PRECISION_FLOAT_LAYER(real_type)) {
            l.output_float_gpu  = cuda_make_float_array(l.output_float, output_size);
            l.delta_float_gpu   = cuda_make_float_array(l.delta_float, output_size);
        } else if (IS_MIX_PRECISION_HALF_LAYER(real_type)) {
            l.output_half_gpu  = cuda_make_half_array(l.output_half, output_size);
            l.delta_half_gpu   = cuda_make_half_array(l.delta_half, output_size);
        }
    #endif

    fprintf(stderr, "res  %3d                %4d x%4d x%4d   ->  %4d x%4d x%4d", index, w2, h2, c2, w, h, c);
    fprintf(stderr, "%14s - %s\n", "", get_real_string(real_type));

    return l;
}

void resize_shortcut_layer(layer *l, int w, int h) {
    assert(l->w == l->out_w);
    assert(l->h == l->out_h);
    l->w = l->out_w = w;
    l->h = l->out_h = h;
    l->outputs = w*h*l->out_c;
    l->inputs = l->outputs;
    l->delta = (real*)realloc(l->delta, l->outputs*l->batch*sizeof(real));
    l->output = (real*)realloc(l->output, l->outputs*l->batch*sizeof(real));

    int output_size = l->outputs*l->batch;

    if (IS_MIX_PRECISION_FLOAT_LAYER(l->real_type)) {
        l->output_float = (float*)realloc(l->output_float, output_size * sizeof(float));
        l->delta_float = (float*)realloc(l->delta_float, output_size * sizeof(float));
    } else if (IS_MIX_PRECISION_HALF_LAYER(l->real_type)) {
        l->output_half = (half_host*)realloc(l->output_half, output_size * sizeof(half_host));
        l->delta_half = (half_host*)realloc(l->delta_half, output_size * sizeof(half_host));
    }

    #ifdef GPU
        cuda_free(l->output_gpu);
        cuda_free(l->delta_gpu);
        l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
        l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);

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

void forward_shortcut_layer(const layer l, network net)
{
    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
    activate_array(l.output, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer(const layer l, network net)
{
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    axpy_cpu(l.outputs*l.batch, l.alpha, l.delta, 1, net.delta, 1);
    shortcut_cpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta);
}

#ifdef GPU
void forward_shortcut_layer_gpu(const layer l, network net) {
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);

    layer fromLayer = net.layers[l.index];

    if (fromLayer.real_type == REAL)
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    else if (IS_MIX_PRECISION_FLOAT_LAYER(fromLayer.real_type))
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_float_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);
    else if (IS_MIX_PRECISION_HALF_LAYER(fromLayer.real_type))
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_half_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_gpu);    

    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
}

void forward_shortcut_layer_half_gpu(const layer l, network net) {
    copy_gpu(l.outputs*l.batch, net.input_half_gpu, 1, l.output_half_gpu, 1);

    layer fromLayer = net.layers[l.index];

    if (fromLayer.real_type == REAL)
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_half_gpu);
    else if (IS_MIX_PRECISION_FLOAT_LAYER(fromLayer.real_type))
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_float_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_half_gpu);
    else if (IS_MIX_PRECISION_HALF_LAYER(fromLayer.real_type))
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_half_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_half_gpu);
    
    activate_array_gpu(l.output_half_gpu, l.outputs*l.batch, l.activation);
}

void forward_shortcut_layer_float_gpu(const layer l, network net) {
    copy_gpu(l.outputs*l.batch, net.input_float_gpu, 1, l.output_float_gpu, 1);

    layer fromLayer = net.layers[l.index];

    if (fromLayer.real_type == REAL)
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_float_gpu);
    else if (IS_MIX_PRECISION_FLOAT_LAYER(fromLayer.real_type))
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_float_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_float_gpu);
    else if (IS_MIX_PRECISION_HALF_LAYER(fromLayer.real_type))
        shortcut_gpu(l.batch, l.w, l.h, l.c, fromLayer.output_half_gpu, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output_float_gpu);
    
    activate_array_gpu(l.output_float_gpu, l.outputs*l.batch, l.activation);
}

void backward_shortcut_layer_gpu(const layer l, network net)
{
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    axpy_gpu(l.outputs*l.batch, l.alpha, l.delta_gpu, 1, net.delta_gpu, 1);
    shortcut_gpu(l.batch, l.out_w, l.out_h, l.out_c, l.delta_gpu, l.w, l.h, l.c, 1, l.beta, net.layers[l.index].delta_gpu);
}
#endif
