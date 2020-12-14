#include "route_layer.h"
#include "cuda.h"
#include "blas.h"
#include "utils.h"

#include <stdio.h>

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes) {
    fprintf(stderr,"route ");
    route_layer l = {}; // zero init
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;

    int i;
    int outputs = 0;
    for (i = 0; i < n; ++i) {
        fprintf(stderr, " %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr, "\n");

    l.outputs = outputs;
    l.inputs = outputs;
    l.delta = (real*)calloc(outputs*batch, sizeof(real));
    l.output = (real*)calloc(outputs*batch, sizeof(real));

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    #ifdef GPU
        l.forward_gpu = forward_route_layer_gpu;
        l.backward_gpu = backward_route_layer_gpu;

        l.delta_gpu =  cuda_make_array(l.delta, outputs*batch);
        l.output_gpu = cuda_make_array(l.output, outputs*batch);
    #endif
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("%d %d, %d %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
        }
    }
    l->inputs = l->outputs;
    l->delta = (real*)realloc(l->delta, l->outputs*l->batch*sizeof(real));
    l->output = (real*)realloc(l->output, l->outputs*l->batch*sizeof(real));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, l->outputs*l->batch);
    l->delta_gpu   = cuda_make_array(l->delta,  l->outputs*l->batch);
#endif
    
}

void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        real *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            copy_cpu(input_size, input + j*input_size, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += input_size;
    }
}

void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        real *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_cpu(input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];

        layer inputLayer = net.layers[index];
        int input_size = l.input_sizes[i];

        if (IS_MIX_PRECISION_HALF_LAYER(l.real_type)) { // to = l.output_half_gpu
            if (inputLayer.real_type == REAL) { // from = inputLayer.output_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_gpu + j*input_size, l.output_half_gpu + offset + j*l.outputs, input_size);
            } else if (inputLayer.real_type == FLOAT) { // inputLayer.output_float_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_float_gpu + j*input_size, l.output_half_gpu + offset + j*l.outputs, input_size);
            } else if (inputLayer.real_type == HALF) { // inputLayer.output_half_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_half_gpu + j*input_size, l.output_half_gpu + offset + j*l.outputs, input_size);
            }
        } else if (IS_MIX_PRECISION_FLOAT_LAYER(l.real_type)) { // to = l.output_float_gpu
            if (inputLayer.real_type == REAL) { // from = inputLayer.output_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_gpu + j*input_size, l.output_float_gpu + offset + j*l.outputs, input_size);
            } else if (inputLayer.real_type == FLOAT) { // from = inputLayer.output_float_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_float_gpu + j*input_size, l.output_float_gpu + offset + j*l.outputs, input_size);
            } else if (inputLayer.real_type == HALF) { // from = inputLayer.output_half_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_half_gpu + j*input_size, l.output_float_gpu + offset + j*l.outputs, input_size);
            }
        } else { // to = l.output_gpu
            if (inputLayer.real_type == REAL) { // from = inputLayer.output_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_gpu + j*input_size, l.output_gpu + offset + j*l.outputs, input_size);
            } else if (inputLayer.real_type == FLOAT) { // from = inputLayer.output_float_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_float_gpu + j*input_size, l.output_gpu + offset + j*l.outputs, input_size);
            } else if (inputLayer.real_type == HALF) { // from = inputLayer.output_half_gpu
                for (j = 0; j < l.batch; ++j)
                    generic_copy_array_gpu(inputLayer.output_half_gpu + j*input_size, l.output_gpu + offset + j*l.outputs, input_size);
            }
        }
        
        offset += input_size;
    }
}

void backward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        real *delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        for(j = 0; j < l.batch; ++j){
            axpy_gpu(input_size, 1, l.delta_gpu + offset + j*l.outputs, 1, delta + j*input_size, 1);
        }
        offset += input_size;
    }
}
#endif
