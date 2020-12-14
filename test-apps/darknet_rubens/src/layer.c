#include "layer.h"
#include "cuda.h"

#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if(l.rand)           free(l.rand);
#ifdef GPU
        if(l.rand_gpu)             cuda_free(l.rand_gpu);
#endif
        return;
    }
    if(l.cweights)           free(l.cweights);
    if(l.indexes)            free(l.indexes);
    if(l.input_layers)       free(l.input_layers);
    if(l.input_sizes)        free(l.input_sizes);
    if(l.map)                free(l.map);
    if(l.rand)               free(l.rand);
    if(l.cost)               free(l.cost);
    if(l.state)              free(l.state);
    if(l.prev_state)         free(l.prev_state);
    if(l.forgot_state)       free(l.forgot_state);
    if(l.forgot_delta)       free(l.forgot_delta);
    if(l.state_delta)        free(l.state_delta);
    if(l.concat)             free(l.concat);
    if(l.concat_delta)       free(l.concat_delta);
    if(l.binary_weights)     free(l.binary_weights);
    if(l.biases)             free(l.biases);
    if(l.bias_updates)       free(l.bias_updates);
    if(l.scales)             free(l.scales);
    if(l.scale_updates)      free(l.scale_updates);
    if(l.weights)            free(l.weights);
    if(l.weight_updates)     free(l.weight_updates);
    if(l.delta)              free(l.delta);
    if(l.output)             free(l.output);
    if(l.squared)            free(l.squared);
    if(l.norms)              free(l.norms);
    if(l.spatial_mean)       free(l.spatial_mean);
    if(l.mean)               free(l.mean);
    if(l.variance)           free(l.variance);
    if(l.mean_delta)         free(l.mean_delta);
    if(l.variance_delta)     free(l.variance_delta);
    if(l.rolling_mean)       free(l.rolling_mean);
    if(l.rolling_variance)   free(l.rolling_variance);
    if(l.x)                  free(l.x);
    if(l.x_norm)             free(l.x_norm);
    if(l.m)                  free(l.m);
    if(l.v)                  free(l.v);
    if(l.z_cpu)              free(l.z_cpu);
    if(l.r_cpu)              free(l.r_cpu);
    if(l.h_cpu)              free(l.h_cpu);
    if(l.binary_input)       free(l.binary_input);

#if MIX_PRECISION_SUPPORT == FLOAT
    if (l.output_float)             free(l.output_float);
    if (l.delta_float)              free(l.delta_float);
    if (l.weights_float)            free(l.weights_float);
    if (l.weight_updates_float)     free(l.weight_updates_float);
    if (l.biases_float)             free(l.biases_float);
    if (l.bias_updates_float)       free(l.bias_updates_float);
    if (l.binary_weights_float)     free(l.binary_weights_float);
    if (l.scales_float)             free(l.scales_float);
    if (l.scale_updates_float)      free(l.scale_updates_float);
    if (l.binary_input_float)       free(l.binary_input_float);
    if (l.mean_float)               free(l.mean_float);
    if (l.mean_delta_float)         free(l.mean_delta_float);
    if (l.variance_float)           free(l.variance_float);
    if (l.variance_delta_float)     free(l.variance_delta_float);
    if (l.rolling_mean_float)       free(l.rolling_mean_float);
    if (l.rolling_variance_float)   free(l.rolling_variance_float);
    if (l.x_float)                  free(l.x_float);
    if (l.x_norm_float)             free(l.x_norm_float);
    if (l.m_float)                  free(l.m_float);
    if (l.v_float)                  free(l.v_float);
    if (l.bias_m_float)             free(l.bias_m_float);
    if (l.scale_m_float)            free(l.scale_m_float);
    if (l.bias_v_float)             free(l.bias_v_float);
    if (l.scale_v_float)            free(l.scale_v_float);
#elif MIX_PRECISION_SUPPORT == HALF
    if (l.output_half)             free(l.output_half);
    if (l.delta_half)              free(l.delta_half);
    if (l.weights_half)            free(l.weights_half);
    if (l.weight_updates_half)     free(l.weight_updates_half);
    if (l.biases_half)             free(l.biases_half);
    if (l.bias_updates_half)       free(l.bias_updates_half);
    if (l.binary_weights_half)     free(l.binary_weights_half);
    if (l.scales_half)             free(l.scales_half);
    if (l.scale_updates_half)      free(l.scale_updates_half);
    if (l.binary_input_half)       free(l.binary_input_half);
    if (l.mean_half)               free(l.mean_half);
    if (l.mean_delta_half)         free(l.mean_delta_half);
    if (l.variance_half)           free(l.variance_half);
    if (l.variance_delta_half)     free(l.variance_delta_half);
    if (l.rolling_mean_half)       free(l.rolling_mean_half);
    if (l.rolling_variance_half)   free(l.rolling_variance_half);
    if (l.x_half)                  free(l.x_half);
    if (l.x_norm_half)             free(l.x_norm_half);
    if (l.m_half)                  free(l.m_half);
    if (l.v_half)                  free(l.v_half);
    if (l.bias_m_half)             free(l.bias_m_half);
    if (l.scale_m_half)            free(l.scale_m_half);
    if (l.bias_v_half)             free(l.bias_v_half);
    if (l.scale_v_half)            free(l.scale_v_half);
#endif

#ifdef GPU
    if(l.indexes_gpu)           cuda_free((real *)l.indexes_gpu);

    if(l.z_gpu)                   cuda_free(l.z_gpu);
    if(l.r_gpu)                   cuda_free(l.r_gpu);
    if(l.h_gpu)                   cuda_free(l.h_gpu);
    if(l.m_gpu)                   cuda_free(l.m_gpu);
    if(l.v_gpu)                   cuda_free(l.v_gpu);
    if(l.prev_state_gpu)          cuda_free(l.prev_state_gpu);
    if(l.forgot_state_gpu)        cuda_free(l.forgot_state_gpu);
    if(l.forgot_delta_gpu)        cuda_free(l.forgot_delta_gpu);
    if(l.state_gpu)               cuda_free(l.state_gpu);
    if(l.state_delta_gpu)         cuda_free(l.state_delta_gpu);
    if(l.gate_gpu)                cuda_free(l.gate_gpu);
    if(l.gate_delta_gpu)          cuda_free(l.gate_delta_gpu);
    if(l.save_gpu)                cuda_free(l.save_gpu);
    if(l.save_delta_gpu)          cuda_free(l.save_delta_gpu);
    if(l.concat_gpu)              cuda_free(l.concat_gpu);
    if(l.concat_delta_gpu)        cuda_free(l.concat_delta_gpu);
    if(l.binary_input_gpu)        cuda_free(l.binary_input_gpu);
    if(l.binary_weights_gpu)      cuda_free(l.binary_weights_gpu);
    if(l.mean_gpu)                cuda_free(l.mean_gpu);
    if(l.variance_gpu)            cuda_free(l.variance_gpu);
    if(l.rolling_mean_gpu)        cuda_free(l.rolling_mean_gpu);
    if(l.rolling_variance_gpu)    cuda_free(l.rolling_variance_gpu);
    if(l.variance_delta_gpu)      cuda_free(l.variance_delta_gpu);
    if(l.mean_delta_gpu)          cuda_free(l.mean_delta_gpu);
    if(l.x_gpu)                   cuda_free(l.x_gpu);
    if(l.x_norm_gpu)              cuda_free(l.x_norm_gpu);
    if(l.weights_gpu)             cuda_free(l.weights_gpu);
    if(l.weight_updates_gpu)      cuda_free(l.weight_updates_gpu);
    if(l.biases_gpu)              cuda_free(l.biases_gpu);
    if(l.bias_updates_gpu)        cuda_free(l.bias_updates_gpu);
    if(l.scales_gpu)              cuda_free(l.scales_gpu);
    if(l.scale_updates_gpu)       cuda_free(l.scale_updates_gpu);
    if(l.output_gpu)              cuda_free(l.output_gpu);
    if(l.delta_gpu)               cuda_free(l.delta_gpu);
    if(l.rand_gpu)                cuda_free(l.rand_gpu);
    if(l.squared_gpu)             cuda_free(l.squared_gpu);
    if(l.norms_gpu)               cuda_free(l.norms_gpu);

    #if MIX_PRECISION_SUPPORT == FLOAT
        if(l.delta_float_gpu)               cuda_free(l.delta_float_gpu);
        if(l.output_float_gpu)              cuda_free(l.output_float_gpu);
        if(l.m_float_gpu)                   cuda_free(l.m_float_gpu);
        if(l.v_float_gpu)                   cuda_free(l.v_float_gpu);
        if(l.bias_m_float_gpu)              cuda_free(l.bias_m_float_gpu);
        if(l.bias_v_float_gpu)              cuda_free(l.bias_v_float_gpu);
        if(l.scale_m_float_gpu)             cuda_free(l.scale_m_float_gpu);
        if(l.scale_v_float_gpu)             cuda_free(l.scale_v_float_gpu);
        if(l.weights_float_gpu)             cuda_free(l.weights_float_gpu);
        if(l.weight_updates_float_gpu)      cuda_free(l.weight_updates_float_gpu);
        if(l.biases_float_gpu)              cuda_free(l.biases_float_gpu);
        if(l.bias_updates_float_gpu)        cuda_free(l.bias_updates_float_gpu);
        if(l.binary_weights_float_gpu)      cuda_free(l.binary_weights_float_gpu);
        if(l.binary_input_float_gpu)        cuda_free(l.binary_input_float_gpu);
        if(l.mean_float_gpu)                cuda_free(l.mean_float_gpu);
        if(l.variance_float_gpu)            cuda_free(l.variance_float_gpu);
        if(l.rolling_mean_float_gpu)        cuda_free(l.rolling_mean_float_gpu);
        if(l.rolling_variance_float_gpu)    cuda_free(l.rolling_variance_float_gpu);
        if(l.mean_delta_float_gpu)          cuda_free(l.mean_delta_float_gpu);
        if(l.variance_delta_float_gpu)      cuda_free(l.variance_delta_float_gpu);
        if(l.scales_float_gpu)              cuda_free(l.scales_float_gpu);
        if(l.scale_updates_float_gpu)       cuda_free(l.scale_updates_float_gpu);
        if(l.x_float_gpu)                   cuda_free(l.x_float_gpu);
        if(l.x_norm_float_gpu)              cuda_free(l.x_norm_float_gpu);
    #elif MIX_PRECISION_SUPPORT == HALF
        if(l.delta_half_gpu)               cuda_free(l.delta_half_gpu);
        if(l.output_half_gpu)              cuda_free(l.output_half_gpu);
        if(l.m_half_gpu)                   cuda_free(l.m_half_gpu);
        if(l.v_half_gpu)                   cuda_free(l.v_half_gpu);
        if(l.bias_m_half_gpu)              cuda_free(l.bias_m_half_gpu);
        if(l.bias_v_half_gpu)              cuda_free(l.bias_v_half_gpu);
        if(l.scale_m_half_gpu)             cuda_free(l.scale_m_half_gpu);
        if(l.scale_v_half_gpu)             cuda_free(l.scale_v_half_gpu);
        if(l.weights_half_gpu)             cuda_free(l.weights_half_gpu);
        if(l.weight_updates_half_gpu)      cuda_free(l.weight_updates_half_gpu);
        if(l.biases_half_gpu)              cuda_free(l.biases_half_gpu);
        if(l.bias_updates_half_gpu)        cuda_free(l.bias_updates_half_gpu);
        if(l.binary_weights_half_gpu)      cuda_free(l.binary_weights_half_gpu);
        if(l.binary_input_half_gpu)        cuda_free(l.binary_input_half_gpu);
        if(l.mean_half_gpu)                cuda_free(l.mean_half_gpu);
        if(l.variance_half_gpu)            cuda_free(l.variance_half_gpu);
        if(l.rolling_mean_half_gpu)        cuda_free(l.rolling_mean_half_gpu);
        if(l.rolling_variance_half_gpu)    cuda_free(l.rolling_variance_half_gpu);
        if(l.mean_delta_half_gpu)          cuda_free(l.mean_delta_half_gpu);
        if(l.variance_delta_half_gpu)      cuda_free(l.variance_delta_half_gpu);
        if(l.scales_half_gpu)              cuda_free(l.scales_half_gpu);
        if(l.scale_updates_half_gpu)       cuda_free(l.scale_updates_half_gpu);
        if(l.x_half_gpu)                   cuda_free(l.x_half_gpu);
        if(l.x_norm_half_gpu)              cuda_free(l.x_norm_half_gpu);
    #endif
#endif
}
