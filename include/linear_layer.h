// src/linear_layer.h
#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#ifdef __cplusplus
extern "C" { 
#endif

/**
 * @brief CUDA kernel to compute forward pass of a linear layer
 *
 * This kernel perform y = xA^T + b for a linear layer
 * 
 * @param input: (N, in_features) 
 * @param output: (N, out_features)
 * @param weight: (in_features, out_features)
 * @param bias: (out_features)
 * @param N : number of samples
 * @param in_features : number of input features
 * @param out_features : number of output features
 */

 __global__ void linear_layer_forward(float* input, float* output, float* weight, float* bias, int N, int in_features, int out_features);

 #ifdef __cplusplus
 }
 #endif

#endif // LINEAR_LAYER_H