// include/relu.h
#ifndef RELU_H
#define RELU_H

#ifdef __cplusplus
extern "C" {
#enfif

/** 
 * @brief CUDA kernel for Relu
 * This kernel performs y = max(0, x)

 * @param input: (N, in_features)
 * @param output : (N, in_features)
 * @param N : number of samples
 * @param in_features : number of input features
*/

__global__ void relu_forward(float *input, float * output, int N, int in_features)

#ifdef __cplusplus
}
#endif

#endif // RELU_H