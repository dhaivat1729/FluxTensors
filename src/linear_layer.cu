/*
    Linear Layer in CUDA 
    y = xA^T + b 

    Dimensions: 
    Input: (N, in_features)
    Output: (N, out_features)
    Weight: (in_features, out_features)
    Bias: (out_features)
*/

#include "linear_layer.h"
#include <iostream>



/** 

    * CUDA kernel to compute forward pass of linear layer

    @param input: (N, in_features)
    @param output: (N, out_features)
    @param weight: (in_features, out_features)
    @param bias: (out_features)
    @param N: number of samples
    @param in_features: number of input features
    @param out_features: number of output features
 */

 __global__ void linear_layer_forward(float* input, float* output, float* weight, float* bias, int N, int in_features, int out_features) {

    // Calculate the row index for the current thread
    // blockIdx.x gives the index of the block in the x dimension
    // blockDim.x gives the number of threads in each block in the x dimension
    // threadIdx.x gives the index of the thread within the block in the x dimension
    // The global row index is computed by multiplying the block index by the block size and adding the thread index
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the column index for the current thread
    // blockIdx.y gives the index of the block in the y dimension
    // blockDim.y gives the number of threads in each block in the y dimension
    // threadIdx.y gives the index of the thread within the block in the y dimension
    // The global column index is computed similarly to the row index
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the calculated row and column indices are within the bounds of the output dimensions
    if (row < N && col < out_features) {
        // Initialize a variable to accumulate the sum for the dot product
        float sum = 0.0f;

        // Perform the dot product between the input row and the weight column
        // Loop over each feature in the input
        for (int i = 0; i < in_features; i++) {
            // Access the input element at the current row and feature index
            // Access the weight element at the current feature index and output column index
            // Multiply the two elements and accumulate the result in sum
            sum += input[row * in_features + i] * weight[i * out_features + col];
        }
        
        // Store the computed sum in the output array
        // The output is calculated by adding the bias for the current output column
        output[row * out_features + col] = sum + bias[col];
    }
 }

 /*
 Example of how to invoke the CUDA kernel
 
 int N = 1000;
 int in_features = 100;
 int out_features = 100;
 
 float *input, *output, *weight, *bias;
 cudaMalloc((void**)&input, N * in_features * sizeof(float));
 cudaMalloc((void**)&output, N * out_features * sizeof(float));
 cudaMalloc((void**)&weight, in_features * out_features * sizeof(float));
 cudaMalloc((void**)&bias, out_features * sizeof(float));


 // Launch the CUDA kernel
 int numBlocks = (N + 1024 - 1) / 1024;
 int numThreads = 1024;
 linear_layer_forward<<<numBlocks, numThreads>>>(input, output, weight, bias, N, in_features, out_features);

 // Wait for the kernel to finish
 cudaDeviceSynchronize();

 // Free the allocated memory
 

 
 */
