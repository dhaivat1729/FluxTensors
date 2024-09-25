/*
    ReLU activation function in CUDA
    y = max(0, x)
*/

#include "relu.h"

/**
 * @param input: (N, in_features)
 * @param output: (N, out_features)
 * @param N: number of samples
 * @param in_features: number of input features
 */

__global__ void 2d_relu_forward(float* input, float* output, int N, int in_features) {

    // since this is 2D relu, let's calculate row and column index
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the index is within the valid range
    if (row < N && col < in_features) {
        // Apply ReLU activation function
        output[row * in_features + col] = max(0, input[row * in_features + col]);
    }
}
