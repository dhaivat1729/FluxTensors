/*
    Linear Layer Implementation in CUDA 
    This implementation computes the output of a linear layer using the formula:
    y = xA^T + b 

    Dimensions: 
    Input: (N, in_features)      // N: number of samples, in_features: number of input features
    Output: (N, out_features)     // out_features: number of output features
    Weight: (in_features, out_features) // Weight matrix for the linear transformation
    Bias: (out_features)          // Bias vector added to the output
*/

#include <torch/extension.h>      // Include PyTorch extension header for CUDA
#include <cuda.h>                 // Include CUDA header for CUDA functions
#include <cuda_runtime.h>         // Include CUDA runtime header for CUDA operations
#include "linear_layer.h"         // Include custom header for linear layer definitions

// CUDA kernel for forward pass of the linear layer
__global__ void linear_layer_forward_kernel(
    const float* input,          // Pointer to input tensor
    const float* weight,         // Pointer to weight matrix
    const float* bias,           // Pointer to bias vector
    float* output,               // Pointer to output tensor
    int num_samples,             // Number of input samples (N)
    int num_input_features,      // Number of input features (in_features)
    int num_output_features)     // Number of output features (out_features)
{
    // Calculate the row index for the output tensor based on block and thread indices
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate the column index for the output tensor based on block and thread indices
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the calculated row and column indices are within the bounds of the output tensor
    if (row < num_samples && col < num_output_features) {
        float sum = 0.0f; // Initialize sum for the current output element
        // Perform the dot product between the input row and the weight column
        for (int i = 0; i < num_input_features; i++) {
            // Remove typeid operations as they are not allowed in device code
            sum += input[row * num_input_features + i] * weight[i * num_output_features + col];
        // // Print size of variables in bytes
        // if (row == 0 && col == 0) {
        //     printf("Size of input element: %lu bytes\n", sizeof(input[0]));
        //     printf("Size of weight element: %lu bytes\n", sizeof(weight[0]));
        //     printf("Size of sum: %lu bytes\n", sizeof(sum));
        // }
        }
        // Store the computed value in the output tensor, adding the bias
        output[row * num_output_features + col] = sum + bias[col];
    }
}

// CUDA implementation of the linear layer forward pass
torch::Tensor linear_layer_forward_cuda(
    torch::Tensor input,         // Input tensor
    torch::Tensor weight,        // Weight tensor
    torch::Tensor bias)          // Bias tensor
{
    // Get the number of samples from the input tensor
    const auto num_samples = input.size(0);
    // Get the number of input features from the input tensor
    const auto num_input_features = input.size(1);
    // Get the number of output features from the weight tensor
    const auto num_output_features = weight.size(1);

    // Create an empty output tensor with the appropriate size and type
    auto output = torch::empty({num_samples, num_output_features}, input.options());

    // Define the number of threads per block (16x16)
    const dim3 threads(16, 16);
    // Calculate the number of blocks needed for the grid
    const dim3 blocks((num_samples + threads.x - 1) / threads.x, 
                      (num_output_features + threads.y - 1) / threads.y);

    // Launch the CUDA kernel for the forward pass
    linear_layer_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),     // Pointer to input data
        weight.data_ptr<float>(),    // Pointer to weight data
        bias.data_ptr<float>(),      // Pointer to bias data
        output.data_ptr<float>(),    // Pointer to output data
        num_samples,                 // Number of input samples
        num_input_features,          // Number of input features
        num_output_features          // Number of output features
    );

    // Return the computed output tensor
    return output;
}

// C++ interface for the linear layer forward pass
torch::Tensor linear_layer_forward(
    torch::Tensor input,          // Input tensor
    torch::Tensor weight,         // Weight tensor
    torch::Tensor bias)           // Bias tensor
{
    // Call the CUDA implementation of the forward pass
    return linear_layer_forward_cuda(input, weight, bias);
}

// PyBind11 module definition for the linear layer extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Expose the forward function to Python
    m.def("forward", &linear_layer_forward, "Linear layer forward (CUDA)");
}