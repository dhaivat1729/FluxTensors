#ifndef LINEAR_LAYER_H // Include guard to prevent multiple inclusions of this header file
#define LINEAR_LAYER_H

#include <torch/extension.h> // Include the PyTorch extension header for CUDA support

// Function declaration for the CUDA implementation of the linear layer forward pass
// This function computes the output of a linear layer given the input, weight, and bias tensors.
// Parameters:
//   - input: A tensor containing the input data (shape: [N, in_features])
//   - weight: A tensor containing the weight matrix (shape: [in_features, out_features])
//   - bias: A tensor containing the bias vector (shape: [out_features])
// Returns:
//   - A tensor containing the output data (shape: [N, out_features])
torch::Tensor linear_layer_forward_cuda(
    torch::Tensor input,      // Input tensor for the linear layer
    torch::Tensor weight,     // Weight tensor for the linear transformation
    torch::Tensor bias);      // Bias tensor to be added to the output

// Function declaration for the C++ interface of the linear layer forward pass
// This function serves as a wrapper to call the CUDA implementation.
// Parameters are the same as for the CUDA function.
// Returns the output tensor computed by the CUDA function.
torch::Tensor linear_layer_forward(
    torch::Tensor input,      // Input tensor for the linear layer
    torch::Tensor weight,     // Weight tensor for the linear transformation
    torch::Tensor bias);      // Bias tensor to be added to the output

#endif // LINEAR_LAYER_H // End of the include guard