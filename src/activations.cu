/*
    Activation Functions in CUDA
    This file contains the implementation of various activation functions using CUDA.
    The activation functions implemented are:
    - ReLU
    - Sigmoid
    - Tanh
    - Softmax
*/
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "activations.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////// ReLU forward pass ///////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CUDA kernel for forward pass of ReLU activation function
__global__ void relu_forward_kernel(const float* input, float* output, int num_datapoints) {


    // For ReLU, input can be of arbirary dimensions. 
    // We simply iterate over each datapoint and apply ReLU function

    // let's calculate the index of the datapoint we are processing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // check if the idx is within the bounds of the input tensor
    if (idx < num_datapoints){
        // apply ReLU function using ternary operator
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

// CUDA implemtation of ReLU forward pass
torch::Tensor relu_forward_cuda(torch::Tensor input){

    // Here we have to iterate through size of input tensor to calculate total number of datapoints
    num_datapoints = 1
    for (int i = 0; i < input.dim(); i++){
        // This way we can received arbitrary shaped tensors    
        num_datapoints *= input.size(i);
    }

    // Create an output tensor of the same size as the input tensor
    auto output = torch::empty_like(input);

    // Define the number of threads per block (16x16)
    const dim3 threads(16, 16);

    // Calculate the number of blocks needed for the grid
    const dim3 blocks((num_datapoints + threads.x - 1) / threads.x);

    // Launch the CUDA kernel for the forward pass
    relu_forward_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), // pointer to the input data
        output.data_ptr<float>(), // pointer to the output data
        num_datapoints // number of datapoints
    );

    // Return the output tensor
    return output;
}

// C++ interfaace for the ReLU forward pass
torch::Tensor relu_forward(torch::Tensor input){
    return relu_forward_cuda(input);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////// Sigmoid forward pass ////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// CUDA kernel for foward pass of sigmoid activation function
__global__ void sigmoid_forward_kernel(const float* input, float* output, int num_datapoints){

    // let's calculate the index of the datapoint we are processing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // check if the idx is within the bounds of the input tensor
    if (idx < num_datapoints){
        // apply sigmoid function using exp function
        // Use the fast math version of exp for better performance
        output[idx] = __fdividef(1.0f, 1.0f + __expf(-input[idx]));
    }
}



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// PyBind11 module definition for the relu layer extension
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("forward", &relu_forward, "ReLU forward (CUDA)");
}



    
