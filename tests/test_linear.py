## This script tests the performance of a custom CUDA linear layer implementation
## against PyTorch's built-in linear layer. It measures the output differences
## and execution times for both implementations.

import torch
from torch.utils.cpp_extension import load
import os
import time
import random

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load the custom CUDA linear layer implementation
linear_layer = load(
    name="linear_layer",
    sources=[os.path.join(project_root, "src/linear_layer.cu")],
    extra_include_paths=[os.path.join(project_root, "include")],  # Include additional headers
    verbose=True,
)

def test_linear_layer(N, in_features, out_features):
    """
    Test the linear layer implementation by comparing the output of the 
    custom CUDA implementation with PyTorch's built-in linear layer.

    Parameters:
    N (int): Number of input samples
    in_features (int): Number of input features
    out_features (int): Number of output features

    Returns:
    max_diff (float): Maximum difference between the outputs of PyTorch and CUDA
    pytorch_fast (bool): True if PyTorch implementation is faster
    cuda_fast (bool): True if CUDA implementation is faster
    """
    # Create random input, weight, and bias using PyTorch
    input_tensor = torch.randn(N, in_features).cuda()  # Input tensor on GPU
    weight_tensor = torch.randn(in_features, out_features).cuda()  # Weight tensor on GPU
    
    # Set bias to random values
    bias_tensor = torch.randn(out_features).cuda()  # Bias tensor on GPU

    # Create a PyTorch linear layer
    torch_linear = torch.nn.Linear(in_features, out_features, bias=True)
    torch_linear.weight = torch.nn.Parameter(weight_tensor.t())  # Set weight
    torch_linear.bias = torch.nn.Parameter(bias_tensor)  # Set bias

    # Measure time for PyTorch forward pass
    start_time = time.time()
    torch_output = torch_linear(input_tensor)  # Forward pass
    pytorch_time = time.time() - start_time  # Calculate elapsed time

    # Measure time for CUDA forward pass
    start_time = time.time()
    cuda_output = linear_layer.forward(input_tensor, weight_tensor, bias_tensor)  # Forward pass
    cuda_time = time.time() - start_time  # Calculate elapsed time

    # Determine which implementation is faster
    pytorch_fast = pytorch_time < cuda_time
    cuda_fast = cuda_time < pytorch_time

    # Calculate the maximum difference between the outputs
    max_diff = torch.abs(torch_output - cuda_output).max()

    return max_diff, pytorch_fast, cuda_fast

if __name__ == "__main__":
    ## Test the linear layer with multiple random input sizes
    test_times = 1000  # Number of tests to run
    print("="*100)
    print(f"Testing linear layer with {test_times} random input sizes. Comparing CUDA and PyTorch outputs.")
    print("="*100)

    # Record maximum differences in output
    max_diffs = []
    # Record if CUDA or PyTorch is faster
    pytorch_fasts = []
    cuda_fasts = []

    # Run tests
    for i in range(test_times):
        # Randomly generate input sizes
        N = random.randint(1, 1000)  # Number of input samples
        in_features = random.randint(1, 1000)  # Input feature size
        out_features = random.randint(1, 1000)  # Output feature size
        
        # Test the linear layer
        max_diff, pytorch_fast, cuda_fast = test_linear_layer(N, in_features, out_features)

        # Record results
        max_diffs.append(max_diff)
        pytorch_fasts.append(pytorch_fast)
        cuda_fasts.append(cuda_fast)

    # Calculate overall results
    max_difference = max(max_diffs).item()  # Maximum difference across all tests
    cuda_faster_count = sum(cuda_fasts)  # Count of times CUDA was faster
    pytorch_faster_count = sum(pytorch_fasts)  # Count of times PyTorch was faster

    # Print summary of results
    print("\n" + "="*50)  # Separator for readability
    print(f"Max difference between PyTorch and CUDA across {test_times} tests: {max_difference}")
    print(f"Custom CUDA code was faster for {cuda_faster_count} times")
    print(f"PyTorch code was faster for {pytorch_faster_count} times")
    print("="*50 + "\n")  # Another separator
