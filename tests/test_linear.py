## test linear layer results and compare against pytorch

import torch
from torch.utils.cpp_extension import load
import os
import time
import random
# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

linear_layer = load(
    name="linear_layer",
    sources=[os.path.join(project_root, "src/linear_layer.cu")],
    extra_include_paths=[os.path.join(project_root, "include")],  # Add this line
    verbose=True,
)

def test_linear_layer(N, in_features, out_features):
    # Create random input, weight, and bias using PyTorch
    input_tensor = torch.randn(N, in_features).cuda()
    weight_tensor = torch.randn(in_features, out_features).cuda()
    
    ## set bias to zero
    bias_tensor = torch.randn(out_features).cuda()

    ## pytorch forward
    torch_linear = torch.nn.Linear(in_features, out_features, bias=True)

    torch_linear.weight = torch.nn.Parameter(weight_tensor.t())
    torch_linear.bias = torch.nn.Parameter(bias_tensor)
    

    # Measure time for PyTorch forward pass
    start_time = time.time()
    torch_output = torch_linear(input_tensor)
    pytorch_time = time.time() - start_time
    # print(f"PyTorch forward pass time: {pytorch_time:.6f} seconds")

    # Measure time for CUDA forward pass
    start_time = time.time()
    cuda_output = linear_layer.forward(input_tensor, weight_tensor, bias_tensor)
    cuda_time = time.time() - start_time
    # print(f"CUDA forward pass time: {cuda_time:.6f} seconds")

    # Compare which takes longer
    if pytorch_time > cuda_time:
        print("CUDA forward pass is faster than PyTorch forward pass.")
    elif pytorch_time < cuda_time:
        print("PyTorch forward pass is faster than CUDA forward pass.")
    else:
        print("Both forward passes took the same amount of time.")

    ## compare
    if not torch.allclose(torch_output, cuda_output, atol=1e-6):
        ## print absolute difference
        print("Absolute difference: ", torch.abs(torch_output - cuda_output).max())


if __name__ == "__main__":
    ## test linear layer with small 
    ## test in 100 times with different input size
    for i in range(100):
        N = random.randint(1, 1000)
        in_features = random.randint(1, 1000)
        out_features = random.randint(1, 1000)
        test_linear_layer(N, in_features, out_features)
