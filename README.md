# FluxTensors
Autodiff engine like PyTorch written in C and Cuda with Python APIs - for my personal learning and development
## Basic Layers in CUDA

This section provides an overview of the basic layers implemented in CUDA for the FluxTensors autodiff engine. Understanding these layers is crucial for grasping how automatic differentiation is handled in frameworks similar to PyTorch.

### Overview

The basic layers include:

1. **Linear Layer**: A fully connected layer that applies a linear transformation to the input.
2. **ReLU Activation**: A non-linear activation function that introduces non-linearity into the model.
3. **Softmax Layer**: A layer that converts logits to probabilities, often used in classification tasks.

### Implementation

Each layer is implemented as a CUDA kernel, allowing for efficient computation on GPUs. The kernels are designed to be simple and modular, making it easy to extend and modify them for learning purposes.

### Usage

To use these layers, you will need to initialize them in your CUDA environment and call the respective kernels during the forward and backward passes of your model. 

### Example

Here is a simple example of how to define and use a linear layer in your model:
